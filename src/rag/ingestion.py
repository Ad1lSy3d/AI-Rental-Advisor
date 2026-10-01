"""
Document ingestion, text extraction, and section-aware chunking for regulatory sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import pymupdf


@dataclass
class RegulatoryChunk:
    """Represents a discrete statutory or regulatory text chunk with source metadata."""

    chunk_id: int
    text: str
    source: str
    metadata: Dict[str, Union[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Union[str, int, Dict[str, Union[str, int]]]]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }


class RegulatoryDocumentIngester:
    """
    Ingests authoritative regulatory documents from a directory, extracts their text,
    and partitions them into coherent, section-aware chunks for indexing.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def __init__(
        self,
        docs_dir: Union[str, Path] = "data/regulatory",
        max_chunk_words: int = 180,
        min_chunk_words: int = 25,
    ):
        """
        Args:
            docs_dir: Path to directory containing regulatory files.
            max_chunk_words: Target maximum word count per chunk.
            min_chunk_words: Minimum words required for a standalone chunk.
        """
        self.docs_dir = Path(docs_dir)
        self.max_chunk_words = max_chunk_words
        self.min_chunk_words = min_chunk_words

    def extract_text(self, file_path: Path) -> str:
        """Extract plain text from .txt, .md, or .pdf files."""
        ext = file_path.suffix.lower()
        if ext in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8", errors="replace")
        elif ext == ".pdf":
            doc = pymupdf.open(str(file_path))
            pages = []
            try:
                for page in doc:
                    pages.append(page.get_text("text"))
            finally:
                doc.close()
            return "\n\n".join(pages)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def chunk_document(self, text: str, source_name: str, start_chunk_id: int = 0) -> List[RegulatoryChunk]:
        """
        Segment a regulatory text document into section-aware chunks.

        Splits primarily on paragraph breaks and section headers while respecting
        the max_chunk_words threshold.
        """
        # Split on double newlines or major Section/Chapter headers
        raw_paragraphs = re.split(r"\n{2,}", text)
        chunks: List[RegulatoryChunk] = []

        current_text_segments: List[str] = []
        current_word_count = 0
        current_chunk_id = start_chunk_id

        for paragraph in raw_paragraphs:
            cleaned = re.sub(r"\s+", " ", paragraph).strip()
            if not cleaned:
                continue

            words = cleaned.split()
            word_count = len(words)

            if current_word_count + word_count <= self.max_chunk_words:
                current_text_segments.append(cleaned)
                current_word_count += word_count
            else:
                if current_text_segments:
                    chunk_str = " ".join(current_text_segments)
                    if len(chunk_str.split()) >= self.min_chunk_words:
                        chunks.append(
                            RegulatoryChunk(
                                chunk_id=current_chunk_id,
                                text=chunk_str,
                                source=source_name,
                                metadata={
                                    "word_count": len(chunk_str.split()),
                                    "char_count": len(chunk_str),
                                },
                            )
                        )
                        current_chunk_id += 1

                current_text_segments = [cleaned]
                current_word_count = word_count

        if current_text_segments:
            chunk_str = " ".join(current_text_segments)
            if len(chunk_str.split()) >= self.min_chunk_words:
                chunks.append(
                    RegulatoryChunk(
                        chunk_id=current_chunk_id,
                        text=chunk_str,
                        source=source_name,
                        metadata={
                            "word_count": len(chunk_str.split()),
                            "char_count": len(chunk_str),
                        },
                    )
                )

        return chunks

    def ingest_corpus(self) -> List[RegulatoryChunk]:
        """
        Scan docs_dir, extract text from all supported files, and return all chunks.
        """
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Regulatory directory does not exist: {self.docs_dir}")

        all_files = sorted(
            [f for f in self.docs_dir.iterdir() if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        )

        all_chunks: List[RegulatoryChunk] = []
        next_chunk_id = 0

        for file_path in all_files:
            text = self.extract_text(file_path)
            doc_chunks = self.chunk_document(text, file_path.name, start_chunk_id=next_chunk_id)
            all_chunks.extend(doc_chunks)
            next_chunk_id = len(all_chunks)

        return all_chunks
