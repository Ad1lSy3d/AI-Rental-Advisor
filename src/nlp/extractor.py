"""
Rental agreement text extraction and clause segmentation using PyMuPDF.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Union
import pymupdf


class AgreementClauseExtractor:
    """
    Extracts text from rental agreement PDFs and segments it into meaningful,
    clause-sized sections while filtering out noise and small fragments.
    """

    def __init__(self, min_words: int = 6):
        """
        Args:
            min_words: Minimum number of words required for a segment to be
                       considered a valid clause rather than an orphan fragment.
        """
        self.min_words = min_words

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Collapse multiple whitespace and clean up line wrapping."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_document_title(text: str) -> bool:
        """Check if snippet is a general document title rather than a clause heading."""
        clean = text.strip()
        if re.match(
            r"^(?:residential|commercial)?\s*(?:lease|rental|tenancy)\s*(?:agreement|contract)\.?$",
            clean,
            re.IGNORECASE,
        ):
            return True
        return False

    @staticmethod
    def _is_heading(text: str, max_words: int = 8) -> bool:
        """
        Check if a text snippet represents an unattached section or clause heading.
        """
        clean = text.strip()
        words = clean.split()
        if not words:
            return False

        # Explicit section / clause prefix pattern
        if re.match(
            r"^(?:SECTION|ARTICLE|CLAUSE|\d+[\.\)]|[A-Z]\.)\s+[A-Za-z\s]{2,}\.?$",
            clean,
            re.IGNORECASE,
        ) and len(words) <= max_words:
            return True

        # Short uppercase or title-case lines without terminal punctuation
        if len(words) <= max_words and not clean.endswith((".", ";", ":")):
            return True

        return False

    def _is_ignorable_fragment(self, text: str) -> bool:
        """
        Identify whether a text block is an overly small fragment, page number,
        or signature line that should not be treated as a clause.
        """
        words = text.split()
        if len(words) < self.min_words:
            return True

        # Page numbering or metadata patterns
        if re.match(r"^(?:page\s+\d+(\s+of\s+\d+)?|date|signature)[\s:_.-]*$", text, re.IGNORECASE):
            return True

        # Pure signature / underscore lines
        if re.match(r"^[\s_.-]{5,}$", text):
            return True

        return False

    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> List[str]:
        """
        Extract text from a rental agreement PDF and split it into clauses.

        Args:
            pdf_path: Path to the target rental agreement PDF.

        Returns:
            List of extracted clause text strings.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = pymupdf.open(str(pdf_path))
        raw_blocks: List[str] = []

        try:
            for page in doc:
                # get_text('blocks') returns layout-aware text blocks
                # format: (x0, y0, x1, y1, text, block_no, block_type)
                blocks = page.get_text("blocks")
                for b in blocks:
                    if b[6] == 0:  # block_type 0 = text
                        cleaned = self._clean_whitespace(b[4])
                        if cleaned:
                            raw_blocks.append(cleaned)
        finally:
            doc.close()

        return self.segment_blocks(raw_blocks)

    def segment_blocks(self, blocks: List[str]) -> List[str]:
        """
        Process extracted raw text blocks into distinct clause-sized segments.

        Combines short headings with subsequent text, splits multi-clause blocks
        if numbered, and discards overly small fragments.
        """
        clauses: List[str] = []
        pending_heading = ""

        # Numbered clause pattern: '1. ', 'Section 2:', 'Article 3.', etc.
        clause_split_pattern = re.compile(
            r"(?=(?:\n|^)\s*(?:(?:SECTION|ARTICLE|CLAUSE|\d+[\.\)]|[A-Z]\.)\s+))",
            re.IGNORECASE,
        )

        for block in blocks:
            # Check if multiple clauses exist in a single block
            sub_segments = clause_split_pattern.split(block)
            sub_segments = [self._clean_whitespace(s) for s in sub_segments if self._clean_whitespace(s)]

            if not sub_segments:
                continue

            for seg in sub_segments:
                if self._is_document_title(seg):
                    continue

                if self._is_heading(seg):
                    if pending_heading:
                        pending_heading = f"{pending_heading} - {seg}"
                    else:
                        pending_heading = seg
                    continue

                if pending_heading:
                    combined = f"{pending_heading}. {seg}" if not pending_heading.endswith((".", ":", ";")) else f"{pending_heading} {seg}"
                    pending_heading = ""
                else:
                    combined = seg

                if not self._is_ignorable_fragment(combined):
                    clauses.append(combined)

        return clauses


def extract_clauses_from_pdf(pdf_path: Union[str, Path], min_words: int = 6) -> List[str]:
    """Convenience function to extract clauses from a PDF."""
    extractor = AgreementClauseExtractor(min_words=min_words)
    return extractor.extract_from_pdf(pdf_path)
