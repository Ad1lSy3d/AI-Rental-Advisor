"""
Hybrid regulatory retrieval system combining BM25 and SBERT + FAISS.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.rag.ingestion import RegulatoryChunk, RegulatoryDocumentIngester


class RegulatoryRetriever:
    """
    Retrieves authoritative regulatory provisions relevant to rental agreement clauses.
    Combines lexical BM25 retrieval with dense semantic retrieval using SBERT and FAISS.
    """

    def __init__(
        self,
        chunks: List[RegulatoryChunk],
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the retriever and build BM25 and FAISS indexes.

        Args:
            chunks: List of RegulatoryChunk objects to index.
            embedding_model_name: HuggingFace model identifier for dense embeddings.
        """
        if not chunks:
            raise ValueError("Cannot initialize RegulatoryRetriever with an empty chunk list.")

        self.chunks = chunks
        self.embedding_model_name = embedding_model_name

        # 1. Build Lexical BM25 Index
        self._tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

        # 2. Build Dense SBERT + FAISS Index
        self.encoder = SentenceTransformer(embedding_model_name)
        chunk_texts = [c.text for c in self.chunks]
        embeddings = self.encoder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)

        # Normalize vectors for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(embeddings)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple case-insensitive word tokenization for BM25."""
        return re.findall(r"\w+", text.lower())

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks using pure BM25 lexical matching."""
        tokens = self._tokenize(query)
        raw_scores = np.array(self.bm25.get_scores(tokens))

        if raw_scores.max() > 0:
            norm_scores = raw_scores / raw_scores.max()
        else:
            norm_scores = raw_scores

        top_indices = np.argsort(norm_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "source": chunk.source,
                "retrieval_score": round(float(norm_scores[idx]), 4),
                "chunk_id": chunk.chunk_id,
            })
        return results

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks using dense SBERT + FAISS cosine similarity."""
        query_emb = self.encoder.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_emb)

        scores, indices = self.faiss_index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "source": chunk.source,
                "retrieval_score": round(float(np.clip(score, 0.0, 1.0)), 4),
                "chunk_id": chunk.chunk_id,
            })
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining BM25 and semantic FAISS similarity.

        Args:
            query: The clause text or search query.
            top_k: Number of highest-ranked chunks to return.
            alpha: Weight for BM25 vs dense retrieval:
                   alpha = 1.0 -> Pure BM25
                   alpha = 0.0 -> Pure SBERT/FAISS
                   alpha = 0.5 -> Equal weighting (default)

        Returns:
            List of dictionaries containing:
            - text: Text of the matching statutory chunk
            - source: Source document filename
            - retrieval_score: Combined hybrid confidence score (0.0 to 1.0)
            - bm25_score: Normalized BM25 score
            - dense_score: Dense cosine similarity score
            - chunk_id: Integer index of the chunk
        """
        # 1. Lexical BM25 Scores across all chunks
        query_tokens = self._tokenize(query)
        bm25_raw = np.array(self.bm25.get_scores(query_tokens))
        if bm25_raw.max() > 0:
            bm25_norm = bm25_raw / bm25_raw.max()
        else:
            bm25_norm = bm25_raw

        # 2. Dense Semantic Scores across all chunks
        query_emb = self.encoder.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_emb)
        dense_raw, _ = self.faiss_index.search(query_emb, len(self.chunks))
        dense_scores = dense_raw[0]
        dense_norm = np.clip(dense_scores, 0.0, 1.0)

        # 3. Hybrid Linear Fusion
        hybrid_scores = (alpha * bm25_norm) + ((1.0 - alpha) * dense_norm)

        # 4. Rank and format top-k
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "source": chunk.source,
                "retrieval_score": round(float(hybrid_scores[idx]), 4),
                "bm25_score": round(float(bm25_norm[idx]), 4),
                "dense_score": round(float(dense_norm[idx]), 4),
                "chunk_id": chunk.chunk_id,
            })

        return results

    @classmethod
    def from_directory(
        cls,
        docs_dir: Union[str, Path] = "data/regulatory",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        max_chunk_words: int = 180,
    ) -> RegulatoryRetriever:
        """
        Ingest all regulatory documents from docs_dir and build the retriever.
        """
        ingester = RegulatoryDocumentIngester(docs_dir=docs_dir, max_chunk_words=max_chunk_words)
        chunks = ingester.ingest_corpus()
        return cls(chunks=chunks, embedding_model_name=embedding_model_name)
