"""
Regulatory RAG module for legal retrieval against authoritative housing and rental sources.
"""

from typing import Optional
from src.rag.ingestion import RegulatoryChunk, RegulatoryDocumentIngester
from src.rag.retriever import RegulatoryRetriever

_GLOBAL_RETRIEVER: Optional[RegulatoryRetriever] = None


def get_regulatory_retriever(
    docs_dir: str = "data/regulatory",
    force_reload: bool = False,
) -> RegulatoryRetriever:
    """
    Get or initialize a cached singleton instance of RegulatoryRetriever.
    """
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None or force_reload:
        _GLOBAL_RETRIEVER = RegulatoryRetriever.from_directory(docs_dir=docs_dir)
    return _GLOBAL_RETRIEVER


__all__ = [
    "RegulatoryChunk",
    "RegulatoryDocumentIngester",
    "RegulatoryRetriever",
    "get_regulatory_retriever",
]
