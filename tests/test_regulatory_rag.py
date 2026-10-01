"""
Unit tests for the Regulatory RAG component (Ingestion, BM25, SBERT + FAISS, and Hybrid Retrieval).
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.ingestion import RegulatoryDocumentIngester
from src.rag.retriever import RegulatoryRetriever


class TestRegulatoryRAG(unittest.TestCase):
    """Test suite for regulatory document ingestion and hybrid retrieval."""

    @classmethod
    def setUpClass(cls):
        cls.docs_dir = Path("data/regulatory")
        if not cls.docs_dir.exists() or not list(cls.docs_dir.glob("*.txt")):
            raise FileNotFoundError(f"Regulatory documents not found in {cls.docs_dir}")

        cls.ingester = RegulatoryDocumentIngester(docs_dir=cls.docs_dir)
        cls.chunks = cls.ingester.ingest_corpus()
        cls.retriever = RegulatoryRetriever(chunks=cls.chunks)

    def test_ingestion_and_chunking(self):
        """Test that documents are ingested and chunked with valid metadata."""
        self.assertGreaterEqual(len(self.chunks), 10, "Should extract at least 10 chunks across corpus.")

        for chunk in self.chunks:
            self.assertIsInstance(chunk.chunk_id, int)
            self.assertIsInstance(chunk.text, str)
            self.assertIsInstance(chunk.source, str)
            self.assertGreater(len(chunk.text.split()), 10, "Chunk should be non-trivial.")
            self.assertTrue(chunk.source.endswith((".txt", ".md", ".pdf")))

    def test_bm25_retrieval(self):
        """Test lexical BM25 retrieval."""
        results = self.retriever.retrieve_bm25("security deposit", top_k=3)
        self.assertEqual(len(results), 3)

        for res in results:
            self.assertIn("text", res)
            self.assertIn("source", res)
            self.assertIn("retrieval_score", res)
            self.assertGreaterEqual(res["retrieval_score"], 0.0)
            self.assertLessEqual(res["retrieval_score"], 1.0)

    def test_dense_faiss_retrieval(self):
        """Test dense SBERT + FAISS semantic retrieval."""
        results = self.retriever.retrieve_dense("landlord inspection 24 hours advance notice", top_k=3)
        self.assertEqual(len(results), 3)

        for res in results:
            self.assertIn("text", res)
            self.assertIn("source", res)
            self.assertIn("retrieval_score", res)
            self.assertGreaterEqual(res["retrieval_score"], 0.0)
            self.assertLessEqual(res["retrieval_score"], 1.0)

    def test_hybrid_retrieval_schema_and_scores(self):
        """Test hybrid retrieval produces required output schema with valid scores."""
        query = "The Tenant shall pay a refundable security deposit of Rs 50,000 before taking possession."
        results = self.retriever.retrieve(query, top_k=3, alpha=0.5)

        self.assertEqual(len(results), 3)
        for res in results:
            # Required output fields
            self.assertIn("text", res)
            self.assertIn("source", res)
            self.assertIn("retrieval_score", res)

            # Type and range assertions
            self.assertIsInstance(res["text"], str)
            self.assertIsInstance(res["source"], str)
            self.assertIsInstance(res["retrieval_score"], float)
            self.assertGreaterEqual(res["retrieval_score"], 0.0)
            self.assertLessEqual(res["retrieval_score"], 1.0)

        # Top result for security deposit should reference relevant deposit act/provisions
        top_text = results[0]["text"].lower()
        self.assertTrue(
            "security deposit" in top_text or "deposit" in top_text or "rent" in top_text,
            "Top result should discuss security deposit or rent provisions.",
        )

    def test_domain_specific_retrieval_queries(self):
        """Test that different rental clauses retrieve relevant statutory sections."""
        test_queries = [
            ("Landlord entering premises after 24 hours notice", ["model_tenancy_act", "rent_control", "transfer_of_property"]),
            ("Structural roof leaks and external plumbing repairs", ["maintenance", "rent_control", "building_code", "defect_liability"]),
            ("Compensation for holding over possession after lease termination", ["model_tenancy_act", "transfer_of_property", "contract_act"]),
        ]

        for query, expected_sources in test_queries:
            results = self.retriever.retrieve(query, top_k=2)
            self.assertGreater(len(results), 0)
            top_source = results[0]["source"].lower()
            matched = any(exp in top_source for exp in expected_sources)
            self.assertTrue(matched, f"Query '{query}' expected sources matching {expected_sources}, got {top_source}")


if __name__ == "__main__":
    unittest.main()
