"""
Unit tests for the FastAPI assessment API.
"""

import io
from pathlib import Path
import sys
import unittest
from PIL import Image
import fitz  # PyMuPDF
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import app


class TestFastAPI(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        """Verify GET /api/health returns 200 OK."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")

    def test_assess_property_with_agreement(self):
        """
        Verify POST /api/assess executes end-to-end multimodal pipeline with both
        images and rental agreement PDF.
        """
        # 1. Create a dummy synthetic image (PNG bytes)
        img = Image.new("RGB", (320, 320), color=(200, 200, 200))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        # 2. Create a small synthetic PDF with realistic clauses
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 70),
            "1. Security Deposit: The tenant shall deposit Rs 50,000 as security deposit.\n"
            "2. Maintenance: Tenant shall pay monthly maintenance charges for water and electricity.\n"
            "3. Termination: Either party may terminate with 30 days written notice.",
            fontsize=12,
        )
        pdf_bytes = doc.tobytes()
        doc.close()

        # 3. Call POST /api/assess
        files = [
            ("images", ("test_room.png", img_bytes, "image/png")),
            ("agreement_pdf", ("lease.pdf", pdf_bytes, "application/pdf")),
        ]
        res = self.client.post("/api/assess", files=files)

        self.assertEqual(res.status_code, 200, f"API returned error: {res.text}")
        data = res.json()

        # 4. Verify all required keys in response
        required_keys = [
            "property_health_score",
            "total_risk",
            "visual_risk",
            "agreement_risk",
            "regulatory_risk",
            "detected_defects",
            "classified_clauses",
            "regulatory_evidence",
            "recommendations",
            "audit_id",
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Missing key '{key}' in response")

        self.assertGreaterEqual(data["property_health_score"], 0.0)
        self.assertLessEqual(data["property_health_score"], 100.0)
        self.assertIsInstance(data["detected_defects"], list)
        self.assertGreater(len(data["classified_clauses"]), 0)
        self.assertGreater(len(data["regulatory_evidence"]), 0)
        self.assertGreater(len(data["recommendations"]), 0)

    def test_assess_property_images_only(self):
        """
        Verify POST /api/assess when agreement_pdf is NOT provided:
        - PDF extraction is skipped
        - Agreement clause classification is skipped
        - agreement_risk is 0.0
        - classified_clauses is []
        - YOLO, regulatory RAG, and risk fusion continue normally
        """
        img = Image.new("RGB", (320, 320), color=(180, 180, 180))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        files = [
            ("images", ("room_only.png", img_bytes, "image/png")),
        ]
        res = self.client.post("/api/assess", files=files)

        self.assertEqual(res.status_code, 200, f"API returned error: {res.text}")
        data = res.json()

        self.assertEqual(data["agreement_risk"], 0.0)
        self.assertEqual(data["classified_clauses"], [])
        self.assertGreaterEqual(data["property_health_score"], 0.0)
        self.assertLessEqual(data["property_health_score"], 100.0)
        self.assertIsInstance(data["detected_defects"], list)
        self.assertGreater(len(data["regulatory_evidence"]), 0)
        self.assertGreater(len(data["regulatory_evidence"]), 0)
        self.assertGreater(len(data["recommendations"]), 0)
        self.assertIn("landlord_questions", data)
        self.assertGreater(len(data["landlord_questions"]), 0)

    def test_regulatory_chat_grounded(self):
        """Verify POST /api/chat retrieves authoritative grounded regulatory evidence."""
        res = self.client.post("/api/chat", json={"query": "Is a 3-month security deposit allowed?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("answer", data)
        self.assertIn("evidence", data)
        self.assertTrue(data.get("grounded", False))
        self.assertGreater(len(data["evidence"]), 0)
        top_ev = data["evidence"][0]
        self.assertIn("source", top_ev)
        self.assertIn("act", top_ev)
        self.assertIn("section", top_ev)
        self.assertIn("passage", top_ev)
        self.assertIn("retrieval_score", top_ev)

    def test_regulatory_chat_insufficient_evidence(self):
        """Verify POST /api/chat handles irrelevant queries gracefully."""
        res = self.client.post("/api/chat", json={"query": "quantum entanglement astrophysics nebula"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["answer"], "Insufficient evidence in the available regulatory sources.")
        self.assertEqual(data["evidence"], [])
        self.assertFalse(data.get("grounded", True))


if __name__ == "__main__":
    unittest.main()

