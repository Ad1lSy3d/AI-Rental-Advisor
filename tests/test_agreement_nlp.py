"""
Unit test for rental agreement PDF clause extraction and classification pipeline.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf

from src.nlp.classifier import AgreementClauseClassifier, classify_rental_agreement
from src.nlp.extractor import AgreementClauseExtractor, extract_clauses_from_pdf


SAMPLE_AGREEMENT_TEXT = """RESIDENTIAL LEASE AGREEMENT

1. SECURITY DEPOSIT.
The Tenant shall pay a refundable security deposit of $2,500 prior to taking possession of the premises. The deposit shall be returned within thirty days after the end of tenancy, subject to lawful deductions for unpaid rent or physical damage beyond normal wear and tear.

2. QUIET ENJOYMENT AND LANDLORD ACCESS.
The Landlord covenants that the Tenant, upon paying rent and observing agreement terms, shall peaceably and quietly hold and enjoy the leased premises during the tenancy. Landlord shall provide Tenant with at least 24 hours advance written notice prior to entering the dwelling unit for non-emergency inspections or showings.

3. UTILITIES AND MAINTENANCE.
The Tenant shall pay the utilities, sanitation fee, electricity, and building management fee generated from using the leased property. The Landlord shall arrange routine servicing of shared building fixtures.

4. REPAIRS AND DAMAGE.
The Tenant shall promptly notify the Landlord of any plumbing leaks or defects. The Tenant shall be responsible for repairing damage caused by the Tenant's negligence or intentional acts.

5. LATE CHARGES AND PENALTIES.
A late payment fee of $50 will be assessed if rent is not received by the fifth day of the calendar month, with daily late charges of $10 until settled in full. A $35 administrative fee will apply to any returned check.

6. LEASE TERMINATION.
Either party may terminate the tenancy by giving 30 days written notice, subject to applicable law. Upon termination, the Tenant shall vacate the premises and return all keys to the Landlord.

Page 1 of 1
"""


class TestAgreementNLP(unittest.TestCase):
    """Test suite for PDF extraction and clause classification."""

    @classmethod
    def setUpClass(cls):
        cls.models_path = Path("models/agreement_tfidf_logreg.pkl")
        if not cls.models_path.exists():
            raise FileNotFoundError(f"Required model not found at {cls.models_path}")

        # Check if a sample PDF already exists in data/agreement/
        data_agreement_dir = Path("data/agreement")
        existing_pdfs = list(data_agreement_dir.glob("*.pdf"))

        if existing_pdfs:
            cls.test_pdf_path = str(existing_pdfs[0])
            cls.temp_file = None
        else:
            # Create a small valid test PDF using PyMuPDF
            cls.temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            cls.test_pdf_path = cls.temp_file.name

            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text((50, 50), SAMPLE_AGREEMENT_TEXT, fontsize=10)
            doc.save(cls.test_pdf_path)
            doc.close()

    @classmethod
    def tearDownClass(cls):
        if cls.temp_file and os.path.exists(cls.test_pdf_path):
            try:
                os.remove(cls.test_pdf_path)
            except OSError:
                pass

    def test_extractor_extracts_clauses_and_filters_fragments(self):
        """Test that extractor segments clauses and ignores fragments like 'Page 1 of 1'."""
        extractor = AgreementClauseExtractor(min_words=6)
        clauses = extractor.extract_from_pdf(self.test_pdf_path)

        self.assertGreater(len(clauses), 0, "Should extract at least one clause.")

        # Ensure no overly small fragments (< 6 words) were included
        for c in clauses:
            self.assertGreaterEqual(len(c.split()), 6, f"Fragment too small: {c}")
            self.assertNotIn("Page 1 of 1", c, "Page numbering footer should be filtered out.")

        # Ensure headings like '1. SECURITY DEPOSIT' are attached to clause text
        has_deposit_clause = any("SECURITY DEPOSIT" in c for c in clauses)
        self.assertTrue(has_deposit_clause, "Heading should be preserved and attached to clause.")

    def test_classifier_outputs_required_fields(self):
        """Test that classifier outputs clause_text, predicted_class, and prediction_probability."""
        classifier = AgreementClauseClassifier(model_path=self.models_path)
        results = classifier.classify_pdf(self.test_pdf_path, min_words=6)

        self.assertGreater(len(results), 0, "Classifier should return results for extracted clauses.")

        valid_classes = set(classifier.classes)

        for res in results:
            self.assertIn("clause_text", res)
            self.assertIn("predicted_class", res)
            self.assertIn("prediction_probability", res)

            # Check types and bounds
            self.assertIsInstance(res["clause_text"], str)
            self.assertGreaterEqual(len(res["clause_text"].split()), 6)

            self.assertIn(res["predicted_class"], valid_classes)
            self.assertIsInstance(res["prediction_probability"], float)
            self.assertGreaterEqual(res["prediction_probability"], 0.0)
            self.assertLessEqual(res["prediction_probability"], 1.0)

    def test_convenience_function(self):
        """Test the top-level classify_rental_agreement convenience function."""
        results = classify_rental_agreement(
            pdf_path=self.test_pdf_path,
            model_path=self.models_path,
            min_words=6,
        )
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
