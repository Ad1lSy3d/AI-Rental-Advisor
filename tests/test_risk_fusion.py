"""
Unit tests for the Multimodal Risk Fusion Layer.
"""

from pathlib import Path
import sys
import unittest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.risk_fusion import (
    FusionResult,
    RiskFusionAssessor,
    calculate_agreement_risk,
    calculate_regulatory_risk,
    calculate_visual_risk,
    fuse_risks,
)


class TestRiskFusion(unittest.TestCase):
    """Test suite for multimodal risk fusion."""

    def test_no_detected_defects(self):
        """
        Verify visual risk behavior when no defects are detected:
        - Empty list -> Rv = 0.0
        - None -> Rv = 0.0
        - Unknown defect -> Rv = 0.0
        - If all inputs are zero, total risk is 0.0 and health score is 100.0.
        """
        self.assertEqual(calculate_visual_risk([]), 0.0)
        self.assertEqual(calculate_visual_risk(None), 0.0)
        self.assertEqual(calculate_visual_risk(["clean_wall", "intact_paint"]), 0.0)

        # Fusion with no defects and no clauses/evidence
        result = fuse_risks(detected_defects=[])
        self.assertEqual(result.visual_risk, 0.0)
        self.assertEqual(result.agreement_risk, 0.0)
        self.assertEqual(result.regulatory_risk, 0.0)
        self.assertEqual(result.total_risk, 0.0)
        self.assertEqual(result.property_health_score, 100.0)
        self.assertEqual(result.detected_defects, [])

    def test_high_visual_risk(self):
        """
        Verify visual risk calculation when severe defects exist:
        - crack -> 0.7
        - mold -> 0.8
        - pest -> 0.6
        - multiple defects -> use highest severity
        """
        # Single defect checks
        self.assertEqual(calculate_visual_risk(["crack"]), 0.7)
        self.assertEqual(calculate_visual_risk(["pest"]), 0.6)
        self.assertEqual(calculate_visual_risk(["mold"]), 0.8)

        # Dictionary format checks
        dict_defects = [{"name": "mold", "confidence": 0.92}]
        self.assertEqual(calculate_visual_risk(dict_defects), 0.8)

        # Multiple defects: highest severity should win (mold = 0.8 > crack = 0.7 > pest = 0.6)
        multi_defects = ["crack", "mold", "pest"]
        self.assertEqual(calculate_visual_risk(multi_defects), 0.8)

        # Fusion test with high visual risk only
        result = fuse_risks(detected_defects=["mold"])
        self.assertEqual(result.visual_risk, 0.8)
        # Total risk = 0.4 * 0.8 + 0.3 * 0 + 0.3 * 0 = 0.32
        self.assertEqual(result.total_risk, 0.32)
        # Health score = 100 * (1 - 0.32) = 68.0
        self.assertEqual(result.property_health_score, 68.0)

    def test_high_agreement_risk(self):
        """
        Verify agreement risk calculation:
        - Penalties -> 0.7 base risk
        - Termination -> 0.7 base risk
        - Repairs -> 0.6 base risk
        - Security Deposit -> 0.5 base risk
        - Maintenance -> 0.4 base risk
        - Tenant Responsibilities -> 0.4 base risk
        - Landlord Responsibilities -> 0.3 base risk
        - Weighted by classifier confidence
        """
        # 1. High risk with full confidence (1.0)
        clauses_full_conf = [{
            "clause_text": "Tenant shall pay hefty penalty for late payment.",
            "predicted_class": "Penalties",
            "prediction_probability": 1.0,
        }]
        risk_full, enriched_full = calculate_agreement_risk(clauses_full_conf)
        self.assertEqual(risk_full, 0.7)
        self.assertEqual(enriched_full[0]["clause_risk"], 0.7)

        # 2. High risk with partial confidence (0.80) -> 0.7 * 0.8 = 0.56
        clauses_partial = [{
            "clause_text": "Immediate termination without refund upon breach.",
            "predicted_class": "Termination",
            "prediction_probability": 0.8,
        }]
        risk_partial, enriched_partial = calculate_agreement_risk(clauses_partial)
        self.assertEqual(risk_partial, 0.56)
        self.assertEqual(enriched_partial[0]["clause_risk"], 0.56)

        # 3. Fusion test with high agreement risk only
        result = fuse_risks(classified_clauses=clauses_full_conf)
        self.assertEqual(result.agreement_risk, 0.7)
        # Total risk = 0.4 * 0 + 0.3 * 0.7 + 0.3 * 0 = 0.21
        self.assertEqual(result.total_risk, 0.21)
        # Health score = 100 * (1 - 0.21) = 79.0
        self.assertEqual(result.property_health_score, 79.0)

    def test_complete_end_to_end_fusion(self):
        """
        Verify complete end-to-end multimodal risk fusion:
        R_total = 0.4*Rv + 0.3*Rl + 0.3*Rr
        Property Health Score = 100 * (1 - R_total)
        Check all 8 required return fields.
        """
        detected_defects = ["crack"]  # Rv = 0.7
        classified_clauses = [{
            "clause_text": "Tenant must pay Rs 500 per day delayed rent penalty.",
            "predicted_class": "Penalties",
            "prediction_probability": 0.90,  # Rl = 0.7 * 0.90 = 0.63
        }]
        supporting_evidence = [
            {
                "text": "Section 74: Compensation for breach of contract where penalty stipulated.",
                "source": "07_indian_contract_act_1872_penalties_liquidated_damages.txt",
                "retrieval_score": 0.75,
            },
            {
                "text": "Section 73: Compensation for loss or damage caused by breach of contract.",
                "source": "07_indian_contract_act_1872_penalties_liquidated_damages.txt",
                "retrieval_score": 0.62,
            },
        ]  # Rr = max(0.75, 0.62) = 0.75

        result = fuse_risks(
            detected_defects=detected_defects,
            classified_clauses=classified_clauses,
            supporting_regulatory_evidence=supporting_evidence,
            wv=0.4,
            wl=0.3,
            wr=0.3,
        )

        # Verification of component risks
        self.assertEqual(result.visual_risk, 0.7)
        self.assertEqual(result.agreement_risk, 0.63)
        self.assertEqual(result.regulatory_risk, 0.75)

        # Expected total risk: 0.4*0.7 + 0.3*0.63 + 0.3*0.75 = 0.28 + 0.189 + 0.225 = 0.694
        self.assertEqual(result.total_risk, 0.694)

        # Expected Property Health Score: 100 * (1 - 0.694) = 30.60
        self.assertEqual(result.property_health_score, 30.60)

        # Verify all 8 required output fields exist and match
        self.assertIsInstance(result.visual_risk, float)
        self.assertIsInstance(result.agreement_risk, float)
        self.assertIsInstance(result.regulatory_risk, float)
        self.assertIsInstance(result.total_risk, float)
        self.assertIsInstance(result.property_health_score, float)
        self.assertEqual(result.detected_defects, ["crack"])
        self.assertEqual(len(result.classified_clauses), 1)
        self.assertEqual(result.classified_clauses[0]["clause_risk"], 0.63)
        self.assertEqual(len(result.supporting_regulatory_evidence), 2)

        # Verify dictionary representation and key indexing
        res_dict = result.to_dict()
        for key in [
            "visual_risk",
            "agreement_risk",
            "regulatory_risk",
            "total_risk",
            "property_health_score",
            "detected_defects",
            "classified_clauses",
            "supporting_regulatory_evidence",
        ]:
            self.assertIn(key, res_dict)
            self.assertEqual(result[key], res_dict[key])

    def test_assessor_with_live_rag_retrieval(self):
        """
        Verify end-to-end integration using RiskFusionAssessor with live RegulatoryRetriever.
        """
        assessor = RiskFusionAssessor()
        clause_text = "The tenant shall pay a non-refundable security deposit of twelve months rent."

        result = assessor.assess_clause_with_rag(
            clause_text=clause_text,
            predicted_class="Security Deposit",
            confidence=0.95,
            detected_defects=["crack"],
            top_k_rag=2,
        )

        self.assertIsInstance(result, FusionResult)
        self.assertEqual(result.visual_risk, 0.7)
        # Security Deposit base risk = 0.5 * 0.95 = 0.475
        self.assertEqual(result.agreement_risk, 0.475)
        # RAG should have retrieved relevant evidence from Model Tenancy Act / Rent Control Act
        self.assertGreater(result.regulatory_risk, 0.0)
        self.assertGreater(len(result.supporting_regulatory_evidence), 0)
        self.assertGreater(result.property_health_score, 0.0)
        self.assertLessEqual(result.property_health_score, 100.0)


if __name__ == "__main__":
    unittest.main()
