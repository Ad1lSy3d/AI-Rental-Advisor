"""
Multimodal Risk Fusion module for the AI Rental Home Advisor.
"""

from src.fusion.risk_fusion import (
    DEFAULT_WEIGHT_AGREEMENT,
    DEFAULT_WEIGHT_REGULATORY,
    DEFAULT_WEIGHT_VISUAL,
    DEFECT_SEVERITY_MAP,
    CLAUSE_RISK_MAP,
    FusionResult,
    RiskFusionAssessor,
    calculate_agreement_risk,
    calculate_regulatory_risk,
    calculate_visual_risk,
    fuse_risks,
)

__all__ = [
    "DEFAULT_WEIGHT_VISUAL",
    "DEFAULT_WEIGHT_AGREEMENT",
    "DEFAULT_WEIGHT_REGULATORY",
    "DEFECT_SEVERITY_MAP",
    "CLAUSE_RISK_MAP",
    "FusionResult",
    "RiskFusionAssessor",
    "calculate_visual_risk",
    "calculate_agreement_risk",
    "calculate_regulatory_risk",
    "fuse_risks",
]
