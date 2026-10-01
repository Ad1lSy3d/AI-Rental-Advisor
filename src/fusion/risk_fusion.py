"""
Risk Fusion module for the AI Rental Home Advisor.

Combines:
1. Visual defect risk from YOLO defect detection (Rv)
2. Agreement risk from classified lease clauses (Rl)
3. Regulatory statutory risk from hybrid RAG retrieval (Rr)

Formulation:
    R_total = wv * Rv + wl * Rl + wr * Rr
    Property Health Score = 100 * (1 - R_total)

Prototype Weights:
    wv = 0.4 (Visual risk)
    wl = 0.3 (Agreement risk)
    wr = 0.3 (Regulatory risk)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Prototype Constants and Risk Mappings
# ---------------------------------------------------------------------------

DEFAULT_WEIGHT_VISUAL: float = 0.4
DEFAULT_WEIGHT_AGREEMENT: float = 0.3
DEFAULT_WEIGHT_REGULATORY: float = 0.3

# Visual defect severity mapping
DEFECT_SEVERITY_MAP: Dict[str, float] = {
    "crack": 0.7,
    "mold": 0.8,
    "pest": 0.6,
}

# Agreement clause class base risk mapping
CLAUSE_RISK_MAP: Dict[str, float] = {
    "Security Deposit": 0.5,
    "Maintenance": 0.4,
    "Repairs": 0.6,
    "Penalties": 0.7,
    "Termination": 0.7,
    "Tenant Responsibilities": 0.4,
    "Landlord Responsibilities": 0.3,
}


def normalize_defect_name(name: str) -> Optional[str]:
    """
    Normalize defect string to standard canonical classes: 'crack', 'mold', 'pest'.
    """
    clean = name.strip().lower()
    if "crack" in clean:
        return "crack"
    if "mold" in clean or "mould" in clean:
        return "mold"
    if "pest" in clean:
        return "pest"
    return clean if clean in DEFECT_SEVERITY_MAP else None


def normalize_clause_class(class_name: str) -> Optional[str]:
    """
    Match clause class name case-insensitively to standard categories.
    """
    clean = class_name.strip().lower()
    for standard_class in CLAUSE_RISK_MAP:
        if clean == standard_class.lower():
            return standard_class
    return None


# ---------------------------------------------------------------------------
# Core Result Data Structure
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """
    Structured outcome of the Multimodal Risk Fusion assessment.
    """
    visual_risk: float
    agreement_risk: float
    regulatory_risk: float
    total_risk: float
    property_health_score: float
    detected_defects: List[Any] = field(default_factory=list)
    classified_clauses: List[Dict[str, Any]] = field(default_factory=list)
    supporting_regulatory_evidence: List[Dict[str, Any]] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "wv": DEFAULT_WEIGHT_VISUAL,
        "wl": DEFAULT_WEIGHT_AGREEMENT,
        "wr": DEFAULT_WEIGHT_REGULATORY,
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a standard dictionary representation."""
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style key access (e.g. result['property_health_score'])."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in FusionResult.")


# ---------------------------------------------------------------------------
# Component Risk Calculation Functions
# ---------------------------------------------------------------------------

def calculate_visual_risk(
    defects: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
) -> float:
    """
    Calculate Visual Risk (Rv) from detected YOLO defects.

    Severity mapping:
    - crack -> 0.7
    - mold -> 0.8
    - pest -> 0.6
    - No detected defect -> 0.0
    - If multiple defects exist, use the highest severity.

    Args:
        defects: List of defect names or defect dictionaries.

    Returns:
        Visual risk score (float in [0.0, 1.0]).
    """
    if not defects:
        return 0.0

    severities: List[float] = []
    for d in defects:
        if isinstance(d, dict):
            defect_name = d.get("class") or d.get("name") or d.get("label") or ""
        elif isinstance(d, str):
            defect_name = d
        else:
            defect_name = str(d)

        canonical = normalize_defect_name(defect_name)
        if canonical and canonical in DEFECT_SEVERITY_MAP:
            severities.append(DEFECT_SEVERITY_MAP[canonical])

    if not severities:
        return 0.0

    return round(float(max(severities)), 4)


def calculate_agreement_risk(
    classified_clauses: Optional[Sequence[Dict[str, Any]]] = None,
    aggregation: str = "mean",
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Agreement Risk (Rl) from classified rental agreement clauses.

    Base risk mapping:
    - Security Deposit -> 0.5
    - Maintenance -> 0.4
    - Repairs -> 0.6
    - Penalties -> 0.7
    - Termination -> 0.7
    - Tenant Responsibilities -> 0.4
    - Landlord Responsibilities -> 0.3

    Each clause risk is weighted by the classifier confidence:
        clause_risk = base_risk * confidence

    Args:
        classified_clauses: List of clause dicts with predicted class and confidence.
        aggregation: Aggregation method across clauses ('mean', 'max', or 'weighted_mean').

    Returns:
        Tuple of (agreement_risk, enriched_classified_clauses).
    """
    if not classified_clauses:
        return 0.0, []

    enriched: List[Dict[str, Any]] = []
    for clause in classified_clauses:
        raw_class = str(
            clause.get("predicted_class")
            or clause.get("class")
            or clause.get("label")
            or ""
        )
        conf = float(
            clause.get("prediction_probability")
            if "prediction_probability" in clause
            else clause.get("confidence", clause.get("score", 1.0))
        )
        conf = max(0.0, min(1.0, conf))

        canonical_class = normalize_clause_class(raw_class) or raw_class
        base_risk = CLAUSE_RISK_MAP.get(canonical_class, 0.5)
        clause_risk = round(base_risk * conf, 4)

        enriched_item = dict(clause)
        enriched_item["base_risk"] = base_risk
        enriched_item["confidence"] = conf
        enriched_item["clause_risk"] = clause_risk
        enriched.append(enriched_item)

    if not enriched:
        return 0.0, []

    if aggregation == "max":
        aggregated_risk = max(c["clause_risk"] for c in enriched)
    elif aggregation == "weighted_mean":
        total_conf = sum(c["confidence"] for c in enriched)
        if total_conf > 0:
            aggregated_risk = sum(c["base_risk"] * c["confidence"] for c in enriched) / total_conf
        else:
            aggregated_risk = sum(c["base_risk"] for c in enriched) / len(enriched)
    else:  # default 'mean'
        aggregated_risk = sum(c["clause_risk"] for c in enriched) / len(enriched)

    clamped_risk = max(0.0, min(1.0, float(aggregated_risk)))
    return round(clamped_risk, 4), enriched


def calculate_regulatory_risk(
    supporting_regulatory_evidence: Optional[Sequence[Dict[str, Any]]] = None,
) -> float:
    """
    Calculate Regulatory Risk (Rr) from hybrid RAG retrieval scores.

    For the first prototype:
        regulatory_risk = highest relevant retrieval score

    Args:
        supporting_regulatory_evidence: List of retrieved chunks with 'retrieval_score'.

    Returns:
        Regulatory risk score (float in [0.0, 1.0]).
    """
    if not supporting_regulatory_evidence:
        return 0.0

    scores: List[float] = []
    for item in supporting_regulatory_evidence:
        score = item.get("retrieval_score", item.get("score", 0.0))
        try:
            scores.append(float(score))
        except (ValueError, TypeError):
            continue

    if not scores:
        return 0.0

    max_score = max(scores)
    clamped_score = max(0.0, min(1.0, max_score))
    return round(clamped_score, 4)


# ---------------------------------------------------------------------------
# Late Fusion Pipeline
# ---------------------------------------------------------------------------

def fuse_risks(
    detected_defects: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
    classified_clauses: Optional[Sequence[Dict[str, Any]]] = None,
    supporting_regulatory_evidence: Optional[Sequence[Dict[str, Any]]] = None,
    wv: float = DEFAULT_WEIGHT_VISUAL,
    wl: float = DEFAULT_WEIGHT_AGREEMENT,
    wr: float = DEFAULT_WEIGHT_REGULATORY,
    clause_aggregation: str = "mean",
) -> FusionResult:
    """
    Perform multimodal risk late fusion.

    Computes:
        R_total = wv * Rv + wl * Rl + wr * Rr
        Property Health Score = 100 * (1 - R_total)

    Args:
        detected_defects: Detected YOLO defect names or dicts.
        classified_clauses: Classified agreement clauses with labels and confidences.
        supporting_regulatory_evidence: Top-k retrieved regulatory chunks.
        wv: Prototype weight for visual risk (default 0.4).
        wl: Prototype weight for agreement risk (default 0.3).
        wr: Prototype weight for regulatory risk (default 0.3).
        clause_aggregation: Method to aggregate multiple clauses ('mean', 'max', 'weighted_mean').

    Returns:
        FusionResult containing visual_risk, agreement_risk, regulatory_risk,
        total_risk, property_health_score, detected_defects, classified_clauses,
        and supporting_regulatory_evidence.
    """
    # 1. Component Risks
    rv = calculate_visual_risk(detected_defects)
    rl, enriched_clauses = calculate_agreement_risk(classified_clauses, aggregation=clause_aggregation)
    rr = calculate_regulatory_risk(supporting_regulatory_evidence)

    # 2. Linear Late Fusion
    r_total_raw = (wv * rv) + (wl * rl) + (wr * rr)
    total_risk = round(max(0.0, min(1.0, float(r_total_raw))), 4)

    # 3. Property Health Score: 100 * (1 - R_total)
    health_score_raw = 100.0 * (1.0 - total_risk)
    property_health_score = round(max(0.0, min(100.0, float(health_score_raw))), 2)

    defects_list = list(detected_defects) if detected_defects is not None else []
    evidence_list = list(supporting_regulatory_evidence) if supporting_regulatory_evidence is not None else []

    return FusionResult(
        visual_risk=rv,
        agreement_risk=rl,
        regulatory_risk=rr,
        total_risk=total_risk,
        property_health_score=property_health_score,
        detected_defects=defects_list,
        classified_clauses=enriched_clauses,
        supporting_regulatory_evidence=evidence_list,
        weights={"wv": wv, "wl": wl, "wr": wr},
    )


class RiskFusionAssessor:
    """
    High-level orchestrator for multimodal property risk assessment.
    Can fuse pre-computed outputs or connect with agreement NLP and regulatory RAG.
    """

    def __init__(
        self,
        wv: float = DEFAULT_WEIGHT_VISUAL,
        wl: float = DEFAULT_WEIGHT_AGREEMENT,
        wr: float = DEFAULT_WEIGHT_REGULATORY,
    ):
        self.wv = wv
        self.wl = wl
        self.wr = wr

    def assess(
        self,
        detected_defects: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        classified_clauses: Optional[Sequence[Dict[str, Any]]] = None,
        supporting_regulatory_evidence: Optional[Sequence[Dict[str, Any]]] = None,
        clause_aggregation: str = "mean",
    ) -> FusionResult:
        """
        Assess property risk given component outputs.
        """
        return fuse_risks(
            detected_defects=detected_defects,
            classified_clauses=classified_clauses,
            supporting_regulatory_evidence=supporting_regulatory_evidence,
            wv=self.wv,
            wl=self.wl,
            wr=self.wr,
            clause_aggregation=clause_aggregation,
        )

    def assess_clause_with_rag(
        self,
        clause_text: str,
        predicted_class: str,
        confidence: float,
        detected_defects: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        top_k_rag: int = 3,
        retriever: Optional[Any] = None,
    ) -> FusionResult:
        """
        End-to-end convenience method: retrieves supporting regulatory evidence for
        a classified clause, then fuses visual, agreement, and regulatory risks.
        """
        from src.rag import get_regulatory_retriever

        rag = retriever or get_regulatory_retriever()
        evidence = rag.retrieve(query=clause_text, top_k=top_k_rag)

        clause_item = [{
            "clause_text": clause_text,
            "predicted_class": predicted_class,
            "prediction_probability": confidence,
        }]

        return self.assess(
            detected_defects=detected_defects,
            classified_clauses=clause_item,
            supporting_regulatory_evidence=evidence,
        )
