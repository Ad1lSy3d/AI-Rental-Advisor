"""
AI Rental Home Advisor - FastAPI Application Server

Exposes multimodal assessment endpoint combining:
- YOLO11 Computer Vision (defect detection)
- Agreement NLP (clause extraction & TF-IDF/Logistic Regression classification)
- Statutory RAG (BM25 + SBERT/FAISS hybrid retrieval)
- Multimodal Risk Fusion (Property Health Score)
"""

from __future__ import annotations

import base64
from pathlib import Path
import random
import tempfile
import re
from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.cv import get_yolo_detector
from src.fusion import CLAUSE_RISK_MAP, fuse_risks
from src.nlp import AgreementClauseClassifier
from src.rag import get_regulatory_retriever

class ChatRequest(BaseModel):
    query: str


# Initialize FastAPI application
app = FastAPI(
    title="AI Rental Home Advisor API",
    description="Multimodal preliminary assessment of rental properties using CV, NLP, and RAG.",
    version="1.0.0",
)

# Configure CORS for local development (Vite frontend on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _format_act_title(source_filename: str) -> str:
    """Format statutory filename into human-readable Act title."""
    clean = Path(source_filename).stem
    # Remove prefix numbers like '01_', '02_'
    parts = clean.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    name = " ".join(parts).title()
    name = name.replace("Mta", "MTA").replace("Nbc", "NBC").replace("Maharera", "MahaRERA")
    return name


def _generate_recommendations(
    defects: List[Dict[str, Any]],
    clauses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate targeted, actionable tenant recommendations based on findings."""
    recs: List[Dict[str, Any]] = []

    # Visual defect recommendations
    defect_classes = {d.get("class", "").lower() for d in defects}
    if "mold" in defect_classes:
        recs.append({
            "id": "rec-mold",
            "icon": "water_drop",
            "title": "Check affected wall and ceiling for moisture before signing.",
            "detail": "Request a handheld moisture meter reading or photographic proof of seal remediation from the landlord under NBC 2016 habitability standards.",
        })
    if "crack" in defect_classes:
        recs.append({
            "id": "rec-crack",
            "icon": "handyman",
            "title": "Document plaster & surface settling cracks on handover sheet.",
            "detail": "Annotate all pre-existing wall settling cracks to prevent unauthorized security deposit deductions upon tenancy completion.",
        })
    if "pest" in defect_classes:
        recs.append({
            "id": "rec-pest",
            "icon": "pest_control",
            "title": "Request documented pest remediation certificate.",
            "detail": "Ensure warranty against pest infestation is confirmed in writing by the landlord prior to taking physical possession.",
        })

    # Agreement clause recommendations
    clause_classes = {c.get("predicted_class", "") for c in clauses}
    if "Penalties" in clause_classes:
        recs.append({
            "id": "rec-penalties",
            "icon": "history_edu",
            "title": "Review penalty clause under Section 74 Indian Contract Act.",
            "detail": "Statutory rules require stipulated penalties to represent genuine pre-estimated damage rather than punitive unilateral forfeiture.",
        })
    if "Termination" in clause_classes:
        recs.append({
            "id": "rec-termination",
            "icon": "history_edu",
            "title": "Verify reciprocal lease termination notice periods.",
            "detail": "Request amendment ensuring at least 30 days mutual written notice rather than immediate unilateral forfeiture of tenancy.",
        })
    if "Security Deposit" in clause_classes:
        recs.append({
            "id": "rec-deposit",
            "icon": "account_balance",
            "title": "Confirm security deposit terms under Model Tenancy Act.",
            "detail": "Section 10 caps residential deposits at maximum two months rent and prohibits deductions for ordinary wear and tear.",
        })
    if "Maintenance" in clause_classes or "Repairs" in clause_classes:
        recs.append({
            "id": "rec-repairs",
            "icon": "handyman",
            "title": "Confirm division of repair liabilities with landlord.",
            "detail": "Clarify whether minor repairs cover pre-existing wear or solely tenant fault, as outlined under Schedule II of Model Tenancy Act.",
        })

    if not clauses:
        recs.append({
            "id": "rec-agreement-optional",
            "icon": "description",
            "title": "Upload agreement for complete lease clause audit.",
            "detail": "Preliminary visual analysis complete. Provide your draft agreement to audit penalties, deposit caps, and termination rights.",
        })

    # Fallback generic recommendation if list is short
    if len(recs) < 2:
        recs.append({
            "id": "rec-general",
            "icon": "verified_user",
            "title": "Review statutory habitability provisions before execution.",
            "detail": "Ensure tenancy agreement conforms with Maharashtra Rent Control Act Section 14 landlord repair covenants.",
        })

    return recs[:4]


def _generate_landlord_questions(
    defects: List[Dict[str, Any]],
    clauses: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Generate dynamic, finding-grounded questions for the tenant to pose to the landlord."""
    questions: List[Dict[str, str]] = []
    defect_classes = {d.get("class", "").lower() for d in defects}

    if "mold" in defect_classes:
        questions.append({
            "category": "Visual Finding · Moisture & Mold",
            "question": "When was this mold and moisture seepage last inspected or treated, and can you provide written documentation of the dampness source repair?",
            "basis": "Grounding: YOLO detected mold / NBC 2016 Part 3 habitability & dampness standards.",
        })
    if "crack" in defect_classes:
        questions.append({
            "category": "Visual Finding · Structural & Settling Cracks",
            "question": "Has this wall crack been inspected by a certified civil engineer, and will this pre-existing condition be formally listed in our move-in handover inventory?",
            "basis": "Grounding: YOLO detected wall crack / MRC Act 1999 s. 14 duty of tenantable repair.",
        })
    if "pest" in defect_classes:
        questions.append({
            "category": "Visual Finding · Pest Infestation",
            "question": "When was pest control treatment last executed for the premises, and does the landlord bear the cost for ongoing extermination?",
            "basis": "Grounding: YOLO detected pest defect / Municipal sanitation regulations.",
        })

    clause_classes = {c.get("predicted_class", "") for c in clauses}
    if "Security Deposit" in clause_classes:
        questions.append({
            "category": "Agreement Audit · Security Deposit",
            "question": "Will the draft security deposit terms be capped at two months' rent as mandated by Section 10 of the Model Tenancy Act?",
            "basis": "Grounding: NLP classified Security Deposit clause / Model Tenancy Act 2021 s. 10(1)(a).",
        })
    if "Penalties" in clause_classes:
        questions.append({
            "category": "Agreement Audit · Penalty Provisions",
            "question": "Can the liquidated damages clause be revised to ensure penalties reflect reasonable pre-estimated loss under Section 74 of the Indian Contract Act?",
            "basis": "Grounding: NLP classified Penalties clause / Indian Contract Act 1872 s. 74.",
        })
    if "Termination" in clause_classes:
        questions.append({
            "category": "Agreement Audit · Termination Notice",
            "question": "Can we confirm a reciprocal written notice period of at least 30 days prior to lease termination for both parties?",
            "basis": "Grounding: NLP classified Termination clause / Transfer of Property Act 1882 s. 106.",
        })
    if "Maintenance" in clause_classes or "Repairs" in clause_classes:
        questions.append({
            "category": "Agreement Audit · Maintenance Liabilities",
            "question": "Who is responsible for pre-existing structural wear and major plumbing/electrical defects under Schedule II of the Model Tenancy Act?",
            "basis": "Grounding: NLP classified Maintenance/Repairs clause / MTA Schedule II division of duties.",
        })

    if not questions:
        questions.append({
            "category": "Tenancy Governance",
            "question": "Will the leave and licence agreement be formally executed in writing and registered under Section 55 of the Maharashtra Rent Control Act, 1999?",
            "basis": "Grounding: MRC Act 1999 Section 55 mandatory registration.",
        })
        questions.append({
            "category": "Move-In Protocol",
            "question": "Can we conduct a joint photographic condition inventory prior to possession and attach it as an addendum to the agreement?",
            "basis": "Grounding: Best tenancy practice under Model Tenancy Act 2021.",
        })

    return questions


@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "app": "AI Rental Home Advisor API"}


@app.post("/api/assess")
async def assess_property(
    images: List[UploadFile] = File(...),
    agreement_pdf: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Multimodal property assessment endpoint.

    Accepts:
    - images: Multiple property photos (JPG/PNG)
    - agreement_pdf: Optional rental agreement PDF document

    Executes pipeline:
    1. YOLO11 defect vision on property images
    2. (Optional) PyMuPDF extraction + TF-IDF/Logistic Regression clause classification
    3. Hybrid BM25 + SBERT/FAISS regulatory retrieval
    4. Multimodal Late Risk Fusion
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one property image must be uploaded.")

    # -------------------------------------------------------------------------
    # 1. Computer Vision Defect Detection (YOLO11)
    # -------------------------------------------------------------------------
    detector = get_yolo_detector()
    all_defects: List[Dict[str, Any]] = []

    for img_idx, img_file in enumerate(images):
        img_bytes = await img_file.read()
        if not img_bytes:
            continue

        detected = detector.detect(
            image_input=img_bytes,
            source_name=img_file.filename or f"image_{img_idx+1}.jpg",
            conf_threshold=0.25,
        )

        # Generate lightweight base64 thumbnail preview for frontend display
        b64_img = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        for d in detected:
            d["image_url"] = b64_img

        all_defects.extend(detected)

    # -------------------------------------------------------------------------
    # 2. Rental Agreement NLP (Extraction & Classification) - OPTIONAL
    # -------------------------------------------------------------------------
    formatted_clauses: List[Dict[str, Any]] = []
    has_agreement = agreement_pdf is not None and bool(agreement_pdf.filename)

    if has_agreement:
        pdf_bytes = await agreement_pdf.read()
        if pdf_bytes:
            # Write temporarily to disk for PyMuPDF extraction
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                tmp_pdf.write(pdf_bytes)
                tmp_pdf_path = Path(tmp_pdf.name)

            try:
                classifier = AgreementClauseClassifier()
                raw_clauses = classifier.classify_pdf(tmp_pdf_path, min_words=6)
            finally:
                if tmp_pdf_path.exists():
                    tmp_pdf_path.unlink()

            # Format clauses with risk badge
            for i, c in enumerate(raw_clauses):
                pred_class = c.get("predicted_class", "Tenant Responsibilities")
                conf = float(c.get("prediction_probability", 0.8))
                base_risk = CLAUSE_RISK_MAP.get(pred_class, 0.4)
                c_risk = round(base_risk * conf, 3)

                if c_risk >= 0.6:
                    badge = "High"
                elif c_risk >= 0.4:
                    badge = "Moderate"
                else:
                    badge = "Low"

                formatted_clauses.append({
                    "id": f"clause-{i+1}",
                    "predicted_class": pred_class,
                    "section": f"Clause {i+1}",
                    "clause_text": c.get("clause_text", ""),
                    "summary": c.get("clause_text", "")[:120] + ("..." if len(c.get("clause_text", "")) > 120 else ""),
                    "prediction_probability": conf,
                    "clause_risk": c_risk,
                    "risk_badge": badge,
                })

    # -------------------------------------------------------------------------
    # 3. Regulatory Statutory RAG Retrieval
    # -------------------------------------------------------------------------
    retriever = get_regulatory_retriever()
    retrieved_evidence_map: Dict[str, Dict[str, Any]] = {}

    if formatted_clauses:
        # Query RAG for the top clauses (prioritizing high-risk categories)
        sorted_clauses = sorted(formatted_clauses, key=lambda x: x["clause_risk"], reverse=True)
        query_clauses = sorted_clauses[:4]

        for c in query_clauses:
            query_text = c["clause_text"]
            matches = retriever.retrieve(query=query_text, top_k=2)
            for m in matches:
                text_key = m["text"][:100]
                if text_key not in retrieved_evidence_map:
                    retrieved_evidence_map[text_key] = m
    else:
        # When no agreement PDF is provided, query RAG based on defect/habitability standards
        defect_classes = {d.get("class", "").lower() for d in all_defects}
        queries: List[str] = []
        if "mold" in defect_classes:
            queries.append("dampness moisture mould landlord habitability repairs")
        if "crack" in defect_classes:
            queries.append("structural safety wall cracks building maintenance repair")
        if "pest" in defect_classes:
            queries.append("sanitation pest control health safety habitability")

        if not queries:
            queries.append("residential habitability building safety maintenance tenant rights")

        for q in queries:
            matches = retriever.retrieve(query=q, top_k=2)
            for m in matches:
                text_key = m["text"][:100]
                if text_key not in retrieved_evidence_map:
                    retrieved_evidence_map[text_key] = m

    all_evidence = list(retrieved_evidence_map.values())
    if not all_evidence:
        # Fallback query if no specific clause matched
        all_evidence = retriever.retrieve(query="security deposit maintenance repair penalty", top_k=2)

    formatted_evidence: List[Dict[str, Any]] = []
    for i, ev in enumerate(all_evidence[:3]):
        source_name = ev.get("source", "Statutory Provisions")
        act_title = _format_act_title(source_name)
        quote_text = ev.get("text", "").strip()

        # Clean up quote for presentation
        if len(quote_text) > 280:
            quote_display = quote_text[:280] + "..."
        else:
            quote_display = quote_text

        formatted_evidence.append({
            "id": f"reg-{i+1}",
            "act": act_title,
            "section": f"Relevance Score: {int(ev.get('retrieval_score', 0.8) * 100)}%",
            "quote": quote_display,
            "jurisdiction": "Maharashtra Residential Tenancy Law",
            "source_file": source_name,
            "retrieval_score": round(float(ev.get("retrieval_score", 0.0)), 4),
        })

    # -------------------------------------------------------------------------
    # 4. Multimodal Late Risk Fusion
    # -------------------------------------------------------------------------
    fusion_result = fuse_risks(
        detected_defects=[d["class"] for d in all_defects],
        classified_clauses=formatted_clauses,
        supporting_regulatory_evidence=formatted_evidence,
    )

    # -------------------------------------------------------------------------
    # 5. Recommendations & Combined Payload
    # -------------------------------------------------------------------------
    recommendations = _generate_recommendations(all_defects, formatted_clauses)

    # Calculate overall confidence
    confidences: List[float] = [c["prediction_probability"] for c in formatted_clauses]
    if all_defects:
        confidences.extend([d["confidence"] for d in all_defects])
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.90

    audit_number = random.randint(1000, 9999)

    return {
        "property_health_score": fusion_result.property_health_score,
        "total_risk": fusion_result.total_risk,
        "visual_risk": fusion_result.visual_risk,
        "agreement_risk": fusion_result.agreement_risk,
        "regulatory_risk": fusion_result.regulatory_risk,
        "detected_defects": all_defects,
        "classified_clauses": formatted_clauses,
        "regulatory_evidence": formatted_evidence,
        "supporting_regulatory_evidence": formatted_evidence,
        "recommendations": recommendations,
        "landlord_questions": _generate_landlord_questions(all_defects, formatted_clauses),
        "audit_id": f"RA-{audit_number}",
        "confidence_overall": round(avg_confidence, 2),
    }


@app.post("/api/chat")
async def chat_regulatory_assistant(request: ChatRequest) -> Dict[str, Any]:
    """
    RAG Chatbot endpoint: Retrieves authoritative regulatory evidence from the
    ingested statutory corpus (BM25 + SBERT + FAISS) and returns a grounded answer.
    """
    query_text = (request.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    retriever = get_regulatory_retriever()
    matches = retriever.retrieve(query=query_text, top_k=3)

    # Threshold for sufficient regulatory evidence
    if not matches or matches[0].get("retrieval_score", 0.0) < 0.25:
        return {
            "query": query_text,
            "answer": "Insufficient evidence in the available regulatory sources.",
            "evidence": [],
            "grounded": False,
        }

    top_match = matches[0]
    top_source = top_match.get("source", "Statutory Provisions")
    top_act = _format_act_title(top_source)
    top_score = top_match.get("retrieval_score", 0.0)
    top_text = top_match.get("text", "").strip()

    # Clean excerpt
    clean_passage = " ".join(top_text.split()[:80])
    if len(clean_passage) < len(top_text):
        clean_passage += "..."

    answer_text = (
        f"According to {top_act}:\n\n"
        f"\"{clean_passage}\"\n\n"
        f"Statutory provisions under {top_act} directly govern this matter."
    )

    evidence_list: List[Dict[str, Any]] = []
    for m in matches:
        src = m.get("source", "")
        act_name = _format_act_title(src)
        raw_text = m.get("text", "").strip()

        # Extract section label if available
        sec_match = re.search(r"(Section\s+\d+[^:\n]*|PART\s+[A-Z]|CHAPTER\s+[IVXLCDM]+)", raw_text, re.IGNORECASE)
        section_label = sec_match.group(1).strip() if sec_match else "Statutory Provision"

        evidence_list.append({
            "source": src,
            "act": act_name,
            "section": section_label,
            "passage": raw_text,
            "retrieval_score": round(float(m.get("retrieval_score", 0.0)), 4),
            "bm25_score": round(float(m.get("bm25_score", 0.0)), 4),
            "dense_score": round(float(m.get("dense_score", 0.0)), 4),
        })

    return {
        "query": query_text,
        "answer": answer_text,
        "top_act": top_act,
        "evidence": evidence_list,
        "grounded": True,
    }


# -----------------------------------------------------------------------------
# Static Frontend Serving
# -----------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


@app.get("/")
async def serve_index():
    """Serve vanilla frontend index.html."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_file)


@app.get("/style.css")
async def serve_style():
    """Serve vanilla frontend style.css."""
    css_file = FRONTEND_DIR / "style.css"
    if not css_file.exists():
        raise HTTPException(status_code=404, detail="frontend/style.css not found")
    return FileResponse(css_file, media_type="text/css")


@app.get("/app.js")
async def serve_app_js():
    """Serve vanilla frontend app.js."""
    js_file = FRONTEND_DIR / "app.js"
    if not js_file.exists():
        raise HTTPException(status_code=404, detail="frontend/app.js not found")
    return FileResponse(js_file, media_type="application/javascript")


@app.get("/assets/{path:path}")
async def serve_assets(path: str):
    """Serve static assets from frontend/assets/."""
    asset_file = FRONTEND_DIR / "assets" / path
    if asset_file.exists() and asset_file.is_file():
        return FileResponse(asset_file)
    raise HTTPException(status_code=404, detail=f"Asset '{path}' not found")


def main():
    """Run FastAPI development server."""
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()

