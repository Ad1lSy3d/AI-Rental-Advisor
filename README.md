# AI Rental Home Advisor

AI Rental Home Advisor is an AI-powered multimodal rental-property assessment system designed to help prospective tenants evaluate a property before signing a lease. By combining computer vision defect detection, natural language processing for rental agreements, and retrieval-augmented generation (RAG) over authoritative housing regulations, the system identifies physical defects, audits lease covenants, grounds findings in statutory provisions, and fuses these signals into an explainable Property Health Score.

---

## Overview

The system processes physical property inspection photos alongside an optional draft rental agreement to produce an integrated preliminary risk evaluation:

```text
Property Images ───────> YOLO11 ───────────────> Defect Findings ───┐
                                                                    │
Rental Agreement ─────> PyMuPDF + TF-IDF ──────> Clause Findings ───┼──> Regulatory RAG ──> Grounded Evidence
(Optional)              + Logistic Regression                       │         │
                                                                    │         │
                                                                    v         v
                                                               Multimodal Risk Fusion
                                                                    │
                                                                    v
                                                           Property Health Score
                                                                    │
                                                                    v
                                                     Recommendations + Landlord Questions
```

The rental agreement upload is **strictly optional**. If a user submits only property images, the system completes the visual inspection, retrieves relevant habitability and building safety regulations, computes risk with zero agreement penalty, and generates targeted inquiries for the landlord.

---

## Key Features

### Property Defect Detection
- **YOLO11 Vision Model**: Evaluates room and surface images to detect visible physical defects.
- **Defect Categories**: Detects wall and plaster cracks, moisture/mold growth, and pest contamination.
- **Real Model Inferences**: Outputs detected defect bounding coordinates, class labels, and model confidence scores with preview thumbnails.

### Rental Agreement Analysis
- **PDF Extraction**: Extracts and segments clause text directly from digital PDF agreements using PyMuPDF.
- **Clause Classification**: Classifies extracted clauses using a trained TF-IDF vectorizer and balanced Logistic Regression classifier.
- **Categorization**: Groups covenants into 7 core lease categories:
  - Security Deposit
  - Maintenance
  - Repairs
  - Penalties
  - Termination
  - Tenant Responsibilities
  - Landlord Responsibilities
- **Optional & Safe Handling**: Safely handles cases where no agreement is provided without runtime errors or fabricated agreement scores.

### Regulatory RAG
Retrieval-Augmented Generation (RAG) is a core structural layer connecting physical findings and lease terms to statutory law, rather than just an ad-hoc chat widget.
- **Hybrid Retrieval**: Integrates lexical BM25 token matching and dense semantic vector search via Sentence-Transformers (`all-MiniLM-L6-v2`) and FAISS inner-product cosine similarity.
- **Authoritative Corpus**: Ingests statutory Acts and codes (including the Model Tenancy Act 2021, Maharashtra Rent Control Act 1999, National Building Code 2016, and Indian Contract Act 1872).
- **Verifiable Evidence**: Every retrieval provides the statutory Act title, relevant section/provision, verbatim quoted passage, and mathematical relevance score.
- **Grounded Ground-Truth Only**: If a query falls outside the indexed legal corpus, the system explicitly returns:
  > *"Insufficient evidence in the available regulatory sources."*

### Risk Fusion
Combines heterogeneous signals using late linear fusion into a unified Property Health Score:
- **Visual Risk ($R_v$)**: Derived from the highest-severity defect detected (mold: 0.8, crack: 0.7, pest: 0.6).
- **Agreement Risk ($R_l$)**: Weighted clause risk scaled by classifier confidence (zero if no agreement is provided).
- **Regulatory Risk ($R_r$)**: Derived from the top hybrid statutory retrieval relevance.
- **Prototype Weights**:
  $$R_{\text{total}} = 0.40 \cdot R_v + 0.30 \cdot R_l + 0.30 \cdot R_r$$
  $$\text{Property Health Score} = \max(0, \min(100, 100 \cdot (1 - R_{\text{total}})))$$
- *Notice*: These weights represent prototype heuristic assumptions for demonstration and do not constitute validated legal or structural engineering formulas.

### AI Rental Chatbot
- Interactive conversational interface backed by `POST /api/chat`.
- Returns answers grounded strictly in retrieved statutory excerpts rather than speculative LLM hallucinations.
- Displays the user query, grounded answer, retrieved source document, specific section/provision, quoted passage, and hybrid relevance score.

### Inspection Summary
- Displays the overall Property Health Score on a 0–100 gauge with condition badges (*Good Condition*, *Moderate Risk*, *High Risk*).
- Provides a sub-risk breakdown across visual, lease, and regulatory components.
- Surfaces actionable tenant recommendations alongside retrieved statutory evidence.

### Landlord Questions
- Dynamically generates targeted questions for the landlord based on the actual defects found (e.g., dampness remediation history, structural crack inspection certificates) and audited lease clauses (e.g., security deposit caps under Model Tenancy Act Section 10).
- Explains the specific finding and statutory basis under each question, with a one-click copy tool.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   Vanilla Web Frontend (HTML5/CSS3/JS)                 │
│  [Dashboard] [Defect Vision] [Agreement Audit] [RAG Chat] [Summary]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP Fetch / FormData
                                    v
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application Server                      │
│            GET /api/health  │  POST /api/assess  │  POST /api/chat     │
└───────┬───────────────────────────┼────────────────────────────┬───────┘
        │                           │                            │
        v                           v                            v
┌──────────────────┐    ┌──────────────────────┐    ┌────────────────────┐
│   YOLO11 Model   │    │ Agreement Classifier │    │  Regulatory RAG    │
│ (PyTorch/Vision) │    │  (PyMuPDF + TF-IDF   │    │  (BM25 + SBERT     │
│                  │    │  + Logistic Regr.)   │    │   + FAISS Index)   │
└───────┬──────────┘    └───────────┬──────────┘    └────────────┬───────┘
        │ Defects                   │ Clauses                    │
        └─────────────────┬─────────┴────────────────────────────┘
                          │ Findings Query
                          v
                ┌──────────────────┐
                │ Statutory Corpus │
                │ (10 Ingested Acts│
                │  & Guidelines)   │
                └─────────┬────────┘
                          │ Retrieved Evidence
                          v
                ┌──────────────────┐
                │   Risk Fusion    │
                │   (Late Linear   │
                │    Heuristic)    │
                └─────────┬────────┘
                          │
                          v
         ┌───────────────────────────────────┐
         │ Property Health Score (0 - 100)   │
         │ Grounded Statutory Evidence       │
         │ Tenant Recommendations            │
         │ Dynamic Landlord Inquiries        │
         └───────────────────────────────────┘
```

---

## Technology Stack

### Backend & Machine Learning
- **Framework**: Python 3.11, FastAPI, Uvicorn, Pydantic
- **Computer Vision**: Ultralytics YOLO11 (`models/yolo11n_rental_home_v2_best.pt`), Pillow
- **Agreement NLP**: PyMuPDF (`fitz`), scikit-learn (`models/agreement_tfidf_logreg.pkl`, TF-IDF Vectorizer + Logistic Regression)
- **Regulatory RAG**: Rank-BM25, Sentence-Transformers (`all-MiniLM-L6-v2`), FAISS (`faiss-cpu`), NumPy

### Frontend
- **Languages**: Semantic HTML5, Vanilla CSS3, Modern JavaScript (ES6+)
- **APIs**: Native `fetch()` API, Browser File & Drag-and-Drop APIs
- **Design System**: Lightweight, responsive custom stylesheet with zero external frontend frameworks.
- **Dependencies**: **Zero Node.js runtime or build dependencies**. The frontend does not use React, Vite, Node.js, npm, package.json, Tailwind CLI, or node_modules. It is served directly by the FastAPI application server.

---

## Project Structure

```text
AI-Rental-Advisor/
├── data/
│   ├── agreement/
│   │   ├── rental_agreement_dataset_v2.csv
│   │   └── sample_rental_agreement.pdf
│   └── regulatory/
│       ├── 01_maharashtra_rent_control_act_1999.txt
│       ├── 02_model_tenancy_act_2021_provisions.txt
│       ├── 03_model_tenancy_act_2021_schedule_ii_maintenance.txt
│       ├── 04_transfer_of_property_act_1882_chapter_v.txt
│       ├── 05_maharashtra_fire_prevention_life_safety_act_2006.txt
│       ├── 06_national_building_code_2016_residential_habitability.txt
│       ├── 07_indian_contract_act_1872_penalties_liquidated_damages.txt
│       ├── 08_maharashtra_municipal_sanitation_building_safety.txt
│       ├── 09_registration_act_1908_maharashtra_tenancy.txt
│       └── 10_maharera_standard_specifications_defect_liability.txt
├── docs/
│   └── reference_screenshots/
│       ├── home_upload.png
│       ├── analyzing.png
│       └── assessment_results.png
├── frontend/
│   ├── assets/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/
│   ├── agreement_tfidf_logreg.pkl
│   └── yolo11n_rental_home_v2_best.pt
├── notebooks/
├── src/
│   ├── cv/
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── risk_fusion.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── extractor.py
│   └── rag/
│       ├── __init__.py
│       ├── ingestion.py
│       └── retriever.py
├── tests/
│   ├── test_agreement_nlp.py
│   ├── test_api.py
│   ├── test_regulatory_rag.py
│   └── test_risk_fusion.py
├── main.py
├── pyproject.toml
└── README.md
```

---

## API Specification

### `GET /api/health`
Returns system status.
- **Response**: `{"status": "ok", "app": "AI Rental Home Advisor API"}`

### `POST /api/assess`
Executes multimodal property evaluation.
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `images` *(required)*: One or more JPG/PNG property image files.
  - `agreement_pdf` *(optional)*: Draft tenancy agreement PDF document.
- **Response Fields**:
  - `property_health_score`: Overall composite score (0–100).
  - `total_risk`, `visual_risk`, `agreement_risk`, `regulatory_risk`: Numerical risk values (0.0–1.0).
  - `detected_defects`: List of detected defects with bounding boxes, classes, and confidence scores.
  - `classified_clauses`: Extracted clauses, predicted categories, risk badges, and probabilities.
  - `regulatory_evidence`: Grounded statutory provisions retrieved for the property.
  - `recommendations`: Actionable tenant advisories based on findings.
  - `landlord_questions`: Dynamically generated inquiries based on identified defects and clauses.
  - `audit_id`: Audit tracking identifier.

### `POST /api/chat`
Queries the regulatory knowledge base via hybrid BM25 + SBERT RAG.
- **Content-Type**: `application/json`
- **Request Body**: `{"query": "Is a 3-month security deposit allowed?"}`
- **Response Fields**:
  - `query`: The submitted query.
  - `answer`: Grounded statutory answer quoting retrieved legislation.
  - `top_act`: Primary governing Act title.
  - `evidence`: Ranked list of matching provisions containing `source`, `act`, `section`, `passage`, `retrieval_score`, `bm25_score`, and `dense_score`.
  - `grounded`: Boolean flag indicating whether sufficient statutory evidence was found.

---

## Installation

### Prerequisites
- Python 3.11
- `uv` (recommended) or standard `pip`

### Setup
```bash
# Clone repository
git clone <repository-url>
cd AI-Rental-Advisor

# Install dependencies using uv
uv sync
```

*Note: No `npm install` or Node.js environment is required.*

---

## Running the Application

Start the FastAPI application server:
```bash
uv run python main.py
```

Then open your browser to:
```text
http://localhost:8000/
```

FastAPI automatically serves the lightweight vanilla frontend (`index.html`, `style.css`, `app.js`) directly on the root endpoint.

---

## Testing

The test suite validates the NLP classification pipeline, BM25 + SBERT retrieval, risk fusion formulas, and FastAPI endpoints.

Run the test suite using `unittest`:
```bash
uv run python -m unittest discover -s tests
```

Current test suite contains **18 automated tests** covering:
- Unit tests for agreement clause extraction & classification (`tests/test_agreement_nlp.py`)
- Unit tests for BM25 and hybrid SBERT/FAISS retrieval (`tests/test_regulatory_rag.py`)
- Unit tests for multimodal risk weighting and fusion (`tests/test_risk_fusion.py`)
- End-to-end API integration tests for health check, image-only assessment, multimodal assessment, and RAG chat (`tests/test_api.py`)

---

## Example Workflows

### 1. Image-Only Preliminary Assessment
1. Upload photos of room walls, ceilings, and flooring.
2. Skip the rental agreement upload.
3. System runs YOLO11 defect detection on the uploaded images.
4. RAG retrieves habitability and maintenance statutes relevant to any detected defects.
5. Risk Fusion calculates score with visual defect risk, regulatory habitability risk, and $0.00$ agreement risk.
6. The summary displays the Property Health Score, defect breakdowns, statutory rights, and defect-specific landlord questions.

### 2. Full Multimodal Assessment (Images + Agreement)
1. Upload room inspection photos and attach a draft lease agreement PDF.
2. System extracts and classifies lease clauses into penalty, maintenance, and deposit categories.
3. System runs YOLO11 defect detection on the images.
4. RAG retrieves statutory evidence for both defect findings and flagged lease clauses.
5. Risk Fusion synthesizes visual, agreement, and regulatory risks into the composite Property Health Score.
6. The summary displays full clause audits, defect lists, statutory citations, and landlord negotiation questions.

---

## Limitations

- **Visual Scope**: Detection is strictly limited to visible surface cracks, mold colonies, and pest indications. Concealed structural flaws, hidden electrical faults, or concealed pipe leakages cannot be detected.
- **Corpus Boundary**: RAG retrieval is bounded by the indexed regulatory corpus. Topics outside the included tenancy acts will return an *"Insufficient evidence"* notice.
- **Prototype Risk Weights**: The risk fusion weights (40% Visual, 30% Agreement, 30% Regulatory) are prototype heuristics and do not represent legally codified or certified building inspection standards.
- **Not Legal Advice**: Regulatory citations provide statutory reference only and do not replace professional legal counsel or a formal building survey.

---

## Future Improvements

- Additional defect categories (e.g., water pipe corrosion, ceiling water stains, electrical socket damage).
- Larger, more diverse training datasets for varying room lighting and architectural styles.
- Enhanced detection for fine hairline settling cracks and low-contrast surface blemishes.
- Expanded agreement clause coverage and fine-grained subclause classification.
- Expanded regulatory corpus covering additional state tenancy laws and municipal municipal bylaws.
- Formal RAG evaluation using Precision@K, Recall@K, and domain-expert human relevance grading.
- Data-driven calibration of multimodal risk fusion weights.
- Multilingual agreement parsing and Hindi/regional language statutory retrieval.
- Mobile-optimized responsive interface and camera integration.

---

## Project Status

**Working academic prototype.**

Implemented components:
- Trained YOLO11 defect detection pipeline
- Agreement PDF text extractor & TF-IDF + Logistic Regression clause classifier
- BM25 + SBERT + FAISS hybrid regulatory retriever
- Grounded regulatory RAG chat interface
- Late Risk Fusion and Property Health Score algorithm
- Finding-driven dynamic landlord inquiry generator
- Lightweight HTML5/CSS3/Vanilla JS frontend
- FastAPI backend serving API endpoints and static assets
- Automated unit and integration test suite

---

## Disclaimer

This project is an academic research prototype and is not a substitute for a licensed professional property inspection, civil engineering assessment, or certified legal advice. All regulatory citations and findings should be independently verified against official statutory publications and current government documentation before entering into any legal tenancy commitment.