# AI Rental Home Advisor

An AI-powered multimodal system for preliminary assessment of rental
properties using computer vision, OCR, NLP, and retrieval-augmented
generation (RAG).

## Overview

The AI Rental Home Advisor analyzes:

1. Property images
2. Rental agreements
3. Authoritative housing and safety information

The system identifies visible property defects, extracts and classifies
important rental-agreement clauses, retrieves relevant supporting
information, and combines the resulting risk indicators into a
Property Health Score.

> This system is intended as an assistive preliminary assessment tool.
> It does not replace professional property inspection, engineering,
> legal advice, or formal regulatory assessment.

## System Architecture

Property Images
       |
       v
YOLO11 / Faster R-CNN
       |
       v
Visible Defect Detection
       |
       v
Visual Risk
       |
       +----------------------+
                              |
Rental Agreement             |
       |                      |
       v                      |
PaddleOCR                    |
       |                      |
       v                      |
BERT / TF-IDF                |
       |                      |
       v                      |
Clause Classification        |
       |                      |
       v                      |
Agreement Risk               |
                              |
Regulatory Documents          |
       |                      |
       v                      |
BM25 + SBERT + FAISS          |
       |                      |
       v                      |
Relevant Evidence             |
       |                      |
       +----------+-----------+
                  |
                  v
            Risk Assessment
                  |
                  v
             Late Fusion
                  |
                  v
        Property Health Score
                  |
                  v
       Explanation & Recommendations

## Main Components

### 1. Computer Vision

Models:
- YOLO11
- Faster R-CNN

Initial defect classes:
- Surface Crack
- Mold
- Pest

The models are fine-tuned on annotated property images and evaluated
using precision, recall, F1-score, mAP, confusion matrix, and inference
performance.

### 2. Rental Agreement Analysis

Pipeline:

Rental Agreement
    -> PaddleOCR
    -> Text Extraction
    -> Clause Segmentation
    -> BERT Classification

Baseline:
- TF-IDF + Logistic Regression

Clause categories:
- Security Deposit
- Maintenance
- Repairs
- Penalties
- Termination
- Tenant Responsibilities
- Landlord Responsibilities

### 3. Regulatory RAG

Authoritative housing, rental, and safety documents are processed into
a searchable knowledge base.

Retrieval methods:
- BM25
- SBERT embeddings
- FAISS

Hybrid retrieval combines keyword-based and semantic retrieval.

### 4. Risk Assessment

The system calculates:

R_total = wv * Rv + wl * Rl + wr * Rr

Where:

Rv = Visual Risk
Rl = Agreement Risk
Rr = Regulatory/Evidence Risk

The resulting risk is converted into a Property Health Score.

## Dataset Structure

data/
├── cv/
├── agreements/
├── regulatory/
└── property_data/

### Computer Vision

Current data:
- Surface crack images
- Mold images
- Pest images

Images must be annotated with bounding boxes before training the
object-detection models.

### Rental Agreements

Rental agreements are collected from suitable public, synthetic, or
anonymized sources and labelled according to clause category.

### Regulatory Documents

Authoritative housing, rental, and safety documents are used as the
RAG knowledge base.

### Property Data

City-wise rental/property CSV files are maintained as auxiliary
property-market data.

## Tech Stack

- Python
- PyTorch
- Torchvision
- Ultralytics YOLO11
- Faster R-CNN
- OpenCV
- PaddleOCR
- BERT
- Scikit-learn
- TF-IDF
- Logistic Regression
- Sentence Transformers
- BM25
- FAISS
- Pandas
- NumPy
- FastAPI

## Project Structure

```text
data/
models/
src/
notebooks/
tests/
frontend/