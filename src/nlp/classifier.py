"""
Agreement clause classification using the trained TF-IDF + Logistic Regression pipeline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.nlp.extractor import AgreementClauseExtractor


class AgreementClauseClassifier:
    """
    Classifies rental agreement clauses into standard categories:
    - Security Deposit
    - Maintenance
    - Repairs
    - Penalties
    - Termination
    - Tenant Responsibilities
    - Landlord Responsibilities
    """

    DEFAULT_MODEL_PATH = Path("models/agreement_tfidf_logreg.pkl")

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the classifier by loading the trained pipeline.

        Args:
            model_path: Path to the serialized model pickle file.
        """
        if model_path is None:
            # Look relative to current working directory or file parent
            candidate = Path.cwd() / self.DEFAULT_MODEL_PATH
            if not candidate.exists():
                candidate = Path(__file__).resolve().parent.parent.parent / self.DEFAULT_MODEL_PATH
            model_path = candidate

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.pipeline = pickle.load(f)

        self.classes = list(self.pipeline.classes_)

    def predict_clause(self, clause_text: str) -> Dict[str, Any]:
        """
        Classify a single clause string.

        Args:
            clause_text: Text of the agreement clause.

        Returns:
            Dictionary with:
            - clause_text: Original text
            - predicted_class: Winning label string
            - prediction_probability: Confidence score (float 0.0 to 1.0)
        """
        results = self.predict_clauses([clause_text])
        return results[0]

    def predict_clauses(self, clauses: List[str]) -> List[Dict[str, Any]]:
        """
        Classify a batch of clause strings.

        Args:
            clauses: List of clause texts.

        Returns:
            List of dictionaries, each containing:
            - clause_text
            - predicted_class
            - prediction_probability
        """
        if not clauses:
            return []

        probabilities = self.pipeline.predict_proba(clauses)
        predictions = self.pipeline.predict(clauses)

        results: List[Dict[str, Any]] = []
        for text, pred_class, probs in zip(clauses, predictions, probabilities):
            confidence = float(probs.max())
            results.append({
                "clause_text": text,
                "predicted_class": str(pred_class),
                "prediction_probability": round(confidence, 4),
            })

        return results

    def classify_pdf(
        self, pdf_path: Union[str, Path], min_words: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Extract clauses from a rental agreement PDF and classify each clause.

        Args:
            pdf_path: Path to the target PDF file.
            min_words: Minimum word threshold for clause segmentation.

        Returns:
            List of classification outputs for all extracted clauses.
        """
        extractor = AgreementClauseExtractor(min_words=min_words)
        clauses = extractor.extract_from_pdf(pdf_path)
        return self.predict_clauses(clauses)


def classify_rental_agreement(
    pdf_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    min_words: int = 6,
) -> List[Dict[str, Any]]:
    """
    Convenience function to extract and classify clauses from a rental agreement PDF.
    """
    classifier = AgreementClauseClassifier(model_path=model_path)
    return classifier.classify_pdf(pdf_path, min_words=min_words)
