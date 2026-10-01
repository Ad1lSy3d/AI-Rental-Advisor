"""
Computer Vision defect detection wrapper using YOLO11.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
from ultralytics import YOLO


class YOLODefectDetector:
    """
    Detector for visible rental property defects (cracks, mold, pests).
    Wraps the trained YOLO11 model.
    """

    DEFAULT_MODEL_PATH = Path("models/yolo11n_rental_home_v2_best.pt")

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path is None:
            candidate = Path.cwd() / self.DEFAULT_MODEL_PATH
            if not candidate.exists():
                candidate = Path(__file__).resolve().parent.parent.parent / self.DEFAULT_MODEL_PATH
            model_path = candidate

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found at: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.names = self.model.names  # e.g. {0: 'crack', 1: 'mold', 2: 'pest'}

    def detect(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        conf_threshold: float = 0.25,
        source_name: str = "image",
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a single image and return detected defects.

        Args:
            image_input: File path, raw bytes, PIL Image, or numpy array.
            conf_threshold: Minimum confidence threshold for detections.
            source_name: Optional identifier or filename for tracing.

        Returns:
            List of detected defect dictionaries.
        """
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            image = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = image_input

        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = self.names.get(cls_id, f"class_{cls_id}")
            confidence = float(box.conf[0].item())
            xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]

            # Determine risk label
            if cls_name == "mold":
                risk_level = "High Risk"
                title = "Mold"
                description = f"Detected moisture/mold defect in {source_name}"
            elif cls_name == "crack":
                risk_level = "Moderate"
                title = "Surface Crack"
                description = f"Detected plaster/surface settling crack in {source_name}"
            elif cls_name == "pest":
                risk_level = "Moderate"
                title = "Pest Infestation"
                description = f"Detected pest activity indicator in {source_name}"
            else:
                risk_level = "Low"
                title = cls_name.capitalize()
                description = f"Detected anomaly in {source_name}"

            detections.append({
                "id": f"defect-{source_name}-{i+1}",
                "class": cls_name,
                "name": title,
                "risk_level": risk_level,
                "confidence": round(confidence, 4),
                "bbox": xyxy,
                "description": description,
                "source_file": source_name,
            })

        return detections


_GLOBAL_DETECTOR: Optional[YOLODefectDetector] = None


def get_yolo_detector() -> YOLODefectDetector:
    """Get or initialize singleton instance of YOLODefectDetector."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = YOLODefectDetector()
    return _GLOBAL_DETECTOR
