"""
Computer Vision module for visible defect detection in rental properties.
"""

from src.cv.detector import YOLODefectDetector, get_yolo_detector

__all__ = [
    "YOLODefectDetector",
    "get_yolo_detector",
]
