"""vision/__init__.py — Модуль детекции объектов для IBVS."""
from vision.cube_segmentation import CubeSegmenter, CubeSegmentationResult
from vision.yolo_detection import YOLOFeatureDetector
from vision.shape_segmenter import GeneralShapeSegmenter, ShapeSegmentationResult
from vision.detector_factory import create_detector

__all__ = [
    "CubeSegmenter",
    "CubeSegmentationResult",
    "YOLOFeatureDetector", 
    "GeneralShapeSegmenter",
    "ShapeSegmentationResult",
    "create_detector",
]