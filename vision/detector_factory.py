"""detector_factory.py — Фабрика детекторов с единым интерфейсом."""
from typing import Optional, Dict, Any
from vision.cube_segmentation import CubeSegmenter as HsvCubeSegmenter
from vision.yolo_detection import YOLOFeatureDetector
from vision.shape_segmenter import GeneralShapeSegmenter


def create_detector(detector_type: str, config: Optional[Dict[str, Any]] = None):
    """
    Фабрика детекторов. Создаёт нужный детектор по типу.
    
    Args:
        detector_type: "hsv" | "yolo" | "universal" | "auto"
        config: параметры из robot.json["vision"]
    
    Returns:
        Объект с методом .detect(bgr) -> Result(corners, ok, meta)
    """
    cfg = config or {}
    
    if detector_type == "hsv":
        print("Детектор: HSV (цветовая сегментация)")
        return HsvCubeSegmenter(cfg)
    
    elif detector_type == "yolo":
        print("Детектор: YOLO (нейросеть)")
        return YOLOFeatureDetector(cfg)
    
    elif detector_type == "universal":
        strategy = cfg.get("strategy", "real")
        print(f"Детектор: Universal (стратегия: {strategy})")
        return GeneralShapeSegmenter(cfg)
    
    elif detector_type == "auto":
        print("Детектор: Auto (HSV → fallback Universal)")
        return _AutoDetector(cfg)
    
    else:
        print(f" Неизвестный тип '{detector_type}', использую 'universal'")
        return GeneralShapeSegmenter(cfg)


class _AutoDetector:
    """Обёртка: пробует HSV, fallback на universal."""
    def __init__(self, config: Dict[str, Any]):
        self._hsv = HsvCubeSegmenter(config)
        self._universal = GeneralShapeSegmenter(config)
        self._last_method = None
        
    def detect(self, bgr):
        # Пробуем HSV (быстрый)
        res = self._hsv.detect(bgr)
        if res.ok:
            self._last_method = "hsv"
            return res
        # Fallback на универсальный
        res = self._universal.detect(bgr)
        self._last_method = "universal" if res.ok else None
        return res
        
    def reset_smoothing(self):
        self._hsv.reset_smoothing()
        self._universal.reset_smoothing()