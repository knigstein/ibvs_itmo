from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
from ultralytics import YOLO


@dataclass
class CubeSegmentationResult:
    corners: Optional[np.ndarray]  # shape (4, 2) или None
    ok: bool
    meta: Dict[str, Any] = field(default_factory=dict)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Приводит 4 точки к фиксированному порядку: TL, TR, BR, BL"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    rem = [i for i in range(4) if not np.allclose(pts[i], tl) and not np.allclose(pts[i], br)]
    others = pts[rem]
    if others.shape[0] != 2:
        return pts
    tr, bl = (others[0], others[1]) if others[0, 0] < others[1, 0] else (others[1], others[0])
    return np.stack([tl, tr, br, bl], axis=0)


class YOLOFeatureDetector:
    """
    Минимальный YOLO-детектор для IBVS.
    Интерфейс полностью совместим с CubeSegmenter.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        
        weights_path = cfg.get("weights_path", "weights/best.pt")
        self._min_conf = float(cfg.get("min_conf", 0.5))
        
        self._ema_alpha = float(cfg.get("ema_alpha", 0.35))
        self._corners_ema: Optional[np.ndarray] = None
        
        self._model = YOLO(weights_path)
        
    def reset_smoothing(self) -> None:
        """Сбросить сглаживание (вызывать при перезапуске задачи)"""
        self._corners_ema = None
        
    def detect(self, bgr: np.ndarray) -> CubeSegmentationResult:
        """
        Детектирует объект и возвращает 4 угла для IBVS.
        
        Args:
            bgr: кадр в формате OpenCV (H, W, 3), dtype uint8
            
        Returns:
            CubeSegmentationResult с углами или None
        """
        # Проверка на пустой кадр
        if bgr is None or bgr.size == 0:
            return CubeSegmentationResult(None, False, {"reason": "empty_frame"})
        
        results = self._model(bgr, verbose=False, conf=self._min_conf)
        
        if len(results[0].boxes) == 0:
            return CubeSegmentationResult(None, False, {"n_detections": 0})
        
        box = results[0].boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        conf = float(results[0].boxes.conf[0])
        cls_id = int(results[0].boxes.cls[0])
        
        x1, y1, x2, y2 = box
        corners = np.array([
            [x1, y1],  # TL: top-left
            [x2, y1],  # TR: top-right
            [x2, y2],  # BR: bottom-right
            [x1, y2]   # BL: bottom-left
        ], dtype=np.float32)
        
        corners = _order_corners(corners)
        
        if self._ema_alpha > 0.0 and self._corners_ema is not None:
            corners = self._ema_alpha * corners + (1.0 - self._ema_alpha) * self._corners_ema
        self._corners_ema = corners.copy()
        
        meta = {
            "n_detections": len(results[0].boxes),
            "confidence": conf,
            "class_id": cls_id,
            "center": corners.mean(axis=0).tolist(),
            "area": float((x2 - x1) * (y2 - y1)),
            "bbox": [x1, y1, x2, y2]
        }
        
        return CubeSegmentationResult(corners, True, meta)