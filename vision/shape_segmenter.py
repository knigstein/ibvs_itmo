"""shape_segmenter.py — Universal geometric shape segmenter for IBVS (top-down).
Supports two strategies:
  - "real": Edge/contrast-based + morphological closing (robust to shadows, holes, texture)
  - "sim":  HSV color-based (optimized for clean simulation environments)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class ShapeType(Enum):
    UNKNOWN = "unknown"
    CIRCLE = "circle"
    POLYGON = "polygon"  # square, rectangle, triangle


@dataclass
class ShapeSegmentationResult:
    corners: Optional[np.ndarray]  # (4, 2) strictly [TL, TR, BR, BL]
    ok: bool
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CubeSegmentationResult:
    """Legacy alias for backward compatibility."""
    corners: Optional[np.ndarray]
    ok: bool
    meta: Dict[str, Any] = field(default_factory=dict)


# ================= CORE GEOMETRY =================

def _order_corners_robust(pts: np.ndarray) -> np.ndarray:
    """
    Устойчивая сортировка 4 точек в порядке: TL, TR, BR, BL.
    Работает корректно даже для повернутых фигур.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL

    return rect


def _get_stable_ibvs_corners(contour: np.ndarray, shape: ShapeType) -> np.ndarray:
    """Возвращает 4 стабильных признака для IBVS."""
    if shape == ShapeType.CIRCLE:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        r = float(radius)
        # Кардинальные точки относительно осей изображения (не вращаются с объектом)
        pts = np.array([
            [cx, cy - r],  # Top
            [cx + r, cy],  # Right
            [cx, cy + r],  # Bottom
            [cx - r, cy]   # Left
        ], dtype=np.float32)
        return _order_corners_robust(pts)
    else:
        # Осевой bounding box для полигонов (стабильнее minAreaRect для IBVS)
        x, y, w, h = cv2.boundingRect(contour)
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


# ================= DETECTOR =================

class GeneralShapeSegmenter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._strategy = cfg.get("strategy", "real")  # "real" | "sim" | "auto"
        self._ema_alpha = float(cfg.get("ema_alpha", 0.35))
        self._corners_ema: Optional[np.ndarray] = None

        # Настройки для стратегии "real"
        self._blur_ksize = int(cfg.get("blur_ksize", 7)) | 1
        self._canny_t1 = int(cfg.get("canny_t1", 50))
        self._canny_t2 = int(cfg.get("canny_t2", 150))
        self._morph_kernel = int(cfg.get("morph_kernel", 25))  # Заполняет дырки до ~12px
        self._min_area = float(cfg.get("min_area", 500))

        # Настройки для стратегии "sim"
        self._hsv_lower = np.array(cfg.get("hsv_lower", [5, 50, 40]), dtype=np.uint8)
        self._hsv_upper = np.array(cfg.get("hsv_upper", [35, 255, 255]), dtype=np.uint8)
        self._sim_morph_k = int(cfg.get("sim_morph_k", 5))

    def reset_smoothing(self) -> None:
        self._corners_ema = None

    def detect(self, bgr: np.ndarray) -> ShapeSegmentationResult:
        if bgr is None or bgr.size == 0:
            return ShapeSegmentationResult(None, False, {"reason": "empty_input"})

        if self._strategy == "sim":
            res = self._detect_sim(bgr)
        elif self._strategy == "real":
            res = self._detect_real(bgr)
        else:  # auto: пробуем HSV, если не вышло → контуры
            res = self._detect_sim(bgr)
            if not res.ok:
                res = self._detect_real(bgr)

        if res.ok and self._ema_alpha > 0.0 and self._corners_ema is not None:
            res.corners = self._ema_alpha * res.corners + (1.0 - self._ema_alpha) * self._corners_ema
        if res.ok:
            self._corners_ema = res.corners.copy()
            res.meta["center"] = res.corners.mean(axis=0).tolist()
        return res

    # ---------- STRATEGY: REAL WORLD (Contrast/Edges) ----------
    def _detect_real(self, bgr: np.ndarray) -> ShapeSegmentationResult:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self._blur_ksize, self._blur_ksize), 0)
        
        # Поиск физических границ (игнорирует плавные тени)
        edges = cv2.Canny(blurred, self._canny_t1, self._canny_t2)
        
        # Морфологическое ЗАКРЫТИЕ: склеивает разрывы от дырок и теней
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._morph_kernel, self._morph_kernel))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meta = {"n_contours": len(contours), "strategy": "real"}

        best_cnt = self._select_best_contour(contours, meta)
        if best_cnt is None:
            meta["reason"] = "no_valid_contour"
            return ShapeSegmentationResult(None, False, meta)

        shape = ShapeType.CIRCLE if meta.get("circularity", 0) > 0.7 else ShapeType.POLYGON
        corners = _get_stable_ibvs_corners(best_cnt, shape)
        meta["shape"] = shape.value
        meta["confidence"] = meta.get("circularity", 0.8)
        return ShapeSegmentationResult(corners, True, meta)

    # ---------- STRATEGY: SIMULATION (HSV Color) ----------
    def _detect_sim(self, bgr: np.ndarray) -> ShapeSegmentationResult:
        blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        
        k = max(3, self._sim_morph_k)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meta = {"n_contours": len(contours), "strategy": "sim"}

        if not contours:
            meta["reason"] = "no_contours"
            return ShapeSegmentationResult(None, False, meta)

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self._min_area:
            meta["reason"] = "area_too_small"
            return ShapeSegmentationResult(None, False, meta)

        shape = ShapeType.CIRCLE if _compute_circularity(c) > 0.7 else ShapeType.POLYGON
        corners = _get_stable_ibvs_corners(c, shape)
        meta["area"] = float(area)
        meta["shape"] = shape.value
        meta["confidence"] = 0.9
        return ShapeSegmentationResult(corners, True, meta)

    # ---------- CONTOUR SELECTION ----------
    def _select_best_contour(self, contours: list, meta: dict) -> Optional[np.ndarray]:
        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area:
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0: continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < 0.3: continue  # отсекаем линии/швы
            
            score = area * (0.4 + 0.6 * circ)
            valid.append((c, area, circ, score))

        if not valid: return None
        best = max(valid, key=lambda x: x[3])
        meta["area"] = float(best[1])
        meta["circularity"] = best[2]
        return best[0]


# ================= HELPERS =================
def _compute_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    return 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0


# ================= BACKWARD COMPATIBILITY =================
CubeSegmenter = GeneralShapeSegmenter