"""Сегментация куба по контрасту (HSV + морфология) → четыре угла minAreaRect."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class CubeSegmentationResult:
    corners: Optional[np.ndarray]
    ok: bool
    meta: Dict[str, Any] = field(default_factory=dict)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    rem = [i for i in range(4) if not np.allclose(pts[i], tl) and not np.allclose(pts[i], br)]
    others = pts[rem]
    if others.shape[0] != 2:
        return pts
    if others[0, 0] < others[1, 0]:
        tr, bl = others[0], others[1]
    else:
        tr, bl = others[1], others[0]
    return np.stack([tl, tr, br, bl], axis=0)


class CubeSegmenter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._hsv_lower = np.array(cfg.get("hsv_lower", [5, 50, 40]), dtype=np.uint8)
        self._hsv_upper = np.array(cfg.get("hsv_upper", [35, 255, 255]), dtype=np.uint8)
        self._blur_ksize = int(cfg.get("blur_ksize", 5)) | 1
        self._morph_ksize = int(cfg.get("morph_ksize", 5))
        self._min_area = float(cfg.get("min_area", 400.0))
        self._ema_alpha = float(cfg.get("ema_alpha", 0.35))
        self._corners_ema: Optional[np.ndarray] = None

    def reset_smoothing(self) -> None:
        self._corners_ema = None

    def detect(self, bgr: np.ndarray) -> CubeSegmentationResult:
        if bgr is None or bgr.size == 0:
            return CubeSegmentationResult(None, False, {"reason": "empty"})

        blurred = cv2.GaussianBlur(bgr, (self._blur_ksize, self._blur_ksize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        k = max(3, self._morph_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meta: Dict[str, Any] = {"n_contours": len(contours)}
        if not contours:
            return CubeSegmentationResult(None, False, meta)

        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        meta["area"] = area
        if area < self._min_area:
            return CubeSegmentationResult(None, False, meta)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        corners = _order_corners(box.astype(np.float32))

        if self._ema_alpha > 0.0 and self._corners_ema is not None:
            corners = self._ema_alpha * corners + (1.0 - self._ema_alpha) * self._corners_ema
        self._corners_ema = corners.copy()
        meta["center"] = corners.mean(axis=0).tolist()
        return CubeSegmentationResult(corners, True, meta)
