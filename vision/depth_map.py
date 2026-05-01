"""Карта глубины с сенсора (RealSense и т.п.), выровненная под RGB — одна камера на роботе."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def median_Z_for_ibvs(
    z_samples: np.ndarray,
    z_min: float,
    z_max: float,
    min_valid: int = 2,
) -> Optional[float]:
    """Робастная глубина по выборке значений Z (метры)."""
    z = np.asarray(z_samples, dtype=float).reshape(-1)
    v = z[np.isfinite(z) & (z > z_min) & (z < z_max)]
    if v.size < min_valid:
        return None
    return float(np.median(v))


class DepthMapProvider:
    """Глубина по пикселям (Z16 × scale, выровнено под цвет)."""

    def __init__(self, z_min: float, z_max: float, min_valid: int = 2):
        self._z_min = float(z_min)
        self._z_max = float(z_max)
        self._min_valid = int(min_valid)

    def Z_for_ibvs(self, corners: np.ndarray, depth_m: np.ndarray) -> Optional[float]:
        """
        depth_m: (H, W) расстояние в метрах, то же разрешение и выравнивание, что у BGR.
        """
        if depth_m is None or depth_m.size == 0:
            return None
        h, w = depth_m.shape[:2]
        zs = []
        for pt in np.asarray(corners, dtype=float).reshape(4, 2):
            u = int(np.clip(round(pt[0]), 0, w - 1))
            v = int(np.clip(round(pt[1]), 0, h - 1))
            z = float(depth_m[v, u])
            if z > self._z_min and z < self._z_max:
                zs.append(z)
        if len(zs) < self._min_valid:
            return None
        return float(np.median(zs))


def build_depth_provider(robot_cfg: Dict[str, Any]) -> Tuple[Optional[DepthMapProvider], str]:
    """
    mode: none | depth_map
    depth_map — в контуре передаётся depth_m (FSM). Источник задаётся в robot.json:
      depth.source: sensor (по умолчанию) — RealSense и т.п.;
      depth.source: sfm_two_view — только в BaseProgSim: одна камера, два положения робота.
    """
    dcfg = robot_cfg.get("depth") or {}
    mode = str(dcfg.get("mode", "none")).lower()
    z_min = float(dcfg.get("z_min_m", 0.12))
    z_max = float(dcfg.get("z_max_m", 3.0))
    min_valid = int(dcfg.get("min_valid_corners", 2))
    if mode == "depth_map":
        return DepthMapProvider(z_min, z_max, min_valid), "depth_map"
    return None, "none"
