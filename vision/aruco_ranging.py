from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class ArucoDetection:
    ok: bool
    distance_m: Optional[float]
    corners: Optional[np.ndarray]
    marker_id: Optional[int]


class ArucoMotionRanging:
    """
    Оценка расстояния по ArUco + калибровка масштаба глубины через известный шаг по Z.
    """

    def __init__(self, cfg: Dict[str, Any], camera_cfg: Dict[str, Any]):
        self.enabled = bool(cfg.get("enabled", False))
        self.marker_size_m = float(cfg.get("marker_size_m", 0.04))
        self.target_id = cfg.get("marker_id", None)
        self.known_start_distance_m = float(cfg.get("known_start_distance_m", 0.50))
        self.probe_descent_m = max(0.0, float(cfg.get("probe_descent_m", 0.05)))
        self.min_observed_delta_m = float(cfg.get("min_observed_delta_m", 0.005))

        fx, fy = camera_cfg.get("focal_length", [600.0, 600.0])
        self._f_eff = 0.5 * (float(fx) + float(fy))

        dict_name = str(cfg.get("dictionary", "DICT_4X4_50"))
        dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
        self._dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self._params = cv2.aruco.DetectorParameters()

        self._raw_top_z: Optional[float] = None
        self._raw_bottom_z: Optional[float] = None
        self.depth_scale: float = 1.0
        self.calibrated_start_distance_m: Optional[float] = None

    @staticmethod
    def _perimeter_px(corners: np.ndarray) -> float:
        pts = np.asarray(corners, dtype=float).reshape(4, 2)
        edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        return float(edges.sum())

    def _distance_from_corners(self, corners: np.ndarray) -> Optional[float]:
        perim = self._perimeter_px(corners)
        if perim <= 1e-9 or self.marker_size_m <= 0.0:
            return None
        side_px = perim / 4.0
        return float((self._f_eff * self.marker_size_m) / max(side_px, 1e-9))

    def detect(self, bgr: np.ndarray) -> ArucoDetection:
        if not self.enabled or bgr is None or bgr.size == 0:
            return ArucoDetection(False, None, None, None)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _rej = cv2.aruco.detectMarkers(gray, self._dict, parameters=self._params)
        if ids is None or len(ids) == 0:
            return ArucoDetection(False, None, None, None)

        ids_flat = ids.reshape(-1)
        pick = 0
        if self.target_id is not None:
            matches = np.where(ids_flat == int(self.target_id))[0]
            if matches.size == 0:
                return ArucoDetection(False, None, None, None)
            pick = int(matches[0])

        c = np.asarray(corners[pick], dtype=float).reshape(4, 2)
        dist = self._distance_from_corners(c)
        return ArucoDetection(dist is not None, dist, c, int(ids_flat[pick]))

    def begin_probe(self, top_detection: ArucoDetection) -> None:
        self._raw_top_z = float(top_detection.distance_m) if top_detection.distance_m is not None else None
        self._raw_bottom_z = None
        self.depth_scale = 1.0
        self.calibrated_start_distance_m = None

    def capture_probe_bottom(self, bottom_detection: ArucoDetection) -> None:
        if bottom_detection.distance_m is None:
            return
        self._raw_bottom_z = float(bottom_detection.distance_m)

    def finalize_probe(self) -> None:
        if self._raw_top_z is None:
            self.depth_scale = 1.0
            self.calibrated_start_distance_m = None
            return

        scale_abs = self.known_start_distance_m / max(self._raw_top_z, 1e-9)
        scale_mbr = None
        if self._raw_bottom_z is not None and self.probe_descent_m > 0.0:
            observed = self._raw_top_z - self._raw_bottom_z
            if observed > self.min_observed_delta_m:
                scale_mbr = self.probe_descent_m / observed

        if scale_mbr is None:
            self.depth_scale = float(scale_abs)
        else:
            self.depth_scale = float(0.5 * (scale_abs + scale_mbr))
        self.calibrated_start_distance_m = float(self._raw_top_z * self.depth_scale)

