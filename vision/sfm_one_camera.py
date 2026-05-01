"""
Две ракурса ОДНОЙ камеры на манипуляторе (два положения робота), без второго сенсора.

Кадр t−1 и кадр t: те же внутренние параметры K, известное относительное положение камеры
из кинематики (T_w_c). Триангуляция углов куба → плотная синтетическая depth_m в текущем
кадре (как у выровненной карты глубины для IBVS).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def K_from_camera_json(cam_cfg: Dict[str, Any]) -> np.ndarray:
    fl = np.asarray(cam_cfg["focal_length"], dtype=float).reshape(2)
    pp = np.asarray(cam_cfg["principal_point"], dtype=float).reshape(2)
    return np.array(
        [[fl[0], 0.0, pp[0]], [0.0, fl[1], pp[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def match_quad_corners_cyclic(uv_a: np.ndarray, uv_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Сопоставление 4 углов между кадрами: перебор циклических сдвигов (один объект, одна камера).
    """
    a = np.asarray(uv_a, dtype=np.float64).reshape(4, 2)
    b = np.asarray(uv_b, dtype=np.float64).reshape(4, 2)
    best_s, best_cost = 0, float("inf")
    for s in range(4):
        cost = sum(np.sum((a[i] - b[(i + s) % 4]) ** 2) for i in range(4))
        if cost < best_cost:
            best_cost = cost
            best_s = s
    b_m = np.stack([b[(i + best_s) % 4] for i in range(4)], axis=0)
    return a.copy(), b_m


def relative_pose_cam2_from_cam1(T_w_c1: np.ndarray, T_w_c2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Точка в системе камеры в положении 1: X1. В системе камеры в положении 2: X2 = R @ X1 + t.
    MuJoCo: p_w = R_wc @ p_c + t_wc.
    """
    R1 = T_w_c1[:3, :3]
    t1 = T_w_c1[:3, 3]
    R2 = T_w_c2[:3, :3]
    t2 = T_w_c2[:3, 3]
    R_21 = R2.T @ R1
    t_21 = R2.T @ (t1 - t2)
    return R_21.astype(np.float64), t_21.astype(np.float64)


def triangulate_points_cam1_frame(
    K: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
    R_21: np.ndarray,
    t_21: np.ndarray,
) -> Optional[np.ndarray]:
    """Возвращает N×3 точек в системе координат камеры в первом (опорном) положении."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R_21, t_21.reshape(3, 1)])
    x1 = np.asarray(uv1, dtype=np.float64).T
    x2 = np.asarray(uv2, dtype=np.float64).T
    if x1.shape[1] < 1 or x1.shape[1] != x2.shape[1]:
        return None
    X_h = cv2.triangulatePoints(P1, P2, x1, x2)
    w = X_h[3]
    ok = np.abs(w) > 1e-10
    if not np.all(ok):
        return None
    X = (X_h[:3] / w).T
    if np.any(X[:, 2] < 0.02):
        return None
    return X


def _points_cam1_to_cam2(X1: np.ndarray, T_w_c1: np.ndarray, T_w_c2: np.ndarray) -> np.ndarray:
    """X1 (N×3) в первой камере → X2 (N×3) во второй (текущей)."""
    N = X1.shape[0]
    hom = np.hstack([X1, np.ones((N, 1))])
    T_c2_w = np.linalg.inv(T_w_c2)
    out = (T_c2_w @ (T_w_c1 @ hom.T)).T[:, :3]
    return out


def _fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Плоскость n·x = d, ||n||=1."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = float(n @ c)
    return n, d


def dense_depth_z_from_plane(
    K_inv: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    height: int,
    width: int,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """
    Для каждого пикселя луч o + λ d, d = K^{-1}[u,v,1]; пересечение с n·x = plane_d.
    Z = (λ d)_z. OpenCV / IBVS: ось Z вперёд от камеры.
    """
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    hom = np.stack([u.ravel(), v.ravel(), np.ones(width * height, dtype=np.float64)], axis=0)
    dirs = (K_inv @ hom).T.reshape(height, width, 3)
    denom = dirs @ plane_n
    lam = np.full((height, width), np.nan, dtype=np.float64)
    m = np.abs(denom) > 1e-9
    lam[m] = plane_d / denom[m]
    pts_z = (lam[..., None] * dirs)[..., 2]
    pts_z = np.where((lam > 0) & (pts_z > z_min) & (pts_z < z_max), pts_z, np.nan)
    return pts_z


class OneCameraTwoPoseSfM:
    """
    Буфер между кадрами: предыдущие углы и T_w_c предыдущего положения одной камеры.
    """

    def __init__(
        self,
        K: np.ndarray,
        height: int,
        width: int,
        *,
        min_baseline_m: float = 0.003,
        z_min: float = 0.12,
        z_max: float = 2.5,
    ):
        self._K = np.asarray(K, dtype=np.float64)
        self._Kinv = np.linalg.inv(self._K)
        self._H = int(height)
        self._W = int(width)
        self._min_bl = float(min_baseline_m)
        self._z_min = float(z_min)
        self._z_max = float(z_max)
        self._prev: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def reset(self) -> None:
        self._prev = None

    def update(
        self,
        corners_cur: Optional[np.ndarray],
        T_w_cur: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        corners_cur: углы на текущем кадре (4×2) или None.
        T_w_cur: 4×4 одной камеры в текущем положении (MuJoCo site).
        Возвращает depth_m (H,W) в метрах для текущего кадра или None (первый кадр / вырождение).
        """
        T_w_cur = np.asarray(T_w_cur, dtype=np.float64).reshape(4, 4)
        if corners_cur is None:
            self._prev = None
            return None
        uv2 = np.asarray(corners_cur, dtype=np.float64).reshape(4, 2)
        if self._prev is None:
            self._prev = (uv2.copy(), T_w_cur.copy())
            return None

        uv1, T_w_prev = self._prev
        R21, t21 = relative_pose_cam2_from_cam1(T_w_prev, T_w_cur)
        if float(np.linalg.norm(t21)) < self._min_bl:
            self._prev = (uv2.copy(), T_w_cur.copy())
            return None

        uv1m, uv2m = match_quad_corners_cyclic(uv1, uv2)
        X1 = triangulate_points_cam1_frame(self._K, uv1m, uv2m, R21, t21)
        if X1 is None:
            self._prev = (uv2.copy(), T_w_cur.copy())
            return None

        X2 = _points_cam1_to_cam2(X1, T_w_prev, T_w_cur)
        if np.any(X2[:, 2] < self._z_min * 0.5):
            self._prev = (uv2.copy(), T_w_cur.copy())
            return None

        n, d = _fit_plane(X2)
        depth_m = dense_depth_z_from_plane(
            self._Kinv, n, d, self._H, self._W, self._z_min, self._z_max
        )
        self._prev = (uv2.copy(), T_w_cur.copy())
        return depth_m
