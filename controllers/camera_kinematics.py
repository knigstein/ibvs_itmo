"""
Преобразование винта IBVS (базис сайта камеры) в мировой базис MuJoCo.

``data.site_xmat[site_id].reshape(3, 3)`` задаёт R: v_world = R @ v_site.
"""
from __future__ import annotations

import numpy as np


def twist_camera_to_world(v_cam: np.ndarray, R_world_from_site: np.ndarray) -> np.ndarray:
    R = np.asarray(R_world_from_site, dtype=float).reshape(3, 3)
    v = np.asarray(v_cam, dtype=float).reshape(6)
    lin_w = R @ v[:3]
    ang_w = R @ v[3:]
    return np.concatenate([lin_w, ang_w])
