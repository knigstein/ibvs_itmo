"""
Кинематика для eye-in-hand IBVS в MuJoCo.

MuJoCo хранит ``data.site_xmat[site_id]`` как матрицу поворота 3×3 (построчно в 9 числах):
вектор в локальном базисе сайта ``v_site`` (столбец) переводится в мировой как
``v_world = R @ v_site``, где ``R = site_xmat.reshape(3, 3)``.

Винт IBVS задаётся в базисе **камеры** (тот же, что у сайта ``real_sense_site`` в MJCF),
порядок: ``[vx, vy, vz, wx, wy, wz]``.
"""
from __future__ import annotations

import numpy as np


def twist_camera_to_world(v_cam: np.ndarray, R_world_from_site: np.ndarray) -> np.ndarray:
    """Переводит винт из базиса сайта камеры в мировой базис MuJoCo.

    Parameters
    ----------
    v_cam : (6,)
        Линейная и угловая часть винта в базисе камеры/сайта.
    R_world_from_site : (3, 3)
        Матрица поворота сайта (как ``data.site_xmat[id].reshape(3, 3)``).
    """
    R = np.asarray(R_world_from_site, dtype=float).reshape(3, 3)
    v = np.asarray(v_cam, dtype=float).reshape(6)
    lin_w = R @ v[:3]
    ang_w = R @ v[3:]
    return np.concatenate([lin_w, ang_w])
