import mujoco
import numpy as np


def get_site_jac(model, data, site_id, jacp=None, jacr=None):
    """Якобиан сайта: 6×nv (линейная и угловая скорость в мире)."""
    if jacp is None:
        jacp = np.zeros((3, model.nv), dtype=np.float64)
    if jacr is None:
        jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp, jacr])


def get_fullM(model, data, M=None):
    if M is None:
        M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, M, data.qM)
    return M
