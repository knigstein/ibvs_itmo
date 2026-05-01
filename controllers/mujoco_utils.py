import mujoco
import numpy as np


def get_site_jac(model, data, site_id):
    """Якобиан сайта: 6×nv (три строки линейной скорости, три — угловой), в мире."""
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp, jacr])


def get_fullM(model, data):
    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, M, data.qM)
    return M
