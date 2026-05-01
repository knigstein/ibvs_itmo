"""
Согласованность закона управления с первым порядком: L @ v должно быть направлено против ошибки e,
т.е. dot(e, L @ v) < 0 (локальное уменьшение ||e|| при ṡ ≈ L v).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ibvs import IBVS

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cam_cfg():
    with open(_ROOT / "config" / "camera.json", "r", encoding="utf8") as f:
        return json.load(f)


@pytest.mark.parametrize("du,dv", [(5.0, 0.0), (0.0, -4.0), (-3.0, 6.0)])
def test_velocity_opposes_error_first_order(cam_cfg, du, dv):
    ibvs = IBVS(cam_cfg, control_coefficient=[0.15] * 6)
    feats = np.asarray(cam_cfg["features"], dtype=float).copy()
    feats[:, 0] += du
    feats[:, 1] += dv
    v, e, L = ibvs.step(feats, Z=0.55)
    if np.linalg.norm(e) < 1e-10:
        pytest.skip("нулевая ошибка")
    lv = L @ v
    assert float(np.dot(e, lv)) < 0.0


def test_first_order_integrated_step_reduces_norm(cam_cfg):
    ibvs = IBVS(cam_cfg, control_coefficient=[0.2] * 6)
    feats = np.asarray(cam_cfg["features"], dtype=float).copy()
    feats += np.array([[2.0, -1.0], [-2.0, 1.0], [1.0, 2.0], [-1.0, -2.0]])
    v, e, L = ibvs.step(feats, Z=0.5)
    dt = 0.02
    e1 = e + (L @ v) * dt
    assert np.linalg.norm(e1) < np.linalg.norm(e)
