"""Формы матриц, нормализация, маска осей IBVS."""
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


def test_normalize_inverse_to_pixels(cam_cfg):
    ibvs = IBVS(cam_cfg)
    u0, v0 = 320.0, 240.0
    x, y = ibvs.normalize((u0, v0))
    fx, fy = cam_cfg["focal_length"]
    cx, cy = cam_cfg["principal_point"]
    assert abs(x * fx + cx - u0) < 1e-9
    assert abs(y * fy + cy - v0) < 1e-9


def test_interaction_matrix_shape_and_Z(cam_cfg):
    ibvs = IBVS(cam_cfg)
    L = ibvs.calculate_interaction_matrix(0.1, -0.05, Z=0.4)
    assert L.shape == (2, 6)


def test_Z_must_be_positive(cam_cfg):
    ibvs = IBVS(cam_cfg)
    with pytest.raises(ValueError, match="положительной"):
        ibvs.calculate_interaction_matrix(0.0, 0.0, Z=0.0)


def test_jacobian_stack_shape(cam_cfg):
    ibvs = IBVS(cam_cfg)
    feats = np.asarray(cam_cfg["features"], dtype=float)
    L = ibvs.get_jacobian(feats, Z=0.5)
    assert L.shape == (2 * len(feats), 6)


def test_active_directions_reduce_columns_effective(cam_cfg):
    """Только x,y,z: винт ненулевой при сдвиге, но вращения нулевые (после проекции закона)."""
    ibvs = IBVS(cam_cfg, active_directions=[0, 1, 2], control_coefficient=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    feats = np.asarray(cam_cfg["features"], dtype=float).copy()
    feats[:, 1] += 3.0
    v, _, _ = ibvs.step(feats, Z=0.5)
    assert np.linalg.norm(v[:3]) > 1e-8
    assert np.allclose(v[3:], 0.0)


def test_feature_count_mismatch_raises(cam_cfg):
    ibvs = IBVS(cam_cfg)
    wrong = np.asarray(cam_cfg["features"][:2], dtype=float)
    with pytest.raises(ValueError, match="признак"):
        ibvs.calculate_error(wrong)
