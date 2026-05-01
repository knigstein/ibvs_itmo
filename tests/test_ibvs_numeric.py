"""Чисто численные проверки IBVS без MuJoCo."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ibvs import IBVS  # noqa: E402


def test_zero_error_zero_velocity():
    cfg_path = ROOT / "config" / "camera.json"
    with open(cfg_path, "r", encoding="utf8") as f:
        cam = json.load(f)
    ibvs = IBVS(cam)
    v, e, _ = ibvs.step(cam["features"], Z=0.5)
    assert np.linalg.norm(e) < 1e-8
    assert np.linalg.norm(v) < 1e-8


def test_error_decreases_small_step():
    with open(ROOT / "config" / "camera.json", "r", encoding="utf8") as f:
        cam = json.load(f)
    ibvs = IBVS(cam, control_coefficient=[0.2] * 6)
    feats = np.asarray(cam["features"], dtype=float)
    # Сдвиг в пикселях (небольшой)
    pert = feats.copy()
    pert[:, 0] += 4.0
    e0 = ibvs.calculate_error(pert)
    v, _, L = ibvs.step(pert, Z=0.5)
    assert np.linalg.norm(v) > 1e-6
    # Локально: L @ (-v) должно быть близко к e (с масштабом коэффициентов)
    pred = L @ (-v)
    assert np.dot(pred, e0) > 0 or np.linalg.norm(e0) < 1e-3


if __name__ == "__main__":
    test_zero_error_zero_velocity()
    test_error_decreases_small_step()
    print("test_ibvs_numeric OK")
