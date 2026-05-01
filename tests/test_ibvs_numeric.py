import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ibvs import IBVS  # noqa: E402


def test_zero_error_zero_velocity():
    with open(ROOT / "config" / "camera.json", "r", encoding="utf8") as f:
        cam = json.load(f)
    ibvs = IBVS(cam)
    v, e, _ = ibvs.step(cam["features"], Z=0.5)
    assert np.linalg.norm(e) < 1e-8
    assert np.linalg.norm(v) < 1e-8


def test_nonzero_velocity_on_perturbation():
    with open(ROOT / "config" / "camera.json", "r", encoding="utf8") as f:
        cam = json.load(f)
    ibvs = IBVS(cam, control_coefficient=[0.2] * 6)
    feats = np.asarray(cam["features"], dtype=float)
    pert = feats.copy()
    pert[:, 0] += 4.0
    v, e, _ = ibvs.step(pert, Z=0.5)
    assert np.linalg.norm(e) > 1e-6
    assert np.linalg.norm(v) > 1e-8


if __name__ == "__main__":
    test_zero_error_zero_velocity()
    test_nonzero_velocity_on_perturbation()
    print("test_ibvs_numeric OK")
