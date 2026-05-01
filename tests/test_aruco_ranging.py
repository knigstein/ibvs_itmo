import pytest

pytest.importorskip("cv2")

from vision.aruco_ranging import ArucoDetection, ArucoMotionRanging


def _det(z: float) -> ArucoDetection:
    return ArucoDetection(ok=True, distance_m=float(z), corners=None, marker_id=0)


def test_probe_requires_bottom_sample_when_enabled():
    cfg = {
        "enabled": True,
        "known_start_distance_m": 0.5,
        "probe_descent_m": 0.05,
        "require_bottom_for_probe_scale": True,
    }
    cam = {"focal_length": [600.0, 600.0]}
    r = ArucoMotionRanging(cfg, cam)
    r.begin_probe(_det(0.52))
    ok = r.finalize_probe()
    assert not ok
    assert r.calibrated_start_distance_m is None
    assert r.last_probe_reason == "no_bottom_detection"


def test_probe_uses_strongest_bottom_delta():
    cfg = {
        "enabled": True,
        "known_start_distance_m": 0.5,
        "probe_descent_m": 0.05,
        "require_bottom_for_probe_scale": True,
        "min_observed_delta_m": 0.001,
        "min_depth_scale": 0.5,
        "max_depth_scale": 2.0,
    }
    cam = {"focal_length": [600.0, 600.0]}
    r = ArucoMotionRanging(cfg, cam)
    r.begin_probe(_det(0.50))
    r.capture_probe_bottom(_det(0.48))
    r.capture_probe_bottom(_det(0.45))
    r.capture_probe_bottom(_det(0.47))
    ok = r.finalize_probe()
    assert ok
    assert r.calibrated_start_distance_m is not None
    assert r.last_probe_reason in ("ok", "fallback_abs_scale")
