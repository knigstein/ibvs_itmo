import numpy as np

from vision.depth_map import DepthMapProvider, build_depth_provider, median_Z_for_ibvs


def test_median_Z_for_ibvs():
    assert median_Z_for_ibvs(np.array([0.5, 0.52, 0.48]), 0.1, 2.0, min_valid=2) == 0.5
    assert median_Z_for_ibvs(np.array([0.01, 0.02]), 0.12, 2.0, min_valid=2) is None


def test_depth_map_provider_corners():
    p = DepthMapProvider(0.1, 2.0, min_valid=2)
    depth = np.full((480, 640), 0.55, dtype=np.float32)
    corners = np.array([[10, 20], [100, 20], [100, 200], [10, 200]], dtype=float)
    assert abs(p.Z_for_ibvs(corners, depth) - 0.55) < 1e-5


def test_build_depth_provider():
    prov, mode = build_depth_provider({"depth": {"mode": "none"}})
    assert prov is None and mode == "none"
    prov, mode = build_depth_provider({"depth": {"mode": "depth_map", "z_min_m": 0.1, "z_max_m": 3.0}})
    assert prov is not None and mode == "depth_map"
