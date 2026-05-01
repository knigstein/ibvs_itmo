"""Одна камера, два положения — триангуляция без второго сенсора."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vision.sfm_one_camera import (
    K_from_camera_json,
    match_quad_corners_cyclic,
    relative_pose_cam2_from_cam1,
    triangulate_points_cam1_frame,
)


def test_match_quad_shift():
    a = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    b = np.roll(a.copy(), 2, axis=0) + 0.01 * np.random.RandomState(0).randn(4, 2)
    am, bm = match_quad_corners_cyclic(a, b)
    assert np.allclose(am, a)
    assert np.max(np.linalg.norm(am - bm, axis=1)) < 0.05


def test_triangulate_synthetic_two_poses_one_camera():
    """Две позы одной камеры в мире; точка впереди; восстановление глубины."""
    fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
    # Камера 1: начало, смотрит по +Z (стандарт OpenCV)
    T_w_c1 = np.eye(4)
    # Камера 2: сдвиг вправо в мире на 5 см (база между «экспозициями» одной камеры)
    T_w_c2 = np.eye(4)
    T_w_c2[0, 3] = 0.05

    P_w = np.array([0.0, 0.0, 1.0])
    def project(T_w_c, p_w):
        R, t = T_w_c[:3, :3], T_w_c[:3, 3]
        p_c = R.T @ (p_w - t)
        u = fx * p_c[0] / p_c[2] + cx
        v = fy * p_c[1] / p_c[2] + cy
        return np.array([u, v])

    uv1 = project(T_w_c1, P_w).reshape(1, 2)
    uv2 = project(T_w_c2, P_w).reshape(1, 2)

    R21, t21 = relative_pose_cam2_from_cam1(T_w_c1, T_w_c2)
    X = triangulate_points_cam1_frame(K, uv1, uv2, R21, t21)
    assert X is not None
    assert X.shape == (1, 3)
    np.testing.assert_allclose(X[0], [0.0, 0.0, 1.0], atol=1e-3)


def test_K_from_camera_json():
    K = K_from_camera_json(
        {"focal_length": [100.0, 101.0], "principal_point": [50.0, 40.0]}
    )
    assert K[0, 0] == 100 and K[1, 1] == 101 and K[0, 2] == 50
