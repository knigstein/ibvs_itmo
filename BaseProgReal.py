from __future__ import annotations

import json
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from ibvs import IBVS
from vision import ArucoMotionRanging, CubeSegmenter
from vision.depth_map import build_depth_provider

try:
    from UniversalRobotAPI import UniversalRobotAPI as UrRtdeRobot
except ImportError:
    UrRtdeRobot = None


def load_configs():
    root = Path(__file__).resolve().parent
    with open(root / "config" / "robot.json", "r", encoding="utf8") as f:
        robot_cfg = json.load(f)
    with open(root / "config" / "camera.json", "r", encoding="utf8") as f:
        cam_cfg = json.load(f)
    return robot_cfg, cam_cfg


class RealSenseGrabber:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, *, with_depth: bool = False):
        if rs is None:
            raise RuntimeError("Установите пакет pyrealsense2")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._with_depth = with_depth
        if with_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._profile = self.pipeline.start(cfg)
        self._align = rs.align(rs.stream.color) if with_depth else None
        self._depth_scale = 0.001
        if with_depth:
            dev = self._profile.get_device()
            ds = dev.first_depth_sensor()
            self._depth_scale = float(ds.get_depth_scale())

    def read(self):
        frames = self.pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        color = frames.get_color_frame()
        if not color:
            return None, None
        bgr = np.asanyarray(color.get_data())
        if not self._with_depth:
            return bgr, None
        df = frames.get_depth_frame()
        if not df:
            return bgr, None
        depth_m = np.asanyarray(df.get_data(), dtype=np.float64) * self._depth_scale
        return bgr, depth_m

    def stop(self):
        self.pipeline.stop()


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    k = rv / theta
    kx, ky, kz = k
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _probe_axis_in_base(robot: "UrRtdeRobot", axis_tcp: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_tcp, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / n
    try:
        tcp_pose = np.asarray(robot.rtde_r.getActualTCPPose(), dtype=float).reshape(6)
        R_base_tcp = _rotvec_to_matrix(tcp_pose[3:])
        axis_base = R_base_tcp @ axis
        nb = float(np.linalg.norm(axis_base))
        if nb > 1e-9:
            return axis_base / nb
    except Exception:
        pass
    return axis


def main():
    robot_cfg, cam_cfg = load_configs()
    if UrRtdeRobot is None:
        raise RuntimeError("Нужен модуль ur_rtde и UniversalRobotAPI.py")
    if rs is None:
        raise RuntimeError("Нужен pyrealsense2")

    dcfg = robot_cfg.get("depth") or {}
    depth_provider, _depth_mode = build_depth_provider(robot_cfg)
    # sfm_two_view — только сим (одна камера, два положения); RealSense depth не открываем
    use_sensor_depth = depth_provider is not None and str(dcfg.get("source", "sensor")).lower() != "sfm_two_view"

    ip = robot_cfg.get("ur_ip", "192.168.90.107")
    robot = UrRtdeRobot(ip)
    cam = RealSenseGrabber(with_depth=use_sensor_depth)
    ibvs = IBVS(cam_cfg)
    segmenter = CubeSegmenter(robot_cfg.get("vision", {}))
    z_default = float(robot_cfg.get("default_Z", 0.5)) + float(robot_cfg.get("ibvs_depth_offset_m", 0.0))
    depth_off = float(robot_cfg.get("ibvs_depth_offset_m", 0.0))
    vmax = float(robot_cfg.get("max_ibvs_speed", 0.25))
    dt = float(robot_cfg.get("control_dt", 1.0 / 30.0))
    zc = robot_cfg.get("z_calibration") or {}
    aruco_ranging = ArucoMotionRanging(zc, cam_cfg)
    align_hold = int(robot_cfg.get("align_hold_steps", 8))
    align_cnt = 0
    descent_speed = abs(float(zc.get("grasp_descent_speed_mps", 0.035)))
    descent_m = max(0.0, float(zc.get("grasp_descent_m", 0.12)))
    lift_m = max(0.0, float(zc.get("lift_after_grasp_m", descent_m)))
    probe_speed = abs(float(zc.get("probe_speed_mps", 0.03)))
    descent_sign = float(zc.get("descent_direction_sign", 1.0))
    if abs(descent_sign) < 1e-9:
        descent_sign = 1.0
    calib_max_retries = max(0, int(zc.get("max_retries", 2)))
    calib_retry_count = 0
    axis_tcp = np.asarray(zc.get("lock_axis_local", [0.0, 0.0, 1.0]), dtype=float).reshape(3)

    class Mode(Enum):
        CALIBRATION_MARKER = auto()
        CALIBRATION_DESCEND = auto()
        CALIBRATION_ASCEND = auto()
        IBVS = auto()
        DESCENT_TO_GRASP = auto()
        LIFT_AFTER_GRASP = auto()

    mode = Mode.CALIBRATION_MARKER if aruco_ranging.enabled else Mode.IBVS
    calib_remaining = 0.0
    grasp_remaining = 0.0

    def _finalize_calibration() -> None:
        nonlocal mode, z_default, calib_retry_count
        ok = aruco_ranging.finalize_probe()
        if ok and aruco_ranging.calibrated_start_distance_m is not None:
            z_default = aruco_ranging.calibrated_start_distance_m + depth_off
            calib_retry_count = 0
            mode = Mode.IBVS
            return
        calib_retry_count += 1
        print("Z-calibration retry", calib_retry_count, "reason:", aruco_ranging.last_probe_reason)
        if calib_retry_count <= calib_max_retries:
            mode = Mode.CALIBRATION_MARKER
            return
        z_default = aruco_ranging.known_start_distance_m + depth_off
        mode = Mode.IBVS

    def _optional_gripper_close() -> None:
        if hasattr(robot, "set_gripper_closed"):
            try:
                robot.set_gripper_closed()
            except Exception:
                pass

    try:
        while True:
            img, depth_m = cam.read()
            if img is None:
                continue
            robot.update_state()
            aruco = aruco_ranging.detect(img)

            if mode == Mode.CALIBRATION_MARKER:
                robot.stop()
                if aruco.ok and aruco.distance_m is not None:
                    aruco_ranging.begin_probe(aruco)
                    calib_remaining = aruco_ranging.probe_descent_m
                    z_default = aruco_ranging.known_start_distance_m + depth_off
                    if calib_remaining <= 1e-6 or probe_speed <= 1e-6:
                        _finalize_calibration()
                    else:
                        mode = Mode.CALIBRATION_DESCEND
                time.sleep(dt)
                continue

            if mode == Mode.CALIBRATION_DESCEND:
                v_cmd = np.zeros(6, dtype=float)
                axis_base = _probe_axis_in_base(robot, axis_tcp)
                v_cmd[:3] = descent_sign * probe_speed * axis_base
                robot.speedL(v_cmd, acceleration=0.25)
                if aruco.ok and aruco.distance_m is not None:
                    aruco_ranging.capture_probe_bottom(aruco)
                calib_remaining -= probe_speed * dt
                if calib_remaining <= 0.0:
                    robot.stop()
                    mode = Mode.CALIBRATION_ASCEND
                    calib_remaining = aruco_ranging.probe_descent_m
                time.sleep(dt)
                continue

            if mode == Mode.CALIBRATION_ASCEND:
                v_cmd = np.zeros(6, dtype=float)
                axis_base = _probe_axis_in_base(robot, axis_tcp)
                v_cmd[:3] = -descent_sign * probe_speed * axis_base
                robot.speedL(v_cmd, acceleration=0.25)
                calib_remaining -= probe_speed * dt
                if calib_remaining <= 0.0:
                    robot.stop()
                    _finalize_calibration()
                time.sleep(dt)
                continue

            if mode == Mode.DESCENT_TO_GRASP:
                v_cmd = np.zeros(6, dtype=float)
                v_cmd[2] = descent_sign * descent_speed
                robot.speedL(v_cmd, acceleration=0.25)
                grasp_remaining -= abs(v_cmd[2]) * dt
                if grasp_remaining <= 0.0:
                    robot.stop()
                    _optional_gripper_close()
                    mode = Mode.LIFT_AFTER_GRASP
                    grasp_remaining = lift_m
                time.sleep(dt)
                continue

            if mode == Mode.LIFT_AFTER_GRASP:
                v_cmd = np.zeros(6, dtype=float)
                v_cmd[2] = -descent_sign * descent_speed
                robot.speedL(v_cmd, acceleration=0.25)
                grasp_remaining -= abs(v_cmd[2]) * dt
                if grasp_remaining <= 0.0:
                    robot.stop()
                    mode = Mode.IBVS
                    align_cnt = 0
                time.sleep(dt)
                continue

            seg = segmenter.detect(img)
            if seg.ok and seg.corners is not None:
                z_meas = None
                if use_sensor_depth and depth_m is not None:
                    z_meas = depth_provider.Z_for_ibvs(seg.corners, depth_m)
                z_use = float(z_meas) * aruco_ranging.depth_scale + depth_off if z_meas is not None else z_default
                Z = np.full(4, z_use, dtype=float)
                v, e, _ = ibvs.step(seg.corners, Z)
                n = np.linalg.norm(v)
                if n > vmax and n > 1e-9:
                    v = v * (vmax / n)
                robot.speedL(v, acceleration=0.25)
                if ibvs.is_converged(e):
                    align_cnt += 1
                else:
                    align_cnt = 0
                if align_cnt >= align_hold:
                    robot.stop()
                    mode = Mode.DESCENT_TO_GRASP
                    grasp_remaining = descent_m
                    align_cnt = 0
            else:
                align_cnt = 0
                robot.stop()
            time.sleep(dt)
    except KeyboardInterrupt:
        robot.stop()
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
