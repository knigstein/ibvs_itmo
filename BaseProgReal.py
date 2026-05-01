from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from ibvs import IBVS
from vision import CubeSegmenter
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

    try:
        while True:
            img, depth_m = cam.read()
            if img is None:
                continue
            robot.update_state()
            seg = segmenter.detect(img)
            if seg.ok and seg.corners is not None:
                z_meas = None
                if use_sensor_depth and depth_m is not None:
                    z_meas = depth_provider.Z_for_ibvs(seg.corners, depth_m)
                z_use = float(z_meas) + depth_off if z_meas is not None else z_default
                Z = np.full(4, z_use, dtype=float)
                v, _, _ = ibvs.step(seg.corners, Z)
                n = np.linalg.norm(v)
                if n > vmax and n > 1e-9:
                    v = v * (vmax / n)
                robot.speedL(v, acceleration=0.25)
            else:
                robot.stop()
            time.sleep(dt)
    except KeyboardInterrupt:
        robot.stop()
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
