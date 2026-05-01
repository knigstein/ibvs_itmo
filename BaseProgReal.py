"""
Реальный UR + RealSense: кадр → сегментация → IBVS → speedL (инструмент/TCP).

Полный FSM как в симуляции требует оценки расстояния до объекта (датчик, триангуляция
или ручной переход фаз). Здесь — непрерывное визуальное серво по кубу.
"""
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
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        if rs is None:
            raise RuntimeError("Установите пакет pyrealsense2")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def stop(self):
        self.pipeline.stop()


def main():
    robot_cfg, cam_cfg = load_configs()
    if UrRtdeRobot is None:
        raise RuntimeError("Нужен модуль ur_rtde и UniversalRobotAPI.py")
    if rs is None:
        raise RuntimeError("Нужен pyrealsense2")

    ip = robot_cfg.get("ur_ip", "192.168.90.107")
    robot = UrRtdeRobot(ip)
    cam = RealSenseGrabber()
    ibvs = IBVS(cam_cfg)
    segmenter = CubeSegmenter(robot_cfg.get("vision", {}))
    z = float(robot_cfg.get("default_Z", 0.5))
    vmax = float(robot_cfg.get("max_ibvs_speed", 0.25))
    dt = float(robot_cfg.get("control_dt", 1.0 / 30.0))

    try:
        while True:
            img = cam.read()
            if img is None:
                continue
            robot.update_state()
            seg = segmenter.detect(img)
            if seg.ok and seg.corners is not None:
                v, _, _ = ibvs.step(seg.corners, z)
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
