"""
Manual teleop for MuJoCo UR5e scene.

Designed to stay close to BaseProgSim setup:
- same model path (IBVS_Scene.xml)
- same MuJoCoArmSim wrapper and controller pipeline
- passive viewer loop + fixed dt
"""
from __future__ import annotations

import os
import time

import cv2
import mujoco.viewer
import numpy as np

from sim_env import MuJoCoArmSim, load_robot_config


def print_controls() -> None:
    print("\n=== Manual Teleop Controls ===")
    print("Translational velocity (camera frame):")
    print("  w/s: +x / -x")
    print("  a/d: +y / -y")
    print("  r/f: +z / -z")
    print("Rotational velocity (camera frame):")
    print("  i/k: +wx / -wx")
    print("  j/l: +wy / -wy")
    print("  u/o: +wz / -wz")
    print("Gripper:")
    print("  g: close, h: open")
    print("Speed:")
    print("  1: slower, 2: faster")
    print("Other:")
    print("  space: zero twist")
    print("  q or Esc: quit")
    print("=============================\n")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "universal_robots_ur5e")
    os.chdir(model_dir)

    robot_cfg = load_robot_config()
    sim = MuJoCoArmSim(model_path=os.path.join(model_dir, "IBVS_Scene.xml"), robot_cfg=robot_cfg)

    dt = float(robot_cfg.get("control_dt", 0.01))
    sync_realtime = bool(robot_cfg.get("sync_realtime", False))
    physics_steps = sim._physics_steps(dt)
    v_lin = 0.08
    v_ang = 0.35
    lin_scale = 1.0
    ang_scale = 1.0
    v_cmd = np.zeros(6, dtype=float)

    print_controls()

    cv2.namedWindow("teleop_camera", cv2.WINDOW_NORMAL)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            frame_start = time.time()
            img = sim.render_camera_bgr()
            cv2.putText(
                img,
                "WASD/RF xyz, IJKLUO rot, G/H gripper, 1/2 speed, Q quit",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("teleop_camera", img)

            key = cv2.waitKey(1) & 0xFF

            # Default: zero command every frame unless key is held/repeated.
            v_cmd[:] = 0.0

            if key == ord("w"):
                v_cmd[0] = +v_lin * lin_scale
            elif key == ord("s"):
                v_cmd[0] = -v_lin * lin_scale
            elif key == ord("a"):
                v_cmd[1] = +v_lin * lin_scale
            elif key == ord("d"):
                v_cmd[1] = -v_lin * lin_scale
            elif key == ord("r"):
                v_cmd[2] = +v_lin * lin_scale
            elif key == ord("f"):
                v_cmd[2] = -v_lin * lin_scale
            elif key == ord("i"):
                v_cmd[3] = +v_ang * ang_scale
            elif key == ord("k"):
                v_cmd[3] = -v_ang * ang_scale
            elif key == ord("j"):
                v_cmd[4] = +v_ang * ang_scale
            elif key == ord("l"):
                v_cmd[4] = -v_ang * ang_scale
            elif key == ord("u"):
                v_cmd[5] = +v_ang * ang_scale
            elif key == ord("o"):
                v_cmd[5] = -v_ang * ang_scale
            elif key == ord("g"):
                sim.set_gripper_closed()
            elif key == ord("h"):
                sim.set_gripper_open()
            elif key == ord("1"):
                lin_scale = max(0.2, lin_scale * 0.8)
                ang_scale = max(0.2, ang_scale * 0.8)
                print(f"Speed scale: lin={lin_scale:.2f}, ang={ang_scale:.2f}")
            elif key == ord("2"):
                lin_scale = min(3.0, lin_scale * 1.25)
                ang_scale = min(3.0, ang_scale * 1.25)
                print(f"Speed scale: lin={lin_scale:.2f}, ang={ang_scale:.2f}")
            elif key == ord(" "):
                v_cmd[:] = 0.0
            elif key == ord("q") or key == 27:
                break

            sim.physics_step_ibvs(v_cmd, physics_steps)
            viewer.sync()
            if sync_realtime:
                elapsed = time.time() - frame_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
