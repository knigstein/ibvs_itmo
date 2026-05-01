"""
Однопоточный цикл: viewer → кадр real_sense → сегментация → IBVS → FSM.

Подстройте эталонные признаки в config/camera.json под вид куба в кадре.
"""
from __future__ import annotations

import os
import time

import mujoco
import mujoco.viewer

from ibvs import IBVS
from sim_env import MuJoCoArmSim, load_camera_config, load_robot_config
from task_fsm import Phase, PickPlaceFSM
# from vision import CubeSegmenter
# from vision.yolo_detector import YOLOFeatureDetector as CubeSegmenter
from vision import create_detector

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'universal_robots_ur5e')

# Change to model directory so MuJoCo resolves relative paths correctly
os.chdir(model_dir)


def main() -> None:
    robot_cfg = load_robot_config()
    cam_cfg = load_camera_config()
    sim = MuJoCoArmSim(model_path=os.path.join(model_dir, "IBVS_Scene.xml"))
    ibvs = IBVS(cam_cfg)
    
    detector_type = robot_cfg.get("vision", {}).get("detector_type", "universal")
    segmenter = create_detector(detector_type, robot_cfg.get("vision", {}))
    fsm = PickPlaceFSM(ibvs, segmenter, robot_cfg, on_phase=lambda p: print("FSM:", p.name))
    fsm.start()

    dt = 0.01
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            img = sim.render_camera_bgr()
            v = fsm.step(sim, img)
            print(fsm.phase)
            if fsm.phase == Phase.RELEASE:
                fsm.finish_release(sim)
                sim.physics_step_hold()
            elif fsm.phase == Phase.TRANSPORT:
                q_tgt = fsm.joint_target_for_transport(sim, dt)
                if q_tgt is not None:
                    sim.physics_step_joint(q_tgt)
            elif fsm.phase == Phase.GRASP_CLOSE:
                sim.physics_step_ibvs(v)
            elif fsm.phase in (Phase.IBVS_APPROACH, Phase.FINAL_ALIGN):
                sim.physics_step_ibvs(v)
            else:
                sim.physics_step_hold()

            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
