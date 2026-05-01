"""
Однопоточный цикл: viewer → один кадр с ОДНОЙ камеры на манипуляторе → сегментация → IBVS → FSM.

Имя камеры MuJoCo: config/camera.json → mujoco_camera.

Глубина в симе:
  - depth.mode: none — default_Z + ibvs_depth_offset_m;
  - depth.mode: depth_map и depth.source: sfm_two_view — синтетическая карта из двух положений
    той же камеры (движение робота + триангуляция), без второго сенсора и без буфера MuJoCo;
  - на реале с RealSense: depth.source не задаётте или sensor — см. BaseProgReal.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import mujoco
import mujoco.viewer

from ibvs import IBVS
from sim_env import MuJoCoArmSim, load_camera_config, load_robot_config
from task_fsm import Phase, PickPlaceFSM
from vision.yolo_detection import YOLOFeatureDetector as CubeSegmenter

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "universal_robots_ur5e")
os.chdir(model_dir)


def main() -> None:
    robot_cfg = load_robot_config()
    cam_cfg = load_camera_config()
    sim = MuJoCoArmSim(
        model_path=os.path.join(model_dir, "IBVS_Scene.xml"),
        robot_cfg=robot_cfg,
        camera_cfg=cam_cfg,
    )
    ibvs = IBVS(cam_cfg)
    segmenter = CubeSegmenter(robot_cfg.get("vision", {}))
    fsm = PickPlaceFSM(
        ibvs,
        segmenter,
        robot_cfg,
        on_phase=lambda p: print("FSM:", p.name),
    )
    fsm.start()

    dcfg = robot_cfg.get("depth") or {}
    _, depth_mode = build_depth_provider(robot_cfg)
    use_sfm = depth_mode == "depth_map" and str(dcfg.get("source", "sensor")).lower() == "sfm_two_view"
    sfm: Optional[OneCameraTwoPoseSfM] = None
    if use_sfm:
        K = K_from_camera_json(cam_cfg)
        sfm = OneCameraTwoPoseSfM(
            K,
            height=480,
            width=640,
            min_baseline_m=float(dcfg.get("sfm_min_baseline_m", 0.003)),
            z_min=float(dcfg.get("z_min_m", 0.12)),
            z_max=float(dcfg.get("z_max_m", 2.5)),
        )

    dt = 0.01
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            img = sim.render_camera_bgr()
            depth_m = None
            if sfm is not None:
                T_w_c = sim.camera_T_w_c()
                seg_sfm = segmenter.detect(img)
                corners = seg_sfm.corners if seg_sfm.ok else None
                depth_m = sfm.update(corners, T_w_c)
            v = fsm.step(sim, img, depth_m=depth_m)

            if fsm.phase == Phase.RELEASE:
                fsm.finish_release(sim)

            sim.sync_gripper_with_phase(fsm.phase)

            if fsm.phase == Phase.TRANSPORT:
                q_tgt = fsm.joint_target_for_transport(sim, dt)
                if q_tgt is not None:
                    sim.physics_step_joint(q_tgt)
            elif fsm.phase == Phase.SEARCH:
                q_s = fsm.joint_target_for_search(sim, dt)
                if q_s is not None:
                    sim.physics_step_joint(q_s)
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
