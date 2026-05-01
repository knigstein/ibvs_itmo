"""
Однопоточный цикл: MuJoCo viewer → кадр с камеры → сегментация → IBVS → FSM (подъезд / захват / перенос).

Перед запуском подстройте эталонные признаки в config/camera.json под вид куба в кадре.
"""
from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from ibvs import IBVS
from sim_env import MuJoCoArmSim, load_camera_config, load_robot_config
from task_fsm import Phase, PickPlaceFSM
from vision import CubeSegmenter


def main() -> None:
    robot_cfg = load_robot_config()
    cam_cfg = load_camera_config()
    sim = MuJoCoArmSim()
    ibvs = IBVS(cam_cfg)
    segmenter = CubeSegmenter(robot_cfg.get("vision", {}))
    fsm = PickPlaceFSM(ibvs, segmenter, robot_cfg, on_phase=lambda p: print("FSM:", p.name))
    fsm.start()

    dt = 0.01
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            img = sim.render_camera_bgr()
            v = fsm.step(sim, img)

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
