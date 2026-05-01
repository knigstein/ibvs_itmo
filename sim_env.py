"""Однопоточная обёртка MuJoCo: камера, IBVS-скорость, PD по суставам, weld захвата."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mujoco
import numpy as np

from controllers.operational_space_controller import OperationalSpaceController


class MuJoCoArmSim:
    def __init__(
        self,
        model_path: str = "universal_robots_ur5e/IBVS_Scene.xml",
        joint_pd_kp: float = 280.0,
        joint_pd_kd: float = 85.0,
        robot_cfg: Optional[Dict[str, Any]] = None,
        camera_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        cfg = robot_cfg or {}
        cam = camera_cfg or {}
        self._render_camera = str(cam.get("mujoco_camera") or "real_sense")

        self.joint_names: List[str] = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._jnt_dof_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names
        ]
        self.eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        self.cam_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "real_sense_site")

        grasp_site_name = str(cfg.get("grasp_site_for_distance", "gripper_2f85_pinch"))
        gs_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, grasp_site_name)
        self.grasp_site_id = gs_id if gs_id >= 0 else self.eef_site_id

        _eq_typ = getattr(mujoco.mjtObj, "mjOBJ_EQUALITY", None)
        if _eq_typ is None:
            _eq_typ = getattr(mujoco.mjtObj, "mjOBJ_EQ", None)
        self.grasp_eq_id = (
            mujoco.mj_name2id(self.model, _eq_typ, "grasp_weld") if _eq_typ is not None else -1
        )
        geom_candidates = cfg.get("target_geom_for_distance")
        if geom_candidates is None:
            geom_candidates = ["small_cube_geom", "work_cube_geom"]
        elif isinstance(geom_candidates, str):
            geom_candidates = [geom_candidates]
        self.work_geom_id = -1
        for gname in geom_candidates:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, str(gname))
            if gid >= 0:
                self.work_geom_id = gid
                break

        gcfg = cfg.get("gripper") or {}
        g_act_name = str(gcfg.get("actuator_name", "gripper_2f85_fingers_actuator"))
        self._gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, g_act_name)
        self._gripper_open = float(gcfg.get("open_ctrl", 0.0))
        self._gripper_closed = float(gcfg.get("closed_ctrl", 255.0))
        self._gripper_cmd = self._gripper_open

        self.controller = OperationalSpaceController(
            self.data,
            self.model,
            self.joint_names,
            eef_site="eef_site",
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        self._joint_pd_kp = joint_pd_kp
        self._joint_pd_kd = joint_pd_kd

        q0 = np.array([np.pi / 2, -np.pi / 2, np.pi / 2 - np.pi / 6, -np.pi / 2 + np.pi / 6, -np.pi / 2, 0.0])
        self.data.qpos[self._jnt_dof_ids] = q0
        mujoco.mj_forward(self.model, self.data)
        if self.grasp_eq_id >= 0:
            self.data.eq_active[self.grasp_eq_id] = 0

        self._last_eef_cube_dist: float = 0.0
        self._apply_gripper_ctrl()

    def _apply_gripper_ctrl(self) -> None:
        if self._gripper_act_id < 0 or self.model.nu <= self._gripper_act_id:
            return
        self.data.ctrl[self._gripper_act_id] = self._gripper_cmd

    def set_gripper_open(self) -> None:
        self._gripper_cmd = self._gripper_open
        self._apply_gripper_ctrl()

    def set_gripper_closed(self) -> None:
        self._gripper_cmd = self._gripper_closed
        self._apply_gripper_ctrl()

    def sync_gripper_with_phase(self, phase: Any) -> None:
        from task_fsm import Phase as P

        if phase in (P.IBVS_APPROACH, P.FINAL_ALIGN, P.SEARCH, P.GRASP_CLOSE):
            self.set_gripper_open()
        elif phase == P.TRANSPORT:
            self.set_gripper_closed()
        elif phase in (P.IDLE, P.DONE, P.RELEASE):
            self.set_gripper_open()

    def mj_forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def camera_T_w_c(self) -> np.ndarray:
        """
        Поза единственной камеры на роботе в мире: p_w = R @ p_c + t (MuJoCo site frame).
        """
        R = self.data.site_xmat[self.cam_site_id].reshape(3, 3).copy()
        t = self.data.site_xpos[self.cam_site_id].copy()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def render_camera_bgr(self, camera: Optional[str] = None) -> np.ndarray:
        name = camera if camera is not None else self._render_camera
        self.renderer.update_scene(self.data, camera=name)
        rgb = self.renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def physics_step_ibvs(self, v_cam: np.ndarray) -> None:
        self._apply_gripper_ctrl()
        v_cam = np.asarray(v_cam, dtype=float).reshape(6)
        self.controller.run_vel_camera_ibvs(v_cam, self.cam_site_id)
        mujoco.mj_step(self.model, self.data)
        self._update_telemetry()

    def physics_step_joint(self, q_des: np.ndarray) -> None:
        self._apply_gripper_ctrl()
        q_des = np.asarray(q_des, dtype=float).reshape(6)
        q = self.data.qpos[self._jnt_dof_ids].copy()
        dq = self.data.qvel[self._jnt_dof_ids].copy()
        tau = self._joint_pd_kp * (q_des - q) - self._joint_pd_kd * dq
        tau += self.data.qfrc_bias[self._jnt_dof_ids]
        tau = np.clip(tau, -150.0, 150.0)
        self.data.qfrc_applied[:6] = tau
        mujoco.mj_step(self.model, self.data)
        self._update_telemetry()

    def physics_step_hold(self) -> None:
        self._apply_gripper_ctrl()
        self.data.qfrc_applied[:6] = 0.0
        mujoco.mj_step(self.model, self.data)
        self._update_telemetry()

    def _update_telemetry(self) -> None:
        if self.work_geom_id >= 0:
            gp = self.data.geom_xpos[self.work_geom_id]
            ep = self.data.site_xpos[self.grasp_site_id]
            self._last_eef_cube_dist = float(np.linalg.norm(ep - gp))
        else:
            self._last_eef_cube_dist = float("nan")

    @property
    def eef_cube_distance(self) -> float:
        return self._last_eef_cube_dist

    def set_grasp_weld(self, active: bool) -> None:
        if self.grasp_eq_id >= 0:
            self.data.eq_active[self.grasp_eq_id] = int(active)

    def get_q(self) -> np.ndarray:
        return self.data.qpos[self._jnt_dof_ids].copy()


def load_robot_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or Path(__file__).resolve().parent / "config" / "robot.json"
    with open(p, "r", encoding="utf8") as f:
        return json.load(f)


def load_camera_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or Path(__file__).resolve().parent / "config" / "camera.json"
    with open(p, "r", encoding="utf8") as f:
        return json.load(f)
