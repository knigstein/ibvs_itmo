"""Однопоточная обёртка MuJoCo: камера, IBVS-скорость, PD по суставам, weld захвата."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

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

        _eq_typ = getattr(mujoco.mjtObj, "mjOBJ_EQUALITY", None)
        if _eq_typ is None:
            _eq_typ = getattr(mujoco.mjtObj, "mjOBJ_EQ", None)
        self.grasp_eq_id = (
            mujoco.mj_name2id(self.model, _eq_typ, "grasp_weld") if _eq_typ is not None else -1
        )
        self.work_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "work_cube_geom")

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

    def mj_forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def render_camera_bgr(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="real_sense")
        rgb = self.renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def physics_step_ibvs(self, v_cam: np.ndarray) -> None:
        v_cam = np.asarray(v_cam, dtype=float).reshape(6)
        self.controller.run_vel_camera_ibvs(v_cam, self.cam_site_id)
        mujoco.mj_step(self.model, self.data)
        self._update_telemetry()

    def physics_step_joint(self, q_des: np.ndarray) -> None:
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
        """Нулевое управление (только интегратор)."""
        self.data.qfrc_applied[:6] = 0.0
        mujoco.mj_step(self.model, self.data)
        self._update_telemetry()

    def _update_telemetry(self) -> None:
        if self.work_geom_id >= 0:
            gp = self.data.geom_xpos[self.work_geom_id]
            ep = self.data.site_xpos[self.eef_site_id]
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
