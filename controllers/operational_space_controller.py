from controllers import JointEffortController

import numpy as np
import mujoco

from .controller_utils import (
    task_space_inertia_matrix,
    pose_error,
)

from .mujoco_utils import (
    get_site_jac,
    get_fullM,
)
from .camera_kinematics import twist_camera_to_world

from .transform_utils import (
    mat2quat,
)

class OperationalSpaceController(JointEffortController):
    def __init__(
        self,
        data,
        model,
        joints,
        eef_site,
        min_effort: np.ndarray,
        max_effort: np.ndarray,
        kp: float,
        ko: float,
        kv: float,
        vmax_xyz: float,
        vmax_abg: float,
    ) -> None:
        
        super().__init__(data, model, joints, min_effort, max_effort)

        self._eef_site = eef_site
        self._kp = kp
        self._ko = ko
        self._kv = kv
        self._vmax_xyz = vmax_xyz
        self._vmax_abg = vmax_abg

        self._eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, eef_site)
        self._jnt_dof_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                          for name in joints]
        self._dof = len(self._jnt_dof_ids)

        self._task_space_gains = np.array([self._kp] * 3 + [self._ko] * 3)
        self._lamb = self._task_space_gains / self._kv
        self._sat_gain_xyz = vmax_xyz / self._kp * self._kv
        self._sat_gain_abg = vmax_abg / self._ko * self._kv
        self._scale_xyz = vmax_xyz / self._kp * self._kv
        self._scale_abg = vmax_abg / self._ko * self._kv

    def run(self, target):
        # target pose is a 7D vector [x, y, z, qx, qy, qz, qw]
        target_pose = target

        # Get the Jacobian matrix for the end-effector.
        J = get_site_jac(
            self._model, 
            self._data, 
            self._eef_id,
        )
        J = J[:, self._jnt_dof_ids]

        # Get the mass matrix and its inverse for the controlled degrees of freedom (DOF) of the robot.
        M_full = get_fullM(
            self._model, 
            self._data,
        )
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the joint velocities for the controlled DOF.
        dq = self._data.qvel[self._jnt_dof_ids].copy()

        # Get the end-effector position, orientation matrix, and twist (spatial velocity).
        ee_pos = self._data.site_xpos[self._eef_id]
        ee_quat = mat2quat(self._data.site_xmat[self._eef_id].reshape(3, 3))
        ee_pose = np.concatenate([ee_pos, ee_quat])

        # Calculate the pose error (difference between the target and current pose).
        pose_err = pose_error(target_pose, ee_pose)

        # Initialize the task space control signal (desired end-effector motion).
        u_task = np.zeros(6)

        # Calculate the task space control signal.
        u_task += self._scale_signal_vel_limited(pose_err)

        # joint space control signal
        u = np.zeros(self._dof)
        
        # Add the task space control signal to the joint space control signal
        u += np.dot(J.T, np.dot(Mx, u_task))

        # Add damping to joint space control signal
        u += -self._kv * np.dot(M, dq)

        # Add gravity compensation to the target effort
        u += self._data.qfrc_bias[self._jnt_dof_ids]

        # send the target effort to the joint effort controller
        super().run(u)

    def run_vel(self, target, rot):

        lin_vel = rot @ target[:3]
        rot_vel = rot @ target[3:]
        target_speed = np.concatenate((lin_vel, rot_vel))

        # Get the Jacobian matrix for the end-effector.
        J = get_site_jac(
            self._model, 
            self._data, 
            self._eef_id,
        )
        J = J[:, self._jnt_dof_ids]


        M_full = get_fullM(
            self._model, 
            self._data,
        )
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the joint velocities for the controlled DOF.
        dq = self._data.qvel[self._jnt_dof_ids].copy()

        dq_des = np.linalg.pinv(J) @ target_speed

        # # Initialize the task space control signal (desired end-effector motion).
        # u_task = np.zeros(6)

        # # Calculate the task space control signal.
        # u_task += self._scale_signal_vel_limited(pose_err)

        # joint space control signal
        u = np.zeros(self._dof)
        
        # Add the task space control signal to the joint space control signal
        # u += np.dot(J.T, np.dot(Mx, u_task))

        # Add P speed control
        u += self._kv*M @ (dq_des - dq)

        # Get the end-effector position, orientation matrix, and twist (spatial velocity).
        # ee_pos = self._data.site_xpos[self._eef_id]
        # ee_quat = mat2quat(self._data.site_xmat[self._eef_id].reshape(3, 3))
        # ee_pose = np.concatenate([ee_pos, ee_quat])

        # # Calculate the pose error (difference between the target and current pose).
        # pose_err = pose_error(target_pose, ee_pose)

        # Initialize the task space control signal (desired end-effector motion).
        # u_task = np.zeros(6)

        # # Calculate the task space control signal.
        # u_task += self._scale_signal_vel_limited(pose_err)

        # # joint space control signal
        # u = np.zeros(self._dof)
        
        # Add the task space control signal to the joint space control signal
        # u += np.dot(J.T, np.dot(Mx, u_task))

        # Add damping to joint space control signal
        # u += -self._kv * np.dot(M, dq)

        # Add gravity compensation to the target effort
        u += self._data.qfrc_bias[self._jnt_dof_ids]

        # send the target effort to the joint effort controller
        super().run(u)

    def run_vel_world(self, twist_world: np.ndarray, site_id: int) -> None:
        """Желаемый винт в мире в точке site_id; mj_jacSite согласован с twist_world."""
        target_speed = np.asarray(twist_world, dtype=float).reshape(6)
        J = get_site_jac(self._model, self._data, site_id)
        J = J[:, self._jnt_dof_ids]
        M_full = get_fullM(self._model, self._data)
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        dq = self._data.qvel[self._jnt_dof_ids].copy()
        dq_des = np.linalg.pinv(J) @ target_speed
        u = self._kv * M @ (dq_des - dq)
        u += self._data.qfrc_bias[self._jnt_dof_ids]
        super().run(u)

    def run_vel_camera_ibvs(self, v_cam: np.ndarray, camera_site_id: int) -> None:
        R = self._data.site_xmat[camera_site_id].reshape(3, 3)
        twist_world = twist_camera_to_world(v_cam, R)
        self.run_vel_world(twist_world, camera_site_id)

    def _scale_signal_vel_limited(self, u_task: np.ndarray) -> np.ndarray:
        """
        Scale the control signal such that the arm isn't driven to move faster in position or orientation than the specified vmax values.

        Parameters:
            u_task (numpy.ndarray): The task space control signal.

        Returns:
            numpy.ndarray: The scaled task space control signal.
        """
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        scale = np.ones(6)
        if norm_xyz > self._sat_gain_xyz:
            scale[:3] *= self._scale_xyz / norm_xyz
        if norm_abg > self._sat_gain_abg:
            scale[3:] *= self._scale_abg / norm_abg

        return self._kv * scale * self._lamb * u_task

