import numpy as np

from RobotState import RobotState

from controllers.operational_space_controller import OperationalSpaceController

import mujoco
import mujoco.viewer

import threading
import time
import cv2


class UniversalRobotAPI(threading.Thread):
    DT = 0.002

    def __init__(self, controls: list, model_path: str = "universal_robots_ur5e/IBVS_Scene.xml"):

        super().__init__() # Required to properly initialize the Thread parent class

        # Load model
        self.__model = mujoco.MjModel.from_xml_path(model_path)
        self.__data = mujoco.MjData(self.__model)
        self.__renderer = mujoco.Renderer(self.__model, height=480, width=640)
        self.__renderer.update_scene(self.__data, camera="real_sense")

        self.__data.qpos = np.array([np.pi/2, -np.pi/2, np.pi/2 -np.pi/6, -np.pi/2 + np.pi/6, -np.pi/2, 0])

        self.__state = RobotState()
        self.__joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        self.__controller = OperationalSpaceController(
            self.__data,
            self.__model,
            self.__joint_names,
            eef_site="eef_site",
            min_effort=-150.0,
            max_effort= 150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        self.__eef_id = mujoco.mj_name2id(self.__model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        self.__cam_site_id = mujoco.mj_name2id(
            self.__model, mujoco.mjtObj.mjOBJ_SITE, "real_sense_site"
        )
        self._jnt_dof_ids = [mujoco.mj_name2id(self.__model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.__joint_names]
        self.__desired_vel = np.array([0, 0, 0, 0, 0, 0])
        self.__stopped = controls


    def run(self):

        # ------------------------
        # Simulation loop
        # ------------------------
        with mujoco.viewer.launch_passive(self.__model, self.__data) as viewer:
            self.__stopped[0] = False

            try:

                while viewer.is_running() and not self.__stopped[0]:
                    mujoco.mj_step(self.__model, self.__data)
                    viewer.sync()
                    self.__controller.run_vel_camera_ibvs(
                        self.__desired_vel, self.__cam_site_id
                    )
                    # time.sleep(0.01)

            except KeyboardInterrupt:
                pass

    def update_state(self):
        # Code to update the robot's state, such as joint positions, end-effector position, etc.
        self.__state.q      = self.__data.qpos[self._jnt_dof_ids]
        self.__state.dq     = self.__data.qvel[self._jnt_dof_ids]
        self.__state.i      = np.zeros((6,))
        self.__state.tau    = np.zeros((6,))
        self.__state.f      = np.zeros((6,))

    def get_imgage(self):
        self.__renderer.update_scene(self.__data, camera="real_sense")
        return cv2.cvtColor(self.__renderer.render(), cv2.COLOR_RGB2BGR)

    def set_start_pose(self, pos):
        self.__data.qpos[self._jnt_dof_ids] = pos
        mujoco.mj_forward(self.__model, self.__data)

    def moveL(self, pose: np.ndarray, speed: float = 0.5, acceleration: float = 0.5):
        pass

    def moveJ(self, joint_q: np.ndarray, speed: float = 1.05, acceleration: float = 1.4):
        pass

    def speedL(self, speed: np.ndarray, acceleration: float = 0.25):
        """Винт (6,) в базисе сайта real_sense_site (IBVS)."""
        self.__desired_vel = speed

    def speedJ(self, joint_dq: np.ndarray, acceleration: float = 0.5):
        pass

    def stop(self):
        print("Stop the robot")
        self.__desired_vel = np.zeros((6,))

    @property
    def state(self) -> RobotState:
        return self.__state
