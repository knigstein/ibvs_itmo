import numpy as np
import rtde_control
import rtde_receive

from RobotState import RobotState


class UniversalRobotAPI:
    DT = 0.002

    def __init__(self, ip_address):

        self.ip_address = ip_address
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)
        self.rtde_c = rtde_control.RTDEControlInterface(ip_address)
        self.__state = RobotState()

    def update_state(self):
        # Code to update the robot's state, such as joint positions, end-effector position, etc.
        self.__state.q      = np.array(self.rtde_r.getActualQ())
        self.__state.dq     = np.array(self.rtde_r.getActualQd())
        self.__state.i      = np.array(self.rtde_r.getActualRobotCurrent())
        self.__state.tau    = np.array(self.rtde_r.getActualMomentum())
        self.__state.f      = np.array(self.rtde_r.getActualTCPForce())

    def moveL(self, pose: np.ndarray, speed: float = 0.5, acceleration: float = 0.5):
        self.rtde_c.speedStop()
        self.rtde_c.moveL(pose, speed, acceleration)

    def moveJ(self, joint_q: np.ndarray, speed: float = 1.05, acceleration: float = 1.4):
        self.rtde_c.speedStop()
        self.rtde_c.moveJ(joint_q, speed, acceleration)

    def speedL(self, speed: np.ndarray, acceleration: float = 0.25):
        self.rtde_c.speedL(speed, acceleration, UniversalRobotAPI.DT)

    def speedJ(self, joint_dq: np.ndarray, acceleration: float = 0.5):
        self.rtde_c.speedJ(joint_dq, acceleration, UniversalRobotAPI.DT)

    def stop(self):
        self.rtde_c.speedJ(np.zeros(6), 0.5, UniversalRobotAPI.DT)

    @property
    def state(self) -> RobotState:
        return self.__state
