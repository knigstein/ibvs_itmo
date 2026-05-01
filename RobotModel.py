from pathlib import Path
from sys import argv
import numpy as np
np.set_printoptions(suppress=True)
 
import pinocchio


class RobotModel:
    def __init__(self, urdf_path, ee_frame):
        # Load the urdf model
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        print("model name: " + self.model.name)

        # Create data required by the algorithms
        self.data = self.model.createData()

        self.ee_frame = ee_frame

    def forward_kinematics(self, q):
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        frame_id = self.model.getFrameId(self.ee_frame)
        return self.data.oMf[frame_id]

    def compute_jacobian(self, q):
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        frame_id = self.model.getFrameId(self.ee_frame)
        return pinocchio.computeFrameJacobian(self.model, self.data, q, frame_id)