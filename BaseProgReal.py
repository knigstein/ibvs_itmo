import numpy as np
np.set_printoptions(suppress=True)

import cv2
import pyrealsense2 as rs
from pathlib import Path

from datetime import datetime
import time
import json

# Import local libraries
# from ibvs import IBVS
from RobotModel import RobotModel
from UniversalRobotAPI import UniversalRobotAPI

CAMERA = "realsense"
# CAMERA = "web"


TIME_STEP = 1/30

with open("config.yaml", 'r+') as config:
    config_info = json.load(config)

# 1. Load the predefined ArUco dictionary
# Example: DICT_6X6_250 dictionary, which has 250 markers of 6x6 bits each
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# 2. Define the ArUco detector parameters
# Default parameters are typically sufficient
aruco_params = cv2.aruco.DetectorParameters()

# 3. Create the ArucoDetector object
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

print("CAMERA =", CAMERA)
cap = None
if CAMERA == "web":
    cap = cv2.VideoCapture(0) # 0 for default camera

pipeline = None
if CAMERA == "realsense":
    # Configure color stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # Using BGR8 format for direct compatibility with OpenCV
    # Start streaming
    pipeline.start(config)

# Configure robot model
model_dir = Path(__file__).parent / "models"
urdf_filename = model_dir / "ur5e_robot.urdf"
# model = RobotModel(urdf_filename, "tool0")
model = RobotModel(urdf_filename, "camera_frame")
robot = UniversalRobotAPI("192.168.90.107")

cc = np.array([640//2, 480//2])

def main():

    initial_q = [0, -np.pi/2, np.pi/2 - np.pi/6, -np.pi/2 + np.pi/6, -np.pi/2, 0]
    robot.moveJ(initial_q)

    try:
        
        ''' Video servoing '''
        while True:


            color_image = None
            if CAMERA == "web":
                ret, color_image = cap.read() # Read frame

            if CAMERA == "realsense":
                # Wait for a new set of frames
                frames = pipeline.wait_for_frames()
                
                # Get the color frame
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue

                # Convert the frame to a NumPy array
                color_image = np.asanyarray(color_frame.get_data())

            if color_image is None:
                continue

            # cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            robot.update_state()

            time.sleep(TIME_STEP)

    except KeyboardInterrupt as ex:
        robot.stop()

if __name__ == '__main__':
    main()
