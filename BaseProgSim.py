import time
import numpy as np
import cv2
import json

from SimRobotApi import UniversalRobotAPI
# from ibvs import IBVS

TIME_STEP = 1/30

def main():


    with open("config.yaml", 'r+') as config:
        config_info = json.load(config)

    is_work = [False]
    robot = UniversalRobotAPI(is_work)
    robot.start()


    try:

        pass

    except KeyboardInterrupt as ex:
        robot.stop()


    print("Main thread: Signaling worker to stop.")
    is_work[0] = True
    robot.join()
    time.sleep(0.5)
    print("Main thread: Worker stopped. Exiting program.")

if __name__ == '__main__':
    main()