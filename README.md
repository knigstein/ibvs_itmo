
# IBVS Control System - ITMO University

This repository contains the implementation of an **Image-Based Visual Servoing (IBVS)** system. Developed as a team project, it provides a framework for controlling robotic manipulators using visual feedback, supporting both high-fidelity simulation and real-world hardware deployment.

## Overview

The system allows for seamless switching between a virtual environment and a physical robot:
* **Simulation**: Powered by the MuJoCo physics engine for safe algorithm testing.
* **Real Robot**: Integrated with the Universal Robots UR5e via RTDE (Real-Time Data Exchange).

## Project Structure

* `config/` — Contains environment and robot configuration files.
* `controllers/` — Implementation of control laws and feedback logic.
* `models/` — XML scene descriptions and 3D models for MuJoCo.
* `universal_robots_ur5e/` — Hardware-specific drivers and API for the UR5e.
* `vision/` — Computer vision modules for feature extraction and image processing.
* `BaseProgSim.py` — Entry point for running the simulation.
* `BaseProgReal.py` — Entry point for running on the physical robot.
* `ibvs.py` — Core mathematical implementation of the IBVS algorithm.
* `task_fsm.py` — Finite State Machine governing high-level task logic.
* `config.yaml` — Centralized configuration for network, camera, and control parameters.

## Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/knigstein/ibvs_itmo.git](https://github.com/knigstein/ibvs_itmo.git)
   cd ibvs_itmo
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before execution, modify `config.yaml` to match your setup:
* **Network**: Set the `robot_ip` to the address of your UR5e controller.
* **Camera**: Input the correct intrinsic matrix and distortion coefficients.
* **Parameters**: Adjust control gains (`lambda`) and convergence thresholds.

## Usage

### Running in Simulation
To verify algorithms in the MuJoCo environment:
```bash
python BaseProgSim.py
```
The simulation provides real-time visualization of the robot's state and camera view.

### Running on Real Robot
To deploy the controller on the physical UR5e manipulator:
```bash
python BaseProgReal.py
```
**Safety Note**: Ensure the Emergency Stop is accessible and the robot's workspace is clear before execution.

## Authors
Developed by the IBVS ITMO Team.
```
