import os
import time

import mujoco
import mujoco.viewer

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'universal_robots_ur5e')

# Change to model directory so MuJoCo resolves relative paths correctly
os.chdir(model_dir)

# Use absolute path to ensure correct path resolution
model = mujoco.MjModel.from_xml_path(os.path.join(model_dir, "IBVS_Scene.xml"))
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:

  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(model, data)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)