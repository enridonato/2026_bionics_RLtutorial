"""
Build a MuJoCo scene with N copies of the FR3 robot ("multi-environment").

    world.xml  (floor, light, target)
       + env0/ FR3 at (0, 0, 0)
       + env1/ FR3 at (spacing, 0, 0)
       + ...

Every copy gets a name prefix ("env0/", "env1/", ...) so that joint names stay unique:
"env0/fr3_joint1", "env1/fr3_joint1", ...
"""
import math
import os

import mujoco
import numpy as np

from robot_learning_course import fr3
from robot_learning_course.paths import package_dir


def base_positions(num_envs, spacing):
    """Place the robots on a square grid, env0 at the origin. Returns an (N, 3) array."""
    cols = math.ceil(math.sqrt(num_envs))
    return np.array([[spacing * (i % cols), spacing * (i // cols), 0.0] for i in range(num_envs)])


def build_scene(num_envs=1, spacing=1.5):
    """Return (model, data, bases): the compiled MuJoCo model, its state, and the robot base positions."""
    world = mujoco.MjSpec.from_file(os.path.join(package_dir("mujoco_models"), "world.xml"))
    bases = base_positions(num_envs, spacing)

    for i in range(num_envs):
        robot = mujoco.MjSpec.from_file(os.path.join(package_dir("mujoco_models"), "fr3", "fr3.xml"))
        frame = world.worldbody.add_frame(pos=bases[i])         # where this robot stands
        world.attach(robot, prefix=f"env{i}/", frame=frame)     # copy the robot into the world

    model = world.compile()
    data = mujoco.MjData(model)

    # Start every robot in the home configuration (both the state and the motor commands).
    for i in range(num_envs):
        qpos_ids, _, ctrl_ids = joint_indices(model, i)
        data.qpos[qpos_ids] = fr3.HOME
        data.ctrl[ctrl_ids] = fr3.HOME
    mujoco.mj_forward(model, data)  # compute positions of all bodies for this configuration
    return model, data, bases


def joint_indices(model, env):
    """Where the 7 joints of robot `env` live inside data.qpos/qvel and data.ctrl."""
    qpos_ids = [model.joint(f"env{env}/{name}").qposadr[0] for name in fr3.JOINT_NAMES]
    qvel_ids = [model.joint(f"env{env}/{name}").dofadr[0] for name in fr3.JOINT_NAMES]
    ctrl_ids = [model.actuator(f"env{env}/{name}").id for name in fr3.JOINT_NAMES]
    return np.array(qpos_ids), np.array(qvel_ids), np.array(ctrl_ids)
