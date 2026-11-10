"""
Simulator: the MuJoCo simulator wrapped in a small Python class.
It plays the role of the REAL ROBOT(S): it does not know anything about learning.

For every robot k (k = 0 .. num_envs-1):
  sensors   joint_angles(k)      the 7 joint angles q [rad]
            flange_position(k)   the flange position x [m], in the robot BASE frame
  motors    command(k, q_des)    7 desired joint angles [rad]
And once:
  set_target(position)           only to draw the green sphere

Used by:  run_sim, run_collect, run_control
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

from robot_learning_course import fr3
from robot_learning_course.scene import build_scene, joint_indices


class Simulator:
    def __init__(self, num_envs=1, spacing=1.5, rate=100.0, headless=False):
        # --- 1. Build the MuJoCo world with N robots
        self.num_envs = num_envs
        self.model, self.data, self.bases = build_scene(num_envs, spacing)
        self.ids = [joint_indices(self.model, k) for k in range(num_envs)]
        self.ee_site = [self.model.site(f"env{k}/{fr3.EE_SITE}").id for k in range(num_envs)]
        # How many physics steps (0.002 s each) in one cycle (0.01 s at 100 Hz) -> 5
        self.steps_per_cycle = max(1, round(1.0 / (rate * self.model.opt.timestep)))
        self.cycle = 1.0 / rate
        self.time = 0.0                                # simulated time [s]

        # --- 2. The 3D window. With a window we run in real time, without it as fast as possible
        self.viewer = None if headless else mujoco.viewer.launch_passive(self.model, self.data)
        self.start = time.time()
        print(f"Simulating {num_envs} FR3 robot(s) at {rate} Hz")

    # ------------------------------------------------------------------ motors
    def command(self, k, q_des):
        """A new joint command for robot k: send it to its 7 position-controlled motors."""
        self.data.ctrl[self.ids[k][2]] = fr3.clip_to_limits(q_des)

    def set_target(self, position):
        """Move the green sphere. Targets are given in the base frame of env0 (= world origin)."""
        self.data.mocap_pos[0] = self.bases[0] + position

    # ------------------------------------------------------------------ sensors
    def joint_angles(self, k):
        return self.data.qpos[self.ids[k][0]].copy()

    def flange_position(self, k):
        """Flange position, expressed in the frame of robot k's base (robot bases are not rotated)."""
        return self.data.site_xpos[self.ee_site[k]] - self.bases[k]

    # ------------------------------------------------------------------ main loop
    def step(self):
        """One cycle of the main loop (10 ms): advance the physics and redraw."""
        for _ in range(self.steps_per_cycle):          # 1. advance the physics: 5 steps of 2 ms
            mujoco.mj_step(self.model, self.data)
        # mj_step updates q, but the body positions it stores refer to the q BEFORE the step:
        # recompute them, otherwise q and x would be 2 ms apart (a few mm of label error!)
        mujoco.mj_kinematics(self.model, self.data)
        self.time += self.cycle
        if self.viewer is not None:
            self.viewer.sync()                         # 2. redraw the 3D window
            time.sleep(max(0.0, self.start + self.time - time.time()))  # 3. wait: 1 s simulated = 1 s real

    def window_open(self):
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
