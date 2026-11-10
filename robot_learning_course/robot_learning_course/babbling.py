"""
Babbling: "motor babbling" -- the robot moves randomly to explore its own body,
like a baby waving its arms. This is how we generate DATA without any model of the robot.

Every `period` seconds each robot gets a new random goal configuration inside the joint limits.
In between, the command slides smoothly (cosine interpolation: it starts and stops with zero speed)
from the old goal to the new one,
so the robot visits many configurations along the way, not only the goals.

Used by:  run_collect
"""
import numpy as np

from robot_learning_course import fr3


class Babbling:
    def __init__(self, num_envs=1, period=1.5, margin=0.1, seed=0):
        self.num_envs = num_envs
        self.period = period                                   # new goal every 1.5 s
        self.margin = margin                                   # stay 0.1 rad away from the limits
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / 50.0                                    # command rate: 50 Hz
        self.t = 0.0                                            # time since the last goal change
        self.q_start = np.tile(fr3.HOME, (num_envs, 1))         # one row per robot
        self.q_goal = self.random_configurations()
        print(f"Babbling with {num_envs} robot(s), new goal every {period} s")

    def random_configurations(self):
        """Uniform random sample inside the joint limits, one configuration per robot: shape (N, 7)."""
        low, high = fr3.JOINT_LOWER + self.margin, fr3.JOINT_UPPER - self.margin
        # TODO (optional exercise 1): uniform sampling of q is not uniform in the workspace!
        #   Try sampling only around HOME:  fr3.HOME + 0.5 * self.rng.standard_normal((self.num_envs, 7))
        #   (clip it with fr3.clip_to_limits). How does the learned model change? Where is it accurate?
        return self.rng.uniform(low, high, size=(self.num_envs, fr3.NUM_JOINTS))

    def step(self):
        """Advance by 1/50 s and return the new command of every robot: shape (N, 7)."""
        self.t += self.dt
        if self.t >= self.period:                  # goal reached: pick a new one
            self.t = 0.0
            self.q_start = self.q_goal
            self.q_goal = self.random_configurations()

        alpha = self.t / self.period               # 0 -> 1 during one period
        s = 0.5 - 0.5 * np.cos(np.pi * alpha)      # also 0 -> 1, but with zero speed at both ends: no jerks
        return (1 - s) * self.q_start + s * self.q_goal
