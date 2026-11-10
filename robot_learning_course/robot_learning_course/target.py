"""
Target: generates the DESIRED trajectory of the flange, in task space (x, y, z), in the base frame of env0.

Shapes:
  point    a fixed point = center
  circle   a vertical circle in the y-z plane, in front of the robot
  eight    a figure "8" (Lissajous curve) in the y-z plane

Used by:  run_control
"""
import math

import numpy as np


class Target:
    def __init__(self, shape="circle", center=(0.45, 0.0, 0.45), radius=0.12, period=8.0):
        self.shape = shape
        self.center = center
        self.radius = radius
        self.period = period
        print(f"Target: {shape}, center {list(center)}, radius {radius} m")

    def position(self, t):
        """Desired position at time t [s]."""
        cx, cy, cz = self.center
        w = 2 * math.pi / self.period          # angular frequency [rad/s]
        r = self.radius
        if self.shape == "circle":
            return np.array([cx, cy + r * math.sin(w * t), cz + r * math.cos(w * t)])
        if self.shape == "eight":
            return np.array([cx, cy + r * math.sin(w * t), cz + 0.5 * r * math.sin(2 * w * t)])
        return np.array([cx, cy, cz])          # "point"
