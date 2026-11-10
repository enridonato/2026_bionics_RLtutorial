"""
Collector: records the dataset for supervised learning.

For forward kinematics we need pairs   (input, label) = (q, x)
  q = the 7 joint angles     (from the simulator's joint sensors)
  x = the flange position    (from the simulator)

When `num_samples` pairs have been collected the dataset is saved as a .npz file
(a zipped bundle of numpy arrays).

Used by:  run_collect
"""
import os

import numpy as np

from robot_learning_course.paths import DATA_DIR


class Collector:
    def __init__(self, num_samples=50000, output=os.path.join(DATA_DIR, "fk_dataset.npz")):
        self.num_samples = num_samples
        self.output = output
        # The dataset grows here, one row per sample
        self.q, self.x, self.env, self.t = [], [], [], []
        self.done = False
        print(f"Collecting {num_samples} samples...")

    def record(self, q, x, k, t):
        """Store a pair (joint angles, flange position) of robot k, measured at time t."""
        if self.done:
            return
        self.q.append(q)
        self.x.append(x)
        self.env.append(k)
        self.t.append(t)
        n = len(self.q)
        if n % 5000 == 0:
            print(f"  {n} / {self.num_samples} samples")
        if n >= self.num_samples:
            self.save()

    def save(self):
        self.done = True
        os.makedirs(os.path.dirname(self.output), exist_ok=True)
        np.savez_compressed(self.output,
                            q=np.array(self.q),         # (N, 7) inputs  [rad]
                            x=np.array(self.x),         # (N, 3) labels  [m]
                            env=np.array(self.env),     # (N,)   which robot produced the sample
                            t=np.array(self.t))         # (N,)   time stamp [s]
        print(f"Saved {len(self.q)} samples to {self.output}")
