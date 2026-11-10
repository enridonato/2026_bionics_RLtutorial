"""
A "shortcut" dataset: sample random configurations q and ask MuJoCo for x = FK(q) directly.

No physics, no motion: 50 000 samples in a few seconds.
Compare it with the dataset collected by moving the (simulated) robots:
  - which one would you be able to build with a REAL robot?
  - which one is cleaner? (no gravity sag, no collisions, perfectly i.i.d. samples)

Run:  make_dataset_fast --n 50000
"""
import argparse
import os

import numpy as np

from robot_learning_course import fr3
from robot_learning_course.kinematics import TrueKinematics
from robot_learning_course.paths import DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "fk_dataset_fast.npz"))
    args = parser.parse_args()

    rng = np.random.default_rng(1)
    q = rng.uniform(fr3.JOINT_LOWER, fr3.JOINT_UPPER, size=(args.n, fr3.NUM_JOINTS))
    x = TrueKinematics().fk(q)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, q=q, x=x)
    print(f"Saved {args.n} samples to {args.out}")


if __name__ == "__main__":
    main()
