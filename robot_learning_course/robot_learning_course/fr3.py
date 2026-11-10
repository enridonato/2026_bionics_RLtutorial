"""
Basic facts about the Franka Research 3 (FR3) robot, collected in one place.

Values copied from the MuJoCo Menagerie model (mujoco_models/fr3/fr3.xml),
which follows the official Franka datasheet.
"""
import numpy as np

# The FR3 has 7 revolute joints -> 7 numbers describe its configuration q.
NUM_JOINTS = 7

JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, NUM_JOINTS + 1)]

# Joint limits [rad]. Note: they are NOT symmetric for joints 4 and 6.
JOINT_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
JOINT_UPPER = np.array([+2.9007, +1.8361, +2.9007, -0.1169, +2.8763, 4.6216, +3.0508])

# A comfortable "home" configuration (arm bent, flange pointing down).
HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Name of the point whose position we learn (the flange, where a tool would go).
EE_SITE = "attachment_site"


def clip_to_limits(q, margin=0.0):
    """Clip a configuration inside the joint limits (optionally keeping a margin, in rad)."""
    return np.clip(q, JOINT_LOWER + margin, JOINT_UPPER - margin)
