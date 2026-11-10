"""
The TRUE forward kinematics of the FR3, computed by MuJoCo from the robot description.

We use it ONLY to check our learned models ("ground truth"): in the real world you would
not have it (or it would be imprecise) -- that is the whole point of learning a model.
"""
import mujoco
import numpy as np

from robot_learning_course import fr3
from robot_learning_course.scene import build_scene, joint_indices


class TrueKinematics:
    def __init__(self):
        self.model, self.data, _ = build_scene(num_envs=1)
        self.qpos_ids, _, _ = joint_indices(self.model, 0)
        self.site = self.model.site(f"env0/{fr3.EE_SITE}").id

    def fk(self, q):
        """Flange position(s) for configuration(s) q: (7,) -> (3,)  or  (N, 7) -> (N, 3)."""
        q = np.atleast_2d(q)
        x = np.zeros((len(q), 3))
        for i in range(len(q)):
            self.data.qpos[self.qpos_ids] = q[i]
            mujoco.mj_kinematics(self.model, self.data)  # geometry only: no forces, no time
            x[i] = self.data.site_xpos[self.site]
        return x[0] if len(x) == 1 else x

    def jacobian(self, q):
        """True 3x7 position Jacobian dx/dq at configuration q."""
        self.data.qpos[self.qpos_ids] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site)
        return jacp[:, self.qpos_ids]  # for hinge joints the qpos and dof indices coincide here
