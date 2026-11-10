"""
Controller: TASK-SPACE control of the FR3 using the LEARNED forward model.

Idea (differential kinematics, "resolved-rate" control):
    we want the flange to move with velocity  v = K * (x_target - x)          (go towards the target)
    joints and flange velocities are linked by the Jacobian:   x_dot = J(q) q_dot
    so we solve for the joint velocities:      q_dot = J^T (J J^T + lambda^2 I)^-1 v    (damped least squares)
    and integrate them into a position command: q_cmd <- q_cmd + q_dot * dt

Where does J come from? From the NETWORK: J = d f_theta(q) / d q, computed by PyTorch autograd.
No robot model (DH parameters, URDF, ...) is used anywhere in this file.

Parameters: model (~/robot_learning_files/models/fk_model.pt), rate (100 Hz), gain K (3.0 1/s),
            damping lambda (0.05), max_joint_vel (1.0 rad/s), feedback ("measured" or "model")

Used by:  run_control
"""
import os

import numpy as np
import torch

from robot_learning_course import fr3
from robot_learning_course.learning.models import load_model
from robot_learning_course.paths import MODELS_DIR, find_file


class Controller:
    def __init__(self, model=os.path.join(MODELS_DIR, "fk_model.pt"), rate=100.0, gain=3.0,
                 damping=0.05, max_joint_vel=1.0, feedback="measured"):
        self.rate = rate
        self.K = gain
        self.damping = damping
        self.max_qdot = max_joint_vel
        self.feedback = feedback

        self.fk = load_model(find_file(model))   # the learned forward kinematics f_theta(q)

        self.q = None          # measured joint angles   (7,)
        self.x = None          # measured flange position (3,)
        self.x_target = None   # desired flange position  (3,)
        self.q_cmd = None      # joint command we are sending (7,)
        print(f"Task-space controller ready (model: {model}, feedback: {feedback})")

    # ------------------------------------------------------------------ the learned model
    def predict(self, q):
        """x = f_theta(q): where the NETWORK thinks the flange is."""
        with torch.no_grad():
            return self.fk(torch.tensor(q, dtype=torch.float32)).numpy().astype(float)

    def jacobian(self, q):
        """J = d f_theta / d q (3x7), by automatic differentiation of the network."""
        # TODO (optional exercise 5): compare with the TRUE Jacobian, from
        #   robot_learning_course.kinematics.TrueKinematics().jacobian(q). Print the difference:
        #   which column is (almost) zero, and why? (hint: look at joint 7 and where the flange is)
        q_t = torch.tensor(q, dtype=torch.float32)
        return torch.autograd.functional.jacobian(self.fk, q_t).numpy().astype(float)

    # ------------------------------------------------------------------ the control law
    def control_step(self, q, x, x_target):
        """One control cycle: from the sensors and the target to the new joint command (7,)."""
        self.q, self.x, self.x_target = q, x, x_target
        if self.q_cmd is None:
            self.q_cmd = self.q.copy()               # start from where the robot is

        # 1. Where is the flange? Measured by the "sensors", or predicted by the model?
        x = self.x if self.feedback == "measured" else self.predict(self.q)
        error = self.x_target - x                    # task-space error [m]

        # 2. Desired flange velocity: proportional to the error
        v = self.K * error
        # TODO (optional exercise 4): add a FEED-FORWARD term, the velocity of the target itself:
        #   v = x_target_dot + K * error, with x_target_dot = (x_target - x_target_previous) * rate.
        #   Look at tracking_error.png before and after: what changes on the circle?

        # 3. Joint velocities through the (learned!) Jacobian, damped least squares
        J = self.jacobian(self.q)                    # 3 x 7
        qdot = J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * np.eye(3), v)
        # TODO (optional exercise 6): 7 joints, 3 constraints -> 4 "free" directions (null space).
        #   Use them to stay close to HOME without disturbing the flange:
        #   qdot += (I - J_pinv @ J) @ (k0 * (fr3.HOME - self.q)),   J_pinv = np.linalg.pinv(J), k0 ~ 1.0
        qdot = np.clip(qdot, -self.max_qdot, self.max_qdot)   # safety: limit the joint speed

        # 4. Integrate: new position command, inside the joint limits
        self.q_cmd = fr3.clip_to_limits(self.q_cmd + qdot / self.rate, margin=0.02)
        return self.q_cmd
