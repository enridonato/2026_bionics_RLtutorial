# Optional exercises

All exercises are **optional** and independent of each other. The code works without them.
Each one is marked in the source with `# TODO (optional exercise N)`.
Difficulty: ★ easy · ★★ medium · ★★★ challenging.
Solutions are discussed in class.

| # | File | What to do | Level |
|---|---|---|---|
| 1 | `babbling.py` | Sample babbling goals around HOME instead of uniformly in the joint limits | ★ |
| 2 | `learning/models.py` | Replace `SiLU` with `ReLU` or `Tanh`; compare position error and Jacobian | ★ |
| 3 | `learning/data.py` | Split train/val/test by **time** instead of randomly | ★★ |
| 4 | `controller.py` | Add a velocity **feed-forward** term to the task-space controller | ★★ |
| 5 | `controller.py` | Compare the learned Jacobian with the true one; explain column 7 | ★ |
| 6 | `controller.py` | Use the **null space** to keep the arm close to HOME | ★★★ |

---

## 1 · Where you babble is where you know ★
In `random_configurations()` replace the uniform sampling with

```python
q = fr3.HOME + 0.5 * self.rng.standard_normal((self.num_envs, 7))
return fr3.clip_to_limits(q, self.margin)
```

Collect a new dataset (use a different `--output` so you keep the old one) and retrain the FK.

**Questions:**
* Is the error on the test split better or worse than before?
* Is the error on *uniformly sampled* configurations (notebook 02, section 3) better or worse?
* Which dataset would you use to control the robot near HOME? And far from it?

## 2 · Smooth or not smooth? ★
In `learning/models.py` change `nn.SiLU()` into `nn.ReLU()` (then `nn.Tanh()`). Retrain with `train_fk --out ~/robot_learning_files/models/fk_relu.pt` and re-run notebook 02, section 4, with that model.

**Questions:**
* How does the position error change?
* What does the Jacobian look like along the joint sweep?
* Run the controller with `run_control --model .../fk_relu.pt`. Do you notice any difference in the motion?

## 3 · An honest test set ★★
Consecutive samples are 10 ms apart and almost identical. A random split puts near-twins in train and test, so the test error is too optimistic.
In `split_indices()` split by **time**: the last 15 % of the recording is the test set, the 15 % before it is validation. Give `load_dataset` an option `return_time=False` that also returns the time stamps `t` when it is `True` (the other scripts and the notebooks expect only `q, x`).

**Expected result:** on our 50k dataset the test error goes from 1.38 mm (random split) to about **53 mm** (time split). The time split is the honest estimate of how the model behaves in **new** motions.

## 4 · Feed-forward ★★
The proportional law `v = K e` cannot follow a moving target without error. At steady state `v` must equal the target speed `V`, so `e = V/K`. For the circle (r = 0.12 m, period 8 s, K = 3): **31 mm**.
Add the target velocity, estimated by finite differences between two control steps:

```python
if self.x_target_prev is not None:
    v = v + (self.x_target - self.x_target_prev) * self.rate
self.x_target_prev = self.x_target.copy()
```

(Initialize `self.x_target_prev = None` in `__init__`.)
**Expected result:** tracking error from about 31 mm to about **3 mm**. Compare `tracking_error.png` before and after.

## 5 · Learned vs. true Jacobian ★
In `jacobian()` (or in `control_step()`), compute `TrueKinematics().jacobian(self.q)` and print the relative error `‖J − J_true‖ / ‖J_true‖` once per second. Create the `TrueKinematics` object **once**, in `__init__`.
**Expected result:** about 4 % on the circle. The last column is ≈ 0: joint 7 rotates the flange about its own axis, which changes its orientation but not its **position**.

## 6 · Using the redundancy ★★★
With 7 joints and 3 constraints, 4 joint-space directions do not move the flange: the **null space** of `J`. Use them to pull the arm towards HOME without disturbing the task:

```python
J_pinv = np.linalg.pinv(J)
N = np.eye(7) - J_pinv @ J
qdot = qdot + N @ (k0 * (fr3.HOME - self.q))     # k0 ~ 1.0
```

**Questions:**
* What happens to the posture during a long run on the "eight", with and without this term?
* Why does this term not change the flange velocity (to first order)?
