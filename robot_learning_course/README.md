# robot_learning_course

**MSc course "Robot Learning"** — hands-on package, Python-only version.
A simulated **Franka Research 3** (MuJoCo) learns its own kinematics from data. The data comes from *motor babbling*, the networks are trained with **PyTorch**, and the robot is then controlled with the learned model.

```
run_collect:   Babbling ─▶ Simulator (MuJoCo, N robots) ─▶ Collector ─▶ dataset.npz ─▶ train_fk / train_ik
                                                                                             │
run_control:   Target ─▶ Controller (learned Jacobian) ─▶ Simulator (MuJoCo, 1 robot)       │
                              ▲                                                              │
                              └───────────────────────── learned model ◀─────────────────────┘
```

---

## 1. Installation (Linux, Windows or macOS, Python 3.10 or newer)

```bash
python -m venv .venv
source .venv/bin/activate                                            # on Windows: .venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough
pip install -e .                                                     # run it inside this folder
```

Check: `run_sim --num-envs 9` must open a window with nine robots. Close the window to stop.

## 2. The hands-on, step by step

| Step | Command | Time |
|---|---|---|
| 0. Warm-up (2-link arm, at home) | `jupyter notebook notebooks/01_planar_arm_toy.ipynb` | 15' |
| 1. Simulator | `run_sim --num-envs 9` | 20' |
| 2. Data collection | `run_collect --num-envs 9 --num-samples 50000` | 25' |
| 3. Forward kinematics | `train_fk` | 30' |
| 4. Inverse kinematics | `train_ik --mode naive` then `--mode distal` | 25' |
| 5. Task-space control | `run_control --shape circle` | 40' |

With a window the simulation runs in real time. Add `--headless` to `run_collect` or `run_control` to run without a window, much faster.

At the end, `run_control` prints the mean tracking error and saves the plot `tracking_error.png`. Other options: `--shape point | circle | eight`, `--feedback model`, `--gain 5`, `--center 0.6 0 0.2`, `--seconds 40`.

Move the robot by hand:
```bash
run_sim --q 0.5 0.0 0.0 -1.5 0.0 1.5 0.8
```

### Where files go, and the fallback
Datasets and models are written to `~/robot_learning_files/{data,models}/`.
If a file is missing, the **pretrained** one in `pretrained/` is used automatically. The pretrained dataset has 200k samples (16 robots, 3 minutes); the models are trained on it. So a failed step never blocks the next one.

## 3. Contents

| File | What it does |
|---|---|
| `robot_learning_course/simulator.py` | The MuJoCo simulator: 1 or N robots (`num_envs`) |
| `robot_learning_course/babbling.py` | Motor babbling: random goals, smooth interpolation |
| `robot_learning_course/collector.py` | Stores the pairs (q, x), saves `.npz` |
| `robot_learning_course/target.py` | Desired trajectory: `point`, `circle`, `eight` |
| `robot_learning_course/controller.py` | Task-space control with the **Jacobian of the learned FK** (damped least squares) |
| `robot_learning_course/run_sim.py`, `run_collect.py`, `run_control.py` | The three experiments: they connect the pieces above |
| `robot_learning_course/learning/models.py` | The MLP (normalization stored inside the model) |
| `robot_learning_course/learning/train_fk.py` | Train x = f(q) |
| `robot_learning_course/learning/train_ik.py` | Train q = g(x): `naive` vs `distal` teacher |
| `robot_learning_course/learning/make_dataset_fast.py` | "Shortcut" dataset without physics (for comparison) |
| `robot_learning_course/kinematics.py` | **True** kinematics from MuJoCo, used only to evaluate |
| `robot_learning_course/fr3.py`, `scene.py`, `paths.py` | Robot constants, multi-robot scene, file locations |
| `notebooks/00…03` | Python primer (self-study), 2-link warm-up, FR3 forward and inverse kinematics |
| `exercises/` | Optional exercises (`# TODO (optional exercise N)` in the code) |
| `mujoco_models/fr3` | FR3 model from MuJoCo Menagerie (Apache-2.0) |

Tests: `python -m pytest test/`

## 4. Reference results (measured with the commands above, on a laptop CPU)

| Experiment | Result |
|---|---|
| Collection, 9 robots, 50k samples, `--headless` | 11 s (56 s of simulated time) |
| FK training, 100 epochs | about 1 minute |
| FK, test error (random split) | 1.38 mm |
| FK, test error (split by time, exercise 3) | 53 mm |
| IK naive / distal teacher (50k data) | 319 mm / 28 mm (median distal 11 mm) |
| Circle tracking, P only / + feed-forward (exercise 4) | 31.3 mm / 2.8 mm |
| Figure eight, P only / + feed-forward | 30.0 mm / 4.0 mm |
| Fixed point, measured / model feedback | 0.0 mm / 6.3 mm |

## 5. Troubleshooting

* **The 3D window is slow.** Use fewer robots, or add `--headless`.
* **`run_sim: command not found`.** Activate the environment (`source .venv/bin/activate`) and check that `pip install -e .` was run inside this folder.
* **macOS: the 3D window does not open.** MuJoCo needs its own launcher there: `mjpython -m robot_learning_course.run_sim --num-envs 9`. The same works for `run_collect` and `run_control`.

## 6. Credits

FR3 model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0). MuJoCo © Google DeepMind.
