# Introduction to Robot Learning, **MSc in Bionics Engineering · 2026/2027** 

A simulated **Franka Research 3** robot arm learns **its own body** from data, then uses what it learned to **move**.
No robotics or machine-learning background is required: everything is explained along the way.

It runs on Linux, Windows and macOS.

```
1 · Babble        →   2 · Collect          →   3 · Learn              →   4 · Control
the robot moves       record (q, x)            a neural network learns     move the hand with the
at random             pairs                    x = f(q) and q = g(x)        learned model only
```

Tools: **MuJoCo** (the simulator), **PyTorch** (the learning).

## What is in here

| Folder | Content |
|---|---|
| [`robot_learning_course/`](robot_learning_course) | The Python package for the lab: simulator, data collection, training, controller |
| [`robot_learning_course/notebooks/`](robot_learning_course/notebooks) | `00` Python primer (**do it before the course**) · `01` 2-link arm (**at home, before the hands-on**) · `02`–`03` FR3 kinematics |
| [`robot_learning_course/exercises/`](robot_learning_course/exercises) | Optional exercises, marked `TODO` in the code |

## Quick start

You need **Python 3.10 or newer**.

```bash
git clone https://github.com/enridonato/2026_bionics_RLtutorial.git
cd 2026_bionics_RLtutorial/robot_learning_course

python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Check that it works: a window with nine FR3 robots should open.

```bash
run_sim --num-envs 9
```

The lab, step by step (commands, expected results, troubleshooting), is in [`robot_learning_course/README.md`](robot_learning_course/README.md).
If a step fails, ready-made data and models in `robot_learning_course/pretrained/` are used automatically, so you can always continue.

## The course in ten chapters

Chapters 1–8: theory. Chapters 9–10: hands-on, with everything already installed.

1. Why should robots learn?
2. Where does the data come from?
3. The robot's body in numbers
4. Learning from examples
5. Neural networks
6. Learning the robot's body
7. From a learned model to motion
8. Where robot learning is going
9. The software: MuJoCo
10. Hands-on: the FR3 learns its own body

## Credits

- FR3 model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0).
- Course material © 2026 Enrico Donato. Questions: [enrico.donato@santannapisa.it](mailto:enrico.donato@santannapisa.it).
