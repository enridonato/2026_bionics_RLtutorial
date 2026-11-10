"""
Task-space control with the learned model = simulator (1 robot) + target generator + controller.
    run_control --shape eight
    run_control --feedback model
At the end the tracking error is plotted in ~/robot_learning_files/models/tracking_error.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # draw plots into files (no window needed)
import matplotlib.pyplot as plt
import numpy as np

from robot_learning_course.controller import Controller
from robot_learning_course.paths import MODELS_DIR
from robot_learning_course.simulator import Simulator
from robot_learning_course.target import Target


def main():
    parser = argparse.ArgumentParser(description="Control the FR3 with the learned model")
    parser.add_argument("--shape", choices=["point", "circle", "eight"], default="circle")
    parser.add_argument("--model", default=os.path.join(MODELS_DIR, "fk_model.pt"))
    parser.add_argument("--feedback", choices=["measured", "model"], default="measured")
    parser.add_argument("--gain", type=float, default=3.0)
    parser.add_argument("--center", type=float, nargs=3, default=[0.45, 0.0, 0.45], help="target center [m]")
    parser.add_argument("--seconds", type=float, default=24.0, help="duration of the experiment")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    sim = Simulator(1, headless=args.headless)
    target = Target(args.shape, args.center)
    controller = Controller(args.model, gain=args.gain, feedback=args.feedback)

    times, errors = [], []                           # tracking error, always with the MEASURED position
    try:
        while sim.window_open() and sim.time < args.seconds:
            x_target = target.position(sim.time)
            sim.set_target(x_target)
            x = sim.flange_position(0)
            sim.command(0, controller.control_step(sim.joint_angles(0), x, x_target))
            sim.step()
            times.append(sim.time)
            errors.append(1000 * float(np.linalg.norm(x_target - x)))
            if len(errors) % 100 == 0:
                print(f"t = {sim.time:5.1f} s | tracking error {errors[-1]:6.1f} mm")
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()

    print(f"Mean tracking error over the last 8 s: {np.mean(errors[-800:]):.1f} mm")
    os.makedirs(MODELS_DIR, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(times, errors)
    plt.xlabel("time [s]"), plt.ylabel("tracking error [mm]"), plt.grid(True)
    plt.title(f"Task-space control: {args.shape}")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "tracking_error.png"), dpi=150)


if __name__ == "__main__":
    main()
