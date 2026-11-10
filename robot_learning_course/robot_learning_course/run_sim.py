"""
The simulator alone: one or many FR3 robots, standing at home or moved by hand.
    run_sim --num-envs 9
    run_sim --q 0.5 0 0 -1.5 0 1.5 0.8
Close the 3D window to stop.
"""
import argparse

from robot_learning_course.simulator import Simulator


def main():
    parser = argparse.ArgumentParser(description="Show the simulated FR3 robots")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--q", type=float, nargs=7, help="joint command for robot 0 [rad]")
    args = parser.parse_args()

    sim = Simulator(args.num_envs)
    if args.q is not None:
        sim.command(0, args.q)
    try:
        while sim.window_open():
            sim.step()
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
