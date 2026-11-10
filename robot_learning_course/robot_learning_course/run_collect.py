"""
Data collection = simulator + motor babbling + collector, all with the same number of robots.
    run_collect --num-envs 9 --num-samples 50000
    run_collect --num-envs 9 --headless                 (no window: much faster)
When the collector has saved the dataset, everything stops.
Closing the 3D window (or Ctrl+C) saves what has been collected so far.
"""
import argparse
import os

from robot_learning_course.babbling import Babbling
from robot_learning_course.collector import Collector
from robot_learning_course.paths import DATA_DIR
from robot_learning_course.simulator import Simulator


def main():
    parser = argparse.ArgumentParser(description="Collect a dataset by motor babbling")
    parser.add_argument("--num-envs", type=int, default=9)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "fk_dataset.npz"))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    sim = Simulator(args.num_envs, headless=args.headless)
    babbling = Babbling(args.num_envs)
    collector = Collector(args.num_samples, args.output)

    cycle = 0
    try:
        while sim.window_open() and not collector.done:
            if cycle % 2 == 0:                       # the simulator runs at 100 Hz, the commands at 50 Hz
                q_cmd = babbling.step()
                for k in range(args.num_envs):
                    sim.command(k, q_cmd[k])
            sim.step()
            for k in range(args.num_envs):           # we record the MEASURED q, not the command
                collector.record(sim.joint_angles(k), sim.flange_position(k), k, sim.time)
            cycle += 1
    except KeyboardInterrupt:
        pass
    finally:
        if not collector.done and len(collector.q) > 0:
            collector.save()                         # keep what we have
        sim.close()


if __name__ == "__main__":
    main()
