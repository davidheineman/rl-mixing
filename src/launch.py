from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from experiments import Experiment, launch_sweep, load_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Launch RL mixing experiments on Beaker via open-instruct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="YAML config file(s) to launch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        help="Path to a Python file that defines a `get_experiments() -> list[Experiment]` function",
    )
    parser.add_argument(
        "--follow", "-f",
        action="store_true",
        help="Stream logs for the last launched job using bstream",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="List experiments from configs without launching",
    )
    args = parser.parse_args()

    if not args.configs and not args.sweep:
        parser.print_help()
        sys.exit(1)

    experiments: list[Experiment] = []

    for config_path in args.configs or []:
        path = Path(config_path)
        if not path.exists():
            print(f"Error: config file not found: {path}")
            sys.exit(1)
        experiments.extend(load_experiments(path))

    if args.sweep:
        sweep_path = Path(args.sweep)
        if not sweep_path.exists():
            print(f"Error: sweep file not found: {sweep_path}")
            sys.exit(1)
        spec = importlib.util.spec_from_file_location("sweep_module", sweep_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "get_experiments"):
            print(f"Error: {sweep_path} must define a `get_experiments()` function")
            sys.exit(1)
        experiments.extend(mod.get_experiments())

    if not experiments:
        print("No experiments found.")
        sys.exit(1)

    if args.list_only:
        print(f"\n{len(experiments)} experiment(s):\n")
        for i, exp in enumerate(experiments):
            print(f"  [{i+1}] {exp.name}")
            print(f"      model: {exp.model}")
            print(f"      mix:   {exp.mix.summary()}")
            print(f"      lr={exp.training.learning_rate}  beta={exp.training.beta}  "
                  f"nspp={exp.training.num_samples_per_prompt_rollout}")
            print()
        return

    if args.dry_run:
        all_ok = all(exp.validate() for exp in experiments)
        if not all_ok:
            sys.exit(1)

    launch_sweep(experiments, dry_run=args.dry_run)

    if args.follow and not args.dry_run:
        last_with_id = next(
            (exp for exp in reversed(experiments) if getattr(exp, "beaker_experiment_id", None)),
            None,
        )
        if last_with_id:
            print(f"\nFollowing experiment: {last_with_id.beaker_experiment_id}")
            os.execvp("bstream", ["bstream", last_with_id.beaker_experiment_id])


if __name__ == "__main__":
    main()
