"""Launch eval-only runs (no real training, just step-0 evals)."""

import copy

from rl_mixing.experiments import Experiment, Mix
from sweeps.defaults import base_experiment


def get_experiments() -> list[Experiment]:
    base = base_experiment(name="eval-only-qwen3-1.7b", mix=Mix([]))

    training = copy.deepcopy(base.training)
    training.total_episodes = 128
    return [base.vary(
        training=training,
        extra_args=base.extra_args + ["--eval_on_step_0", "true"],
    )]
