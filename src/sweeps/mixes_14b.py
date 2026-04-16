import copy

from sweeps.defaults import DATASET, base_experiment
from sweeps.mixes_1b import NVIDIA_WEIGHTS, nvidia_mix, natural_mix

from experiments import Experiment, Mix


def base_experiment_14b(name: str, mix: Mix) -> Experiment:
    exp = base_experiment(name, mix)
    exp.model = "Qwen/Qwen3-14B-Base"
    exp.beaker = copy.deepcopy(exp.beaker)
    exp.beaker.num_nodes = 2
    return exp


def get_experiments() -> list[Experiment]:
    return [
        base_experiment_14b(
            name="nvidia-mix-qwen3-14b",
            mix=nvidia_mix(DATASET, NVIDIA_WEIGHTS),
        ),
        base_experiment_14b(
            name="natural-mix-qwen3-14b",
            mix=natural_mix(DATASET, list(NVIDIA_WEIGHTS.keys())),
        ),
    ]
