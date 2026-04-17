import copy

from sweeps.defaults import DATASET, base_experiment
from sweeps.mixes_1b import NVIDIA_WEIGHTS, nvidia_mix, natural_mix

from experiments import Experiment, InfraConfig, Mix


def base_experiment_14b_10x(name: str, mix: Mix) -> Experiment:
    exp = base_experiment(name, mix)
    exp.model = "Qwen/Qwen3-14B-Base"
    exp.training = copy.deepcopy(exp.training)
    exp.training.total_episodes *= 10
    exp.beaker = copy.deepcopy(exp.beaker)
    exp.beaker.num_nodes = 2
    exp.infra = InfraConfig(
        num_learners_per_node=[4, 4],
        vllm_num_engines=4,
        vllm_tensor_parallel_size=2,
    )
    return exp


def get_experiments() -> list[Experiment]:
    return [
        base_experiment_14b_10x(
            name="nvidia-mix-qwen3-14b-10x",
            mix=nvidia_mix(DATASET, NVIDIA_WEIGHTS),
        ),
        base_experiment_14b_10x(
            name="natural-mix-qwen3-14b-10x",
            mix=natural_mix(DATASET, list(NVIDIA_WEIGHTS.keys())),
        ),
    ]
