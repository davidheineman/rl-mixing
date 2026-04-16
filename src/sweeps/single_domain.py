from experiments import Experiment, Mix, MixComponent
from sweeps.defaults import DATASET, base_experiment


TEST_DOMAINS = [
    "dapo_math",
    "skywork_math",
    # "math_proofs",
    "multiturn_chat",
    "reasoning_gym",
    # "competitive_coding",
    "structured_outputs",
    "instruction_following",
    "mcqa",
]


def get_experiments() -> list[Experiment]:
    base = base_experiment(name="single-domain-qwen3-1.7b", mix=Mix([]))

    experiments = []
    for domain in TEST_DOMAINS:
        mix = Mix([MixComponent(DATASET, 1.0, split=domain)])
        exp = base.vary_mix(mix, name_suffix=domain)
        experiments.append(exp)

    return experiments
