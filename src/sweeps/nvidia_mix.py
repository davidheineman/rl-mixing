from rl_mixing.experiments import Experiment, Mix, MixComponent
from sweeps.defaults import DATASET, base_experiment


def get_experiments() -> list[Experiment]:
    """ Mixture weights based on stage 1 training: https://arxiv.org/pdf/2604.12374v1#page=26.84 """
    mix = Mix([
        ### More useful domains
        MixComponent(DATASET, 0.0136, split="dapo_math"),
        MixComponent(DATASET, 0.0544, split="skywork_math"),
        # MixComponent(DATASET, 0.0068, split="math_proofs"),
        MixComponent(DATASET, 0.0136, split="multiturn_chat"),
        MixComponent(DATASET, 0.0272, split="reasoning_gym"),
        # MixComponent(DATASET, 0.1224, split="competitive_coding"),
        MixComponent(DATASET, 0.0272, split="structured_outputs"),
        MixComponent(DATASET, 0.1224, split="instruction_following"),
        MixComponent(DATASET, 0.0408, split="mcqa"),
        
        ### Less useful domains
        # MixComponent(DATASET, 0.2721, split="agentic_tool_use"),
        # MixComponent(DATASET, 0.0408, split="swe_pivot"),
        # MixComponent(DATASET, 0.0136, split="identity_following"),
        # MixComponent(DATASET, 0.0272, split="calendar"),
        # MixComponent(DATASET, 0.0272, split="safety"),
        # MixComponent(DATASET, 0.0136, split="workplace_assistant"),
        # MixComponent(DATASET, 0.1769, split="genrm"),
    ])

    # Recompute mixture weights so they sum to 1
    total = sum(c.proportion for c in mix.components)
    if abs(total - 1.0) > 1e-6:
        for c in mix.components:
            c.proportion = c.proportion / total

    return [base_experiment(name="nvidia-mix-qwen3-1.7b", mix=mix)]
