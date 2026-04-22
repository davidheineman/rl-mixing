from typing import Any
from huggingface_hub import HfApi

from experiments import Experiment, Mix, MixComponent
from sweeps.defaults import DATASET, base_experiment

# Mixture weights from stage 1 training: https://arxiv.org/pdf/2604.12374v1#page=26.84
NVIDIA_WEIGHTS = {
    ### More useful domains
    "dapo_math":              0.0136,
    "skywork_math":           0.0544,
    "math_proofs":            0.0068,
    "multiturn_chat":         0.0136,
    "reasoning_gym":          0.0272,
    "competitive_coding":     0.1224,
    "structured_outputs":     0.0272,
    "instruction_following":  0.1224,
    "mcqa":                   0.0408,

    ### Less useful domains
    # "agentic_tool_use":     0.2721,
    # "swe_pivot":            0.0408,
    # "identity_following":   0.0136,
    # "calendar":             0.0272,
    # "safety":               0.0272,
    # "workplace_assistant":  0.0136,
    # "genrm":                0.1769,
}


def nvidia_mix(dataset: str, weights: dict[str, float]) -> Mix:
    """Build a Mix from explicit per-domain weights, renormalized to sum to 1."""
    active = {d: w for d, w in weights.items() if d in NVIDIA_WEIGHTS.keys()}
    total = sum(active.values())
    return Mix([
        MixComponent(dataset, w / total, split=d)
        for d, w in active.items()
    ])


def _get_split_sizes(dataset: str, splits: list[str]) -> dict[str, int]:
    """Fetch dataset sizes on HuggingFace"""
    api = HfApi()
    info = api.dataset_info(dataset)
    card: Any = info.card_data
    if card is None or card.dataset_info is None:
        raise ValueError(f"No dataset_info metadata found for {dataset}")

    split_list = card.dataset_info.get("splits", [])
    return {
        s["name"]: s["num_examples"]
        for s in split_list
        if s["name"] in splits
    }


def natural_mix(dataset: str, domains: list[str]) -> Mix:
    """Build a Mix where proportions match the natural dataset split sizes."""
    sizes = _get_split_sizes(dataset, domains)

    missing = set(domains) - set(sizes)
    if missing:
        raise ValueError(f"Could not find sizes for splits: {missing}")

    total = sum(sizes[d] for d in domains)
    return Mix([
        MixComponent(dataset, sizes[d] / total, split=d)
        for d in domains
    ])


def get_experiments() -> list[Experiment]:
    return [
        base_experiment(
            name="nvidia-mix-qwen3-1.7b",
            mix=nvidia_mix(DATASET, NVIDIA_WEIGHTS),
        ),
        base_experiment(
            name="natural-mix-qwen3-1.7b",
            mix=natural_mix(DATASET, list(NVIDIA_WEIGHTS.keys())),
        ),
    ]
