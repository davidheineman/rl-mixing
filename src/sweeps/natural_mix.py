from huggingface_hub import HfApi

from rl_mixing.experiments import Experiment, Mix, MixComponent
from sweeps.defaults import DATASET, base_experiment


ALL_DOMAINS = [
  ### More useful domains
  "dapo_math",
  "skywork_math",
  # "math_proofs",
  "multiturn_chat",
  "reasoning_gym",
  # "competitive_coding",
  "structured_outputs",
  "instruction_following",
  "mcqa",
  
  ### Less useful domains
  # "agentic_tool_use",
  # "swe_pivot",
  # "identity_following",
  # "calendar",
  # "safety",
  # "workplace_assistant",
  # "genrm",
]


def get_split_sizes(dataset: str, splits: list[str]) -> dict[str, int]:
    """Fetch dataset sizes on HuggingFace"""
    api = HfApi()
    info = api.dataset_info(dataset)
    card = info.card_data
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
    sizes = get_split_sizes(dataset, domains)

    missing = set(domains) - set(sizes)
    if missing:
        raise ValueError(f"Could not find sizes for splits: {missing}")

    total = sum(sizes[d] for d in domains)
    return Mix([
        MixComponent(dataset, sizes[d] / total, split=d)
        for d in domains
    ])


def get_experiments() -> list[Experiment]:
    mix = natural_mix(DATASET, ALL_DOMAINS)
    return [base_experiment(name="natural-mix-qwen3-1.7b", mix=mix)]
