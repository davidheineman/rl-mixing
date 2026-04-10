"""
Convert Nemotron-RL-Super-Training-Blends data to open-instruct RLVR format
and optionally push to HuggingFace Hub.

The Nemotron blends use a NeMo Gym format with heterogeneous schemas per task
type. This script reads the raw JSONL files (after placeholder resolution),
maps each record to the open-instruct schema, and routes to the correct
verifier name.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# open-instruct target schema:
#   messages:      list[dict] with role/content  (the prompt)
#   ground_truth:  str or list[str]              (verifiable answer)
#   dataset:       str                           (verifier routing key)
# ---------------------------------------------------------------------------

# Map Nemotron dataset names → open-instruct verifier names.
# Records whose dataset name is not in this map are skipped (unsupported).
VERIFIER_MAP: dict[str, str] = {
    # --- Math ---
    "super_v3_lcsft_step1000_dapo17k": "math",
    "super_v3_lcsft_step1000_skyworks": "math",
    "super_v3_lcsft_step1000_skyworks_no_omni": "math",
    "super_v3_lcsft_step1000_lean": "math",

    # --- Instruction Following (IFEval-style) ---
    "super_v3_lcsft_step1000_instruction_following": "ifeval",
    "super_v3_lcsft_step1000_calendar_v2": "ifeval",
    "super_v3_lcsft_step1000_multichallenge_vanilla_and_advanced_len40k": "ifeval",
    "super_v3_lcsft_step1000_structured_outputs": "ifeval",
    "inverse_ifeval": "ifeval",

    # --- Competitive Coding ---
    "super_v3_lcsft_step1000_comp_coding": "code",

    # --- Knowledge MCQA ---
    "super_v3_lcsft_step1000_stem_mcqa": "string_matcher",

    # --- Reasoning / Puzzle ---
    "super_v3_lcsft_step1000_reasoning_gym": "puzzle",

    # --- Safety (LLM judge) ---
    "safety_v0.3.0": "general-safety",
    "super_v3_lcsft_step1000_jailbreak_and_overrefusal": "general-safety",
    "super_v3_lcsft_step1000_jailbreak_and_overrefusal_harder": "general-safety",

    # --- Identity Following (LLM judge) ---
    "super_identity_w_principle_genrm": "general-quality",

    # --- Agentic (LLM judge -- multi-turn tool use) ---
    "super_v3_lcsft_step1000_tau_pivot": "general-quality",
    "super_v3_lcsft_step1000_single_step_swe_swegym_and_scale": "general-quality",
    "super_v3_lcsft_step1000_workbench": "general-quality",

    # --- GenRM / RLHF (LLM judge) ---
    "hs3": "general-quality",
    "hs4_20260106_combinedrubricsonly": "general-quality",
    "lmarena_5k": "general-quality",

    # --- HuggingFace repo names (from individual datasets, not blends) ---
    "nvidia/Nemotron-RL-instruction_following": "ifeval",
    "nvidia/Nemotron-RL-coding-competitive_coding": "code",
    "nvidia/Nemotron-RL-knowledge-mcqa": "string_matcher",
    "open-r1/codeforces": "code",
    "deepmind/code_contests": "code",
}

# Task types that we can convert with high fidelity (ground-truth verifiable).
# The rest need LLM judges and are lower fidelity.
HIGH_FIDELITY_VERIFIERS = {"math", "gsm8k", "ifeval", "code", "string_matcher", "puzzle"}


# ---------------------------------------------------------------------------
# Per-task-type conversion logic
# ---------------------------------------------------------------------------

@dataclass
class ConvertedRecord:
    messages: list[dict[str, str]]
    ground_truth: str | list[str]
    dataset: str
    source: str = ""


def extract_messages(record: dict) -> list[dict[str, str]] | None:
    """Extract chat messages from a Nemotron record."""
    rcp = record.get("responses_create_params")
    if not rcp or "input" not in rcp:
        return None
    msgs = rcp["input"]
    if not isinstance(msgs, list) or len(msgs) == 0:
        return None
    result = []
    for m in msgs:
        if isinstance(m, dict) and "role" in m and "content" in m:
            result.append({"role": m["role"], "content": m["content"]})
        elif isinstance(m, dict) and "content" in m:
            result.append({"role": "user", "content": m["content"]})
    return result if result else None


def convert_math(record: dict, verifier: str) -> ConvertedRecord | None:
    """Convert math records (DAPO, Skywork, Math-Proofs)."""
    messages = extract_messages(record)
    if not messages:
        return None
    answer = record.get("expected_answer")
    if answer is None:
        return None
    return ConvertedRecord(
        messages=messages,
        ground_truth=str(answer),
        dataset=verifier,
        source=record.get("dataset", ""),
    )


def convert_ifeval(record: dict, verifier: str) -> ConvertedRecord | None:
    """Convert instruction-following records.

    The Nemotron IF datasets use IFEval-style constraints with
    instruction_id_list + kwargs, which is the same format as
    open-instruct's IFEvalVerifier.
    """
    messages = extract_messages(record)
    if not messages:
        return None

    instruction_ids = record.get("instruction_id_list")
    kwargs_list = record.get("kwargs")

    if instruction_ids and kwargs_list:
        constraint = repr([{
            "instruction_id": instruction_ids,
            "kwargs": kwargs_list,
        }])
        return ConvertedRecord(
            messages=messages,
            ground_truth=constraint,
            dataset="ifeval",
            source=record.get("dataset", ""),
        )

    # Fallback: some IF records might have expected_answer directly
    answer = record.get("expected_answer")
    if answer is not None:
        return ConvertedRecord(
            messages=messages,
            ground_truth=str(answer),
            dataset=verifier,
            source=record.get("dataset", ""),
        )
    return None


def convert_code(record: dict, verifier: str) -> ConvertedRecord | None:
    """Convert competitive coding records.

    The verifier_metadata contains unit tests as {inputs, outputs}.
    open-instruct's CodeVerifier expects test cases as the label.
    """
    messages = extract_messages(record)
    if not messages:
        return None

    vm = record.get("verifier_metadata", {})
    unit_tests = vm.get("unit_tests")
    if unit_tests:
        test_cases = []
        inputs = unit_tests.get("inputs", [])
        outputs = unit_tests.get("outputs", [])
        for inp, out in zip(inputs, outputs):
            test_cases.append({"input": inp, "output": out})
        return ConvertedRecord(
            messages=messages,
            ground_truth=json.dumps(test_cases),
            dataset="code",
            source=record.get("dataset", ""),
        )
    return None


def convert_mcqa(record: dict, verifier: str) -> ConvertedRecord | None:
    """Convert MCQA records. Answer is a letter like 'A', 'B', etc."""
    messages = extract_messages(record)
    if not messages:
        return None

    answer = record.get("expected_answer")
    if answer is None:
        return None

    # Wrap in answer tags for StringMatcherVerifier
    return ConvertedRecord(
        messages=messages,
        ground_truth=str(answer),
        dataset="string_matcher",
        source=record.get("dataset", ""),
    )


def convert_generic(record: dict, verifier: str) -> ConvertedRecord | None:
    """Generic fallback: use expected_answer if present, or empty string for
    judge-based verifiers."""
    messages = extract_messages(record)
    if not messages:
        return None

    answer = record.get("expected_answer", "")
    return ConvertedRecord(
        messages=messages,
        ground_truth=str(answer) if answer else "",
        dataset=verifier,
        source=record.get("dataset", ""),
    )


# Dispatch table: verifier name → converter function
CONVERTERS: dict[str, Any] = {
    "math": convert_math,
    "gsm8k": convert_math,
    "ifeval": convert_ifeval,
    "code": convert_code,
    "string_matcher": convert_mcqa,
    "puzzle": convert_generic,
    "general-safety": convert_generic,
    "general-quality": convert_generic,
}


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_record(record: dict) -> ConvertedRecord | None:
    """Convert a single Nemotron record to open-instruct format."""
    ds_name = record.get("dataset", "")

    verifier = VERIFIER_MAP.get(ds_name)
    if verifier is None:
        return None

    converter = CONVERTERS.get(verifier, convert_generic)
    return converter(record, verifier)


def convert_file(
    input_path: Path,
    high_fidelity_only: bool = False,
) -> list[dict]:
    """Convert a full JSONL file, returning list of open-instruct dicts."""
    results = []
    stats: Counter = Counter()

    for record in iter_jsonl(input_path):
        ds_name = record.get("dataset", "unknown")
        stats[f"total/{ds_name}"] += 1

        converted = convert_record(record)
        if converted is None:
            stats[f"skipped/{ds_name}"] += 1
            continue

        if high_fidelity_only and converted.dataset not in HIGH_FIDELITY_VERIFIERS:
            stats[f"filtered/{ds_name}"] += 1
            continue

        results.append({
            "messages": converted.messages,
            "ground_truth": converted.ground_truth,
            "dataset": converted.dataset,
        })
        stats[f"converted/{converted.dataset}"] += 1

    return results, stats


def convert_all(
    input_dir: Path,
    output_dir: Path,
    splits: list[str] | None = None,
    high_fidelity_only: bool = False,
) -> dict[str, list[dict]]:
    """Convert all JSONL files in input_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_splits = {}
    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        split_name = jsonl_path.stem
        if splits and split_name not in splits:
            continue

        log.info(f"Converting {jsonl_path.name}...")
        records, stats = convert_file(jsonl_path, high_fidelity_only=high_fidelity_only)

        if not records:
            log.warning(f"  No records converted from {jsonl_path.name}")
            continue

        out_path = output_dir / f"{split_name}.jsonl"
        with out_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        log.info(f"  Wrote {len(records)} records to {out_path.name}")
        log.info(f"  Stats:")
        for key in sorted(stats):
            log.info(f"    {key}: {stats[key]}")

        all_splits[split_name] = records

    return all_splits


def push_to_hub(
    output_dir: Path,
    repo_id: str,
    private: bool = False,
    max_shard_size: str = "50MB",
):
    """Push converted data to HuggingFace Hub as a dataset."""
    from datasets import Dataset, DatasetDict

    splits = {}
    for jsonl_path in sorted(output_dir.glob("*.jsonl")):
        split_name = jsonl_path.stem
        records = list(iter_jsonl(jsonl_path))
        if records:
            splits[split_name] = Dataset.from_list(records)
            log.info(f"  {split_name}: {len(records)} records")

    if not splits:
        log.error("No splits to push")
        return

    dataset_dict = DatasetDict(splits)
    log.info(f"Pushing to {repo_id} (max_shard_size={max_shard_size})...")
    dataset_dict.push_to_hub(repo_id, private=private, max_shard_size=max_shard_size)
    log.info(f"Done! https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nemotron RL blends to open-instruct format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory with filled JSONL files (after fill_placeholders.py)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/converted"),
        help="Where to write converted JSONL files",
    )
    parser.add_argument(
        "--splits", nargs="*", default=None,
        help="Which splits to convert (e.g., rlvr1 rlvr2). Default: all.",
    )
    parser.add_argument(
        "--high-fidelity-only", action="store_true",
        help="Only include tasks with ground-truth verifiers (math, IF, code, MCQA). "
             "Excludes judge-based tasks (safety, GenRM, identity).",
    )
    parser.add_argument(
        "--push-to-hub", type=str, default=None,
        help="HuggingFace repo ID to push to (e.g., davidheineman/nemotron-rlvr-openinstruct)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Make the HF dataset private",
    )
    parser.add_argument(
        "--max-shard-size", type=str, default="50MB",
        help="Max parquet shard size for HF upload (default: 50MB). "
             "Smaller shards help the HF dataset viewer load without timeouts.",
    )
    args = parser.parse_args()

    convert_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        high_fidelity_only=args.high_fidelity_only,
    )

    if args.push_to_hub:
        push_to_hub(
            args.output_dir, args.push_to_hub,
            private=args.private,
            max_shard_size=args.max_shard_size,
        )


if __name__ == "__main__":
    main()
