"""
Convert individual HuggingFace datasets to open-instruct RLVR format.

Each dataset has its own schema, so we define per-dataset converter functions.
The output format is the same as open-instruct RLVR:

    messages:      list[dict] with role/content  (the prompt)
    ground_truth:  str                           (verifiable answer)
    dataset:       str                           (verifier routing key)

Usage:
    # Convert all datasets locally
    python ingest/convert_individual.py --output-dir ingest/converted_individual

    # Convert a specific dataset
    python ingest/convert_individual.py --output-dir ingest/converted_individual \
        --only nvidia/Nemotron-RL-coding-competitive_coding

    # Convert + push to hub
    python ingest/convert_individual.py --output-dir ingest/converted_individual \
        --push-to-hub davidheineman/nemotron-individual-openinstruct
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output record
# ---------------------------------------------------------------------------

@dataclass
class Record:
    messages: list[dict[str, str]]
    ground_truth: str
    dataset: str  # verifier routing key


# ---------------------------------------------------------------------------
# Per-dataset converters
#
# Each returns a list of Record. The HF dataset object is passed in.
# ---------------------------------------------------------------------------

def _msgs_from_rcp(row: dict) -> list[dict[str, str]] | None:
    """Extract messages from responses_create_params.input (Nemotron-style)."""
    rcp = row.get("responses_create_params")
    if not rcp:
        return None
    msgs = rcp.get("input")
    if not msgs or not isinstance(msgs, list):
        return None
    return [{"role": m["role"], "content": m["content"]} for m in msgs if "role" in m and "content" in m]


def convert_dapo_math(ds) -> list[Record]:
    """BytedTsinghua-SIA/DAPO-Math-17k — math problems with ground truth answers."""
    records = []
    for row in ds:
        prompt = row.get("prompt", [])
        if not prompt:
            continue
        messages = [{"role": m["role"], "content": m["content"]} for m in prompt]
        gt = row.get("reward_model", {}).get("ground_truth", "")
        if gt is None:
            continue
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="math"))
    return records


def convert_skywork_math(ds) -> list[Record]:
    """Skywork/Skywork-OR1-RL-Data (math split) — same schema as DAPO."""
    records = []
    for row in ds:
        prompt = row.get("prompt", [])
        if not prompt:
            continue
        messages = [{"role": m["role"], "content": m["content"]} for m in prompt]
        gt = row.get("reward_model", {}).get("ground_truth", "")
        if gt is None:
            continue
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="math"))
    return records


def convert_math_proofs(ds) -> list[Record]:
    """nvidia/Nemotron-Math-Proofs-v1 — Lean math proofs with formal statements.

    The messages field contains (role, content, reasoning_content). We use
    the first user message as the prompt and the formal_statement as ground truth.
    """
    records = []
    for row in ds:
        messages = row.get("messages", [])
        if not messages:
            continue
        prompt_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") == "user"]
        if not prompt_msgs:
            prompt_msgs = [{"role": "user", "content": row.get("problem", "")}]
        gt = row.get("formal_statement", "")
        if not gt:
            continue
        records.append(Record(messages=prompt_msgs, ground_truth=str(gt), dataset="math"))
    return records


def convert_agentic_tool_use(ds) -> list[Record]:
    """nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 — multi-turn
    tool-use trajectories. Each row is a behavior cloning step with expected_action.
    """
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        expected = row.get("expected_action")
        gt = json.dumps(expected) if expected else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="general-quality"))
    return records


def convert_swe_pivot(ds) -> list[Record]:
    """nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1 — SWE agent trajectories."""
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        expected = row.get("expected_action")
        gt = json.dumps(expected) if expected else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="general-quality"))
    return records


def convert_identity_following(ds) -> list[Record]:
    """nvidia/Nemotron-RL-Identity-Following-v1 — identity/principle following.

    Uses a GenRM judge with a principle rubric as ground truth.
    """
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        gt = row.get("principle", "")
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="general-quality"))
    return records


def convert_calendar(ds) -> list[Record]:
    """nvidia/Nemotron-RL-Instruction-Following-Calendar-v2 — calendar-based IF.

    The expected calendar state serves as ground truth.
    """
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        cal_state = row.get("exp_cal_state")
        gt = json.dumps(cal_state) if cal_state else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="ifeval"))
    return records


def convert_multiturn_chat(ds) -> list[Record]:
    """nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1 — multi-turn
    chat with rubric-based evaluation.
    """
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        rubric = row.get("rubric")
        gt = json.dumps(rubric) if rubric else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="general-quality"))
    return records


def convert_reasoning_gym(ds) -> list[Record]:
    """nvidia/Nemotron-RL-ReasoningGym-v1 — puzzle/reasoning tasks with
    verifiable answers.
    """
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            messages = [{"role": "user", "content": row.get("question", "")}]
        answer = row.get("answer", "")
        if not answer:
            continue
        records.append(Record(messages=messages, ground_truth=str(answer), dataset="puzzle"))
    return records


def convert_safety(ds) -> list[Record]:
    """nvidia/Nemotron-RL-Safety-v1 — safety preference data.

    Uses the prompt + principle as the task, with empty ground truth (judge-based).
    """
    records = []
    for row in ds:
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        messages = [{"role": "user", "content": prompt}]
        gt = row.get("principle", "")
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="general-safety"))
    return records


def convert_workplace_assistant(ds) -> list[Record]:
    """nvidia/Nemotron-RL-agent-workplace_assistant — tool-calling workplace tasks.

    Ground truth is a list of expected tool calls.
    """
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        gt_calls = row.get("ground_truth")
        gt = json.dumps(gt_calls) if gt_calls else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="general-quality"))
    return records


def convert_competitive_coding(ds) -> list[Record]:
    """nvidia/Nemotron-RL-coding-competitive_coding — competitive programming
    with unit test verification.
    """
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        vm = row.get("verifier_metadata", {})
        unit_tests = vm.get("unit_tests") if vm else None
        if not unit_tests:
            continue
        test_cases = []
        inputs = unit_tests.get("inputs", [])
        outputs = unit_tests.get("outputs", [])
        for inp, out in zip(inputs, outputs):
            test_cases.append({"input": inp, "output": out})
        records.append(Record(
            messages=messages,
            ground_truth=json.dumps(test_cases),
            dataset="code",
        ))
    return records


def convert_structured_outputs(ds) -> list[Record]:
    """nvidia/Nemotron-RL-instruction_following-structured_outputs — structured
    output following with schema validation.
    """
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        schema = row.get("schema_str", "")
        gt = schema if schema else ""
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="ifeval"))
    return records


def convert_instruction_following(ds) -> list[Record]:
    """nvidia/Nemotron-RL-instruction_following — IFEval-style instruction following.

    Has instruction_id_list and kwargs, matching open-instruct's IFEvalVerifier.
    """
    records = []
    for row in ds:
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        messages = [{"role": "user", "content": prompt}]
        instruction_ids = row.get("instruction_id_list")
        kwargs_list = row.get("kwargs")
        if instruction_ids and kwargs_list:
            gt = json.dumps([{
                "instruction_id": instruction_ids,
                "kwargs": kwargs_list,
            }])
        else:
            continue
        records.append(Record(messages=messages, ground_truth=gt, dataset="ifeval"))
    return records


def convert_mcqa(ds) -> list[Record]:
    """nvidia/Nemotron-RL-knowledge-mcqa — multiple choice QA with letter answers."""
    records = []
    for row in ds:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        answer = row.get("expected_answer", "")
        if not answer:
            continue
        records.append(Record(messages=messages, ground_truth=str(answer), dataset="string_matcher"))
    return records


def convert_genrm(ds) -> list[Record]:
    """nvidia/Nemotron-RLHF-GenRM-v1 — GenRM preference data with paired responses.

    The messages field contains pairs of conversations. We extract the prompt
    (first user turn) and use the ranking as ground truth metadata.
    """
    records = []
    for row in ds:
        all_messages = row.get("messages", [])
        if not all_messages or len(all_messages) < 1:
            continue
        first_conv = all_messages[0]
        prompt_msgs = []
        for m in first_conv:
            if m.get("role") == "user":
                prompt_msgs.append({"role": "user", "content": m.get("content", "")})
                break
        if not prompt_msgs:
            continue
        ranking = row.get("ranking", 0)
        gt = json.dumps({"ranking": ranking})
        records.append(Record(messages=prompt_msgs, ground_truth=gt, dataset="general-quality"))
    return records


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    hf_name: str
    split: str
    converter: Any
    output_name: str  # name for the output split/file


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        hf_name="BytedTsinghua-SIA/DAPO-Math-17k",
        split="train",
        converter=convert_dapo_math,
        output_name="dapo_math",
    ),
    DatasetSpec(
        hf_name="Skywork/Skywork-OR1-RL-Data",
        split="math",
        converter=convert_skywork_math,
        output_name="skywork_math",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-Math-Proofs-v1",
        split="lean",
        converter=convert_math_proofs,
        output_name="math_proofs",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
        split="train",
        converter=convert_agentic_tool_use,
        output_name="agentic_tool_use",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
        split="train",
        converter=convert_swe_pivot,
        output_name="swe_pivot",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-Identity-Following-v1",
        split="train",
        converter=convert_identity_following,
        output_name="identity_following",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-Instruction-Following-Calendar-v2",
        split="train",
        converter=convert_calendar,
        output_name="calendar",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
        split="train",
        converter=convert_multiturn_chat,
        output_name="multiturn_chat",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-ReasoningGym-v1",
        split="train",
        converter=convert_reasoning_gym,
        output_name="reasoning_gym",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-Safety-v1",
        split="train",
        converter=convert_safety,
        output_name="safety",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-agent-workplace_assistant",
        split="train",
        converter=convert_workplace_assistant,
        output_name="workplace_assistant",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-coding-competitive_coding",
        split="train",
        converter=convert_competitive_coding,
        output_name="competitive_coding",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-instruction_following-structured_outputs",
        split="train",
        converter=convert_structured_outputs,
        output_name="structured_outputs",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-instruction_following",
        split="train",
        converter=convert_instruction_following,
        output_name="instruction_following",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RL-knowledge-mcqa",
        split="train",
        converter=convert_mcqa,
        output_name="mcqa",
    ),
    DatasetSpec(
        hf_name="nvidia/Nemotron-RLHF-GenRM-v1",
        split="train",
        converter=convert_genrm,
        output_name="genrm",
    ),
]

DATASET_BY_HF_NAME = {spec.hf_name: spec for spec in DATASETS}


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert_dataset(spec: DatasetSpec, output_dir: Path) -> tuple[int, str]:
    """Download, convert, and write one dataset. Returns (num_records, output_path)."""
    log.info(f"Loading {spec.hf_name} (split={spec.split})...")
    ds = load_dataset(spec.hf_name, split=spec.split, trust_remote_code=True)

    log.info(f"  {len(ds)} rows loaded. Converting...")
    records = spec.converter(ds)

    out_path = output_dir / f"{spec.output_name}.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps({
                "messages": rec.messages,
                "ground_truth": rec.ground_truth,
                "dataset": rec.dataset,
            }) + "\n")

    log.info(f"  Wrote {len(records)} records to {out_path}")
    return len(records), str(out_path)


def push_to_hub(
    output_dir: Path,
    repo_id: str,
    private: bool = False,
    max_shard_size: str = "50MB",
):
    """Push all converted JSONL files to HuggingFace Hub."""
    from datasets import Dataset, DatasetDict

    def iter_jsonl(path: Path):
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    splits = {}
    for jsonl_path in sorted(output_dir.glob("*.jsonl")):
        split_name = jsonl_path.stem
        data = list(iter_jsonl(jsonl_path))
        if data:
            splits[split_name] = Dataset.from_list(data)
            log.info(f"  {split_name}: {len(data)} records")

    if not splits:
        log.error("No splits to push")
        return

    dataset_dict = DatasetDict(splits)
    log.info(f"Pushing to {repo_id}...")
    dataset_dict.push_to_hub(repo_id, private=private, max_shard_size=max_shard_size)
    log.info(f"Done! https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert individual HF datasets to open-instruct RLVR format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ingest/converted_individual"),
        help="Where to write converted JSONL files",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Only convert these datasets (by HF name). Default: all.",
    )
    parser.add_argument(
        "--push-to-hub", type=str, default=None,
        help="HuggingFace repo ID to push to",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--max-shard-size", type=str, default="50MB")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = DATASETS
    if args.only:
        specs = [DATASET_BY_HF_NAME[name] for name in args.only if name in DATASET_BY_HF_NAME]
        missing = set(args.only) - set(DATASET_BY_HF_NAME)
        if missing:
            log.warning(f"Unknown datasets: {missing}")

    summary = []
    for spec in specs:
        try:
            n, path = convert_dataset(spec, args.output_dir)
            summary.append((spec.hf_name, spec.output_name, n))
        except Exception as e:
            log.error(f"Failed to convert {spec.hf_name}: {e}")
            summary.append((spec.hf_name, spec.output_name, -1))

    log.info("\n=== Summary ===")
    for hf_name, output_name, n in summary:
        status = f"{n} records" if n >= 0 else "FAILED"
        log.info(f"  {hf_name:<60s} → {output_name:<25s} {status}")

    if args.push_to_hub:
        push_to_hub(
            args.output_dir, args.push_to_hub,
            private=args.private,
            max_shard_size=args.max_shard_size,
        )


if __name__ == "__main__":
    main()
