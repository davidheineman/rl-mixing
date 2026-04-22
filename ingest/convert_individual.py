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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

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
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _load_hf_dataset(hf_name: str, split: str) -> Iterator[dict]:
    """Load a HuggingFace dataset, falling back to raw JSONL download when
    the datasets library cannot handle the schema (e.g. heterogeneous JSON
    types across rows)."""
    from datasets import load_dataset
    try:
        ds = load_dataset(hf_name, split=split)
        log.info(f"  {len(ds)} rows loaded via datasets library.")
        yield from ds
    except Exception as e:
        log.warning(f"  load_dataset failed ({e}), falling back to raw JSONL download...")
        yield from _load_raw_jsonl(hf_name, split)


def _load_raw_jsonl(hf_name: str, split: str) -> Iterator[dict]:
    """Download raw JSONL from the HF repo and iterate rows."""
    from huggingface_hub import hf_hub_download, list_repo_tree

    candidates = [f"{split}.jsonl", f"data/{split}.jsonl"]
    repo_files = {e.path for e in list_repo_tree(hf_name, repo_type="dataset")}
    # Also check inside data/ subfolder
    try:
        repo_files |= {e.path for e in list_repo_tree(hf_name, repo_type="dataset", path_in_repo="data")}
    except Exception:
        pass

    jsonl_path = None
    for c in candidates:
        if c in repo_files:
            jsonl_path = c
            break

    if jsonl_path is None:
        raise FileNotFoundError(
            f"Could not find JSONL file for split '{split}' in {hf_name}. "
            f"Available files: {sorted(repo_files)}"
        )

    local_path = hf_hub_download(hf_name, jsonl_path, repo_type="dataset")
    count = 0
    with open(local_path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
                count += 1
    log.info(f"  {count} rows loaded via raw JSONL download ({jsonl_path}).")


# ---------------------------------------------------------------------------
# Per-dataset converters
#
# Each returns a list of Record. An iterator of row dicts is passed in.
# ---------------------------------------------------------------------------

def _msgs_from_rcp(row: dict) -> list[dict[str, str]] | None:
    """Extract messages from responses_create_params.input (Nemotron-style).
    Handles content being either a string or a list of content parts."""
    rcp = row.get("responses_create_params")
    if not rcp:
        return None
    msgs = rcp.get("input")
    if not msgs or not isinstance(msgs, list):
        return None
    result = []
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m:
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            # Multi-part content (e.g. tool use): join text parts
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", str(part)))
                else:
                    parts.append(str(part))
            content = "\n".join(parts)
        result.append({"role": m["role"], "content": str(content)})
    return result if result else None


def convert_dapo_math(rows: Iterator[dict]) -> list[Record]:
    """BytedTsinghua-SIA/DAPO-Math-17k — math problems with ground truth answers."""
    records = []
    for row in rows:
        prompt = row.get("prompt", [])
        if not prompt:
            continue
        messages = [{"role": m["role"], "content": m["content"]} for m in prompt]
        gt = row.get("reward_model", {}).get("ground_truth", "")
        if gt is None:
            continue
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="math"))
    return records


def convert_skywork_math(rows: Iterator[dict]) -> list[Record]:
    """Skywork/Skywork-OR1-RL-Data (math split) — same schema as DAPO."""
    records = []
    for row in rows:
        prompt = row.get("prompt", [])
        if not prompt:
            continue
        messages = [{"role": m["role"], "content": m["content"]} for m in prompt]
        gt = row.get("reward_model", {}).get("ground_truth", "")
        if gt is None:
            continue
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="math"))
    return records


def convert_math_proofs(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-Math-Proofs-v1 — Lean 4 math proofs with formal statements.

    Ground truth is a JSON dict with ``header`` (Lean imports) and
    ``formal_statement`` (the theorem to prove).  Routes to the ``lean``
    verifier which compiles the proof in a Lean 4 sandbox.
    """
    records = []
    for row in rows:
        messages = row.get("messages", [])
        if not messages:
            continue
        prompt_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") == "user"]
        if not prompt_msgs:
            prompt_msgs = [{"role": "user", "content": row.get("problem", "")}]
        formal_statement = row.get("formal_statement", "")
        if not formal_statement:
            continue
        gt = json.dumps({
            "header": row.get("lean_header", ""),
            "formal_statement": formal_statement,
        })
        records.append(Record(messages=prompt_msgs, ground_truth=gt, dataset="lean"))
    return records


def convert_agentic_tool_use(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1 — multi-turn
    tool-use trajectories with expected_action."""
    records = []
    for row in rows:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        expected = row.get("expected_action")
        gt = json.dumps(expected) if expected else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="general-quality"))
    return records


def convert_swe_pivot(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1 — SWE agent trajectories."""
    records = []
    for row in rows:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        expected = row.get("expected_action")
        gt = json.dumps(expected) if expected else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="general-quality"))
    return records


def convert_identity_following(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-Identity-Following-v1 — identity/principle following."""
    records = []
    for row in rows:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        gt = row.get("principle", "")
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="general-quality"))
    return records


def convert_calendar(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-Instruction-Following-Calendar-v2 — calendar-based IF.

    Ground truth is a calendar state, not IFEval constraints, so this
    routes to the LM judge verifier.
    """
    records = []
    for row in rows:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        cal_state = row.get("exp_cal_state")
        gt = json.dumps(cal_state) if cal_state else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="general-quality"))
    return records


def convert_multiturn_chat(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1 — multi-turn
    chat with rubric-based evaluation."""
    records = []
    for row in rows:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        rubric = row.get("rubric")
        gt = json.dumps(rubric) if rubric else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="general-quality"))
    return records


def convert_reasoning_gym(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-ReasoningGym-v1 — puzzle/reasoning tasks with
    verifiable answers."""
    records = []
    for row in rows:
        messages = _msgs_from_rcp(row)
        if not messages:
            messages = [{"role": "user", "content": row.get("question", "")}]
        answer = row.get("answer", "")
        if not answer:
            continue
        records.append(Record(messages=messages, ground_truth=str(answer), dataset="puzzle"))
    return records


def convert_safety(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-Safety-v1 — safety preference data."""
    records = []
    for row in rows:
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        messages = [{"role": "user", "content": prompt}]
        gt = row.get("principle", "")
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="general-safety"))
    return records


def convert_workplace_assistant(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-agent-workplace_assistant — tool-calling workplace tasks."""
    records = []
    for row in rows:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        gt_calls = row.get("ground_truth")
        gt = json.dumps(gt_calls) if gt_calls else ""
        records.append(Record(messages=messages, ground_truth=gt, dataset="general-quality"))
    return records


def convert_competitive_coding(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-coding-competitive_coding — competitive programming
    with unit test verification.

    Tests are stdin/stdout format so this routes to ``code_stdio``
    (which calls the ``/test_program_stdio`` endpoint).
    """
    records = []
    for row in rows:
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
            dataset="code_stdio",
        ))
    return records


def convert_structured_outputs(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-instruction_following-structured_outputs — structured
    output following with schema validation.

    Ground truth is a JSON schema string, not IFEval constraints, so this
    routes to the LM judge verifier.
    """
    records = []
    for row in rows:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        schema = row.get("schema_str", "")
        gt = schema if schema else ""
        records.append(Record(messages=messages, ground_truth=str(gt), dataset="general-quality"))
    return records


def convert_instruction_following(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-instruction_following — IFEval-style instruction following.

    open-instruct's IFEvalVerifier uses ast.literal_eval() to parse the
    ground truth, so we must emit Python-syntax (None/True/False) not
    JSON-syntax (null/true/false). We use repr() for this.
    """
    records = []
    for row in rows:
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        messages = [{"role": "user", "content": prompt}]
        instruction_ids = row.get("instruction_id_list")
        kwargs_list = row.get("kwargs")
        if instruction_ids and kwargs_list:
            gt = repr([{
                "instruction_id": instruction_ids,
                "kwargs": kwargs_list,
            }])
        else:
            continue
        records.append(Record(messages=messages, ground_truth=gt, dataset="ifeval"))
    return records


def convert_mcqa(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RL-knowledge-mcqa — multiple choice QA with letter answers."""
    records = []
    for row in rows:
        messages = _msgs_from_rcp(row)
        if not messages:
            continue
        answer = row.get("expected_answer", "")
        if not answer:
            continue
        records.append(Record(messages=messages, ground_truth=str(answer), dataset="string_matcher"))
    return records


def convert_genrm(rows: Iterator[dict]) -> list[Record]:
    """nvidia/Nemotron-RLHF-GenRM-v1 — GenRM preference data with paired responses.

    The messages field contains nested conversations. We extract the prompt
    (first user turn) and use the ranking as ground truth metadata.
    """
    records = []
    for row in rows:
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
    output_name: str


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
        hf_name="nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1",
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
    rows = _load_hf_dataset(spec.hf_name, spec.split)

    log.info(f"  Converting...")
    records = spec.converter(rows)

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
            log.error(f"Failed to convert {spec.hf_name}: {e}", exc_info=True)
            summary.append((spec.hf_name, spec.output_name, -1))

    log.info("\n=== Summary ===")
    for hf_name, output_name, n in summary:
        status = f"{n} records" if n >= 0 else "FAILED"
        log.info(f"  {hf_name:<60s} -> {output_name:<25s} {status}")

    if args.push_to_hub:
        push_to_hub(
            args.output_dir, args.push_to_hub,
            private=args.private,
            max_shard_size=args.max_shard_size,
        )


if __name__ == "__main__":
    main()
