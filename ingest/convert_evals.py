"""
Convert evaluation benchmarks to open-instruct RLVR format for in-loop eval.

Supported evals (matching oe-eval config names):
  - gpqa:0shot_cot::hamish_zs_reasoning_deepseek
  - ifeval::hamish_zs_reasoning_deepseek
  - mmlu:cot::hamish_zs_reasoning_deepseek          (all 57 MMLU subjects)
  - codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek
  - mbppplus:0-shot-chat::tulu-thinker_deepseek
  - livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags
  - ifeval_ood::tulu-thinker-deepseek

Output format (open-instruct RLVR):
    messages:      list[dict] with role/content  (the prompt)
    ground_truth:  str                           (verifiable answer)
    dataset:       str                           (verifier routing key)

Usage:
    # Convert all evals
    python ingest/convert_evals.py --output-dir ingest/converted_evals

    # Convert specific evals
    python ingest/convert_evals.py --output-dir ingest/converted_evals --only gpqa ifeval

    # Convert + push to hub
    python ingest/convert_evals.py --output-dir ingest/converted_evals \
        --push-to-hub davidheineman/eval-openinstruct
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
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
    dataset: str


# ---------------------------------------------------------------------------
# GPQA (diamond) — string_matcher verifier
#
# Config: gpqa:0shot_cot::hamish_zs_reasoning_deepseek
# Source: Idavidrein/gpqa, config=gpqa_diamond, split=train
# ---------------------------------------------------------------------------

GPQA_DESCRIPTION = (
    'Answer the following multiple-choice question by giving the correct '
    'answer letter in parentheses. Provide CONCISE reasoning for the answer, '
    'and make sure to finish the response with "Therefore, the answer is '
    '(ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D).\n\n'
)
GPQA_FINAL = (
    '\n\nAnswer the above question and REMEMBER to finish your response with '
    'the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where '
    '(ANSWER_LETTER) is one of (A), (B), (C), (D).'
)
GPQA_SHUFFLE_SEED = 111


def _preprocess_gpqa(text: str) -> str:
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def convert_gpqa(rows: Iterator[dict]) -> list[Record]:
    records = []
    choice_labels = ["A", "B", "C", "D"]
    for idx, row in enumerate(rows):
        choices = [
            _preprocess_gpqa(row["Incorrect Answer 1"]),
            _preprocess_gpqa(row["Incorrect Answer 2"]),
            _preprocess_gpqa(row["Incorrect Answer 3"]),
            _preprocess_gpqa(row["Correct Answer"]),
        ]
        random.Random(GPQA_SHUFFLE_SEED + idx).shuffle(choices)
        correct_idx = choices.index(_preprocess_gpqa(row["Correct Answer"]))
        correct_letter = choice_labels[correct_idx]

        query = "Question: " + row["Question"] + "\nChoices:\n"
        query += "".join(f" ({lab}) {ch}\n" for lab, ch in zip(choice_labels, choices))

        prompt = GPQA_DESCRIPTION + query + GPQA_FINAL
        messages = [{"role": "user", "content": prompt}]
        records.append(Record(messages=messages, ground_truth=correct_letter, dataset="string_matcher"))
    return records


# ---------------------------------------------------------------------------
# IFEval — ifeval verifier
#
# Config: ifeval::hamish_zs_reasoning_deepseek
# Source: HuggingFaceH4/ifeval, split=train
# ---------------------------------------------------------------------------

def convert_ifeval(rows: Iterator[dict]) -> list[Record]:
    records = []
    for row in rows:
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        messages = [{"role": "user", "content": prompt}]
        instruction_ids = row.get("instruction_id_list")
        kwargs_list = row.get("kwargs")
        if not instruction_ids or not kwargs_list:
            continue
        gt = repr([{
            "instruction_id": instruction_ids,
            "kwargs": kwargs_list,
        }])
        records.append(Record(messages=messages, ground_truth=gt, dataset="ifeval"))
    return records


# ---------------------------------------------------------------------------
# MMLU (57 subjects, CoT) — string_matcher verifier
#
# Config: mmlu:cot::hamish_zs_reasoning_deepseek
# Source: cais/mmlu, per-subject configs, split=test
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


def convert_mmlu(subjects: list[str] | None = None) -> list[Record]:
    from datasets import load_dataset

    if subjects is None:
        subjects = MMLU_SUBJECTS
    choice_labels = ["A", "B", "C", "D"]
    records = []
    for sub in subjects:
        description = (
            f"The following are multiple choice questions about "
            f"{sub.replace('_', ' ')}. Summarize your reasoning concisely, "
            f"then conclude with 'Therefore, the answer is: X' where X is "
            f"one of A, B, C, or D.\n\n"
        )
        try:
            ds = load_dataset("cais/mmlu", sub, split="test")
        except Exception as e:
            log.warning(f"  Failed to load MMLU subject '{sub}': {e}")
            continue
        for row in ds:
            question = row["question"]
            choices = row["choices"]
            answer_idx = row["answer"]
            correct_letter = choice_labels[answer_idx]

            query = f"Question: {question}\n"
            for lab, ch in zip(choice_labels, choices):
                query += f" {lab}. {ch}\n"

            prompt = description + query
            messages = [{"role": "user", "content": prompt}]
            records.append(Record(
                messages=messages,
                ground_truth=correct_letter,
                dataset="string_matcher",
            ))
        log.info(f"  MMLU {sub}: {len(ds)} rows")
    return records


# ---------------------------------------------------------------------------
# HumanEval+ — code verifier
#
# Config: codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek
# Source: evalplus/humanevalplus, split=test
#
# NOTE: The code verifier sends ground_truth to an external code execution
# API. For function-level tests (HumanEval/MBPP), the ground_truth stores
# the assertion test code + entry point as JSON. The code execution API
# needs to support this format (run: model_code + test_assertions).
# ---------------------------------------------------------------------------

HUMANEVAL_TEMPLATE = (
    "Complete the following function:\n{prompt}\n"
    "Provide CONCISE reasoning on how to arrive at the answer, and make sure "
    "to finish the response with the following, where (CODE) is the code for "
    "the complete function:\n\n"
    "Here is the completed function:\n\n```python\n(CODE)\n```"
)


def convert_humanevalplus(rows: Iterator[dict]) -> list[Record]:
    records = []
    for row in rows:
        func_prompt = row.get("prompt", "")
        if not func_prompt:
            continue
        user_prompt = HUMANEVAL_TEMPLATE.format(prompt=func_prompt)
        messages = [{"role": "user", "content": user_prompt}]

        test_code = row.get("test", "")
        entry_point = row.get("entry_point", "")
        full_test = test_code
        if entry_point and f"check({entry_point})" not in test_code:
            full_test += f"\ncheck({entry_point})"

        gt = json.dumps([{
            "input": "",
            "output": "",
            "test_code": full_test,
            "entry_point": entry_point,
            "setup_code": func_prompt,
        }])
        records.append(Record(messages=messages, ground_truth=gt, dataset="code"))
    return records


# ---------------------------------------------------------------------------
# MBPP+ — code verifier
#
# Config: mbppplus:0-shot-chat::tulu-thinker_deepseek
# Source: evalplus/mbppplus, split=test
# ---------------------------------------------------------------------------

MBPP_TEMPLATE = (
    "{code_prompt}\n\n"
    "Provide CONCISE reasoning on how to arrive at the answer, and make sure "
    "to finish the response with the following:\n\n"
    "Here is the completed function:\n\n```python\n(CODE)\n```\n"
    "where (CODE) is the code for the complete function."
)


def convert_mbppplus(rows: Iterator[dict]) -> list[Record]:
    records = []
    for row in rows:
        text = row.get("prompt", "")
        code = row.get("code", "")
        if not text:
            continue
        code_prompt = text + code.split(":")[0] + ":"
        user_prompt = MBPP_TEMPLATE.format(code_prompt=code_prompt)
        messages = [{"role": "user", "content": user_prompt}]

        test_code = row.get("test", "")
        entry_point = code.split("(")[0].replace("def ", "").strip() if "def " in code else ""
        gt = json.dumps([{
            "input": "",
            "output": "",
            "test_code": test_code,
            "entry_point": entry_point,
            "setup_code": code_prompt,
        }])
        records.append(Record(messages=messages, ground_truth=gt, dataset="code"))
    return records


# ---------------------------------------------------------------------------
# LiveCodeBench — code verifier
#
# Config: livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags
# Source: livecodebench/code_generation_lite, version_tag=release_v3, split=test
# ---------------------------------------------------------------------------

LCB_TEMPLATE = (
    "### Question:\n{problem_statement}\n\n"
    "### Format:\n"
    "Provide CONCISE reasoning on how to arrive at the answer.\n"
    "{format_instruction}\n\n"
    "### Answer: (use the provided format with backticks)\n\n"
)
LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)


def _decode_private_tests(encoded: str | None) -> list[dict]:
    """Decode private test cases from base64-encoded, zlib-compressed JSON."""
    if not encoded:
        return []
    try:
        import base64
        import zlib
        decoded = json.loads(zlib.decompress(base64.b64decode(encoded)))
        return decoded if isinstance(decoded, list) else []
    except Exception:
        return []


def convert_livecodebench(rows: Iterator[dict]) -> list[Record]:
    records = []
    for row in rows:
        problem_statement = row.get("question_content", "")
        if not problem_statement:
            continue

        starter_code = row.get("starter_code", "").strip()
        if starter_code:
            format_instruction = (
                "You will use the following starter code to write the solution "
                "to the problem and enclose your code within delimiters."
                f"\n```python\n{starter_code}\n```"
            )
        else:
            format_instruction = (
                "Read the inputs from stdin solve the problem and write the "
                "answer to stdout (do not directly test on the sample inputs). "
                "Enclose your code within delimiters as follows. Ensure that "
                "when the python program runs, it reads the inputs, runs the "
                "algorithm and writes output to STDOUT.\n"
                "```python\n# YOUR CODE HERE\n```"
            )

        user_prompt = LCB_TEMPLATE.format(
            problem_statement=problem_statement,
            format_instruction=format_instruction,
        )
        messages = [
            {"role": "system", "content": LCB_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        public_tests = json.loads(row["public_test_cases"]) if row.get("public_test_cases") else []
        private_tests = _decode_private_tests(row.get("private_test_cases"))
        all_tests = public_tests + private_tests

        test_cases = [{"input": t["input"], "output": t["output"]} for t in all_tests]
        if not test_cases:
            continue

        records.append(Record(
            messages=messages,
            ground_truth=json.dumps(test_cases),
            dataset="code",
        ))
    return records


# ---------------------------------------------------------------------------
# IFEval OOD (IFBench) — ifeval verifier
#
# Config: ifeval_ood::tulu-thinker-deepseek
# Source: allenai/IFBench_test2, split=train
# ---------------------------------------------------------------------------

def convert_ifeval_ood(rows: Iterator[dict]) -> list[Record]:
    records = []
    for row in rows:
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        messages = [{"role": "user", "content": prompt}]
        instruction_ids = row.get("instruction_id_list")
        kwargs_list = row.get("kwargs")
        if not instruction_ids or not kwargs_list:
            continue
        gt = repr([{
            "instruction_id": instruction_ids,
            "kwargs": kwargs_list,
        }])
        records.append(Record(messages=messages, ground_truth=gt, dataset="ifeval"))
    return records


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass
class EvalSpec:
    name: str
    hf_name: str
    hf_config: str | None
    split: str
    converter: Any
    output_name: str
    hf_kwargs: dict | None = None


def _make_eval_specs() -> list[EvalSpec]:
    return [
        EvalSpec(
            name="gpqa:0shot_cot::hamish_zs_reasoning_deepseek",
            hf_name="Idavidrein/gpqa",
            hf_config="gpqa_diamond",
            split="train",
            converter=convert_gpqa,
            output_name="gpqa",
        ),
        EvalSpec(
            name="ifeval::hamish_zs_reasoning_deepseek",
            hf_name="HuggingFaceH4/ifeval",
            hf_config=None,
            split="train",
            converter=convert_ifeval,
            output_name="ifeval",
        ),
        EvalSpec(
            name="mmlu:cot::hamish_zs_reasoning_deepseek",
            hf_name="cais/mmlu",
            hf_config=None,  # handled specially (57 subjects)
            split="test",
            converter=None,  # handled specially
            output_name="mmlu",
        ),
        EvalSpec(
            name="codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek",
            hf_name="evalplus/humanevalplus",
            hf_config=None,
            split="test",
            converter=convert_humanevalplus,
            output_name="humanevalplus",
        ),
        EvalSpec(
            name="mbppplus:0-shot-chat::tulu-thinker_deepseek",
            hf_name="evalplus/mbppplus",
            hf_config=None,
            split="test",
            converter=convert_mbppplus,
            output_name="mbppplus",
        ),
        EvalSpec(
            name="livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags",
            hf_name="livecodebench/code_generation_lite",
            hf_config=None,
            split="test",
            converter=convert_livecodebench,
            output_name="livecodebench",
            hf_kwargs={"_lcb_jsonl": True},  # custom loading, see convert_eval()
        ),
        EvalSpec(
            name="ifeval_ood::tulu-thinker-deepseek",
            hf_name="allenai/IFBench_test2",
            hf_config=None,
            split="train",
            converter=convert_ifeval_ood,
            output_name="ifeval_ood",
        ),
    ]


EVAL_SPECS = _make_eval_specs()
EVAL_BY_NAME = {s.output_name: s for s in EVAL_SPECS}


# ---------------------------------------------------------------------------
# Conversion pipeline
# ---------------------------------------------------------------------------

def load_hf(spec: EvalSpec) -> Iterator[dict]:
    from datasets import load_dataset
    kwargs = {}
    if spec.hf_config:
        kwargs["name"] = spec.hf_config
    if spec.hf_kwargs:
        kwargs.update(spec.hf_kwargs)
    try:
        ds = load_dataset(spec.hf_name, split=spec.split, **kwargs)
    except RuntimeError as e:
        if "Dataset scripts" in str(e):
            ds = load_dataset(spec.hf_name, split=spec.split, trust_remote_code=True, **kwargs)
        else:
            raise
    log.info(f"  Loaded {len(ds)} rows from {spec.hf_name}")
    return iter(ds)


def _load_lcb_jsonl() -> Iterator[dict]:
    """Load LiveCodeBench data directly from JSONL files (release_v3 = test + test2 + test3)."""
    from huggingface_hub import hf_hub_download
    total = 0
    for fname in ["test.jsonl", "test2.jsonl", "test3.jsonl"]:
        path = hf_hub_download("livecodebench/code_generation_lite", fname, repo_type="dataset")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
                    total += 1
    log.info(f"  Loaded {total} rows from livecodebench/code_generation_lite (release_v3)")


def convert_eval(spec: EvalSpec, output_dir: Path) -> tuple[int, str]:
    log.info(f"Converting {spec.name}...")

    if spec.output_name == "mmlu":
        records = convert_mmlu()
    elif spec.hf_kwargs and spec.hf_kwargs.get("_lcb_jsonl"):
        rows = _load_lcb_jsonl()
        records = spec.converter(rows)
    else:
        rows = load_hf(spec)
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
        description="Convert eval benchmarks to open-instruct RLVR format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ingest/converted_evals"),
        help="Where to write converted JSONL files",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Only convert these evals (by output name: gpqa, ifeval, mmlu, "
             "humanevalplus, mbppplus, livecodebench, ifeval_ood). Default: all.",
    )
    parser.add_argument(
        "--push-to-hub", type=str, default=None,
        help="HuggingFace repo ID to push to",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--max-shard-size", type=str, default="50MB")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = EVAL_SPECS
    if args.only:
        specs = [EVAL_BY_NAME[name] for name in args.only if name in EVAL_BY_NAME]
        missing = set(args.only) - set(EVAL_BY_NAME)
        if missing:
            log.warning(f"Unknown evals (available: {list(EVAL_BY_NAME)}): {missing}")

    summary = []
    for spec in specs:
        try:
            n, path = convert_eval(spec, args.output_dir)
            summary.append((spec.name, spec.output_name, n))
        except Exception as e:
            log.error(f"Failed to convert {spec.name}: {e}", exc_info=True)
            summary.append((spec.name, spec.output_name, -1))

    log.info("\n=== Summary ===")
    for name, output_name, n in summary:
        status = f"{n} records" if n >= 0 else "FAILED"
        log.info(f"  {name:<65s} -> {output_name:<20s} {status}")

    if args.push_to_hub:
        push_to_hub(
            args.output_dir, args.push_to_hub,
            private=args.private,
            max_shard_size=args.max_shard_size,
        )


if __name__ == "__main__":
    main()
