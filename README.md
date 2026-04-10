data mixing for RL.

### notes

🚦 **Current status:** I have a nice default in `configs/examples/gsm_math_qwen_2b.yaml`. Need to (1) setup the Nemotron dataset to get mix domains (we need to consume all SOURCES, not just the final mix they provided), (2) set small / large experiment scales (mixing used 1 node for 3 hours. Our math example is [1 node for 6 hours](https://beaker.org/orgs/ai2/workspaces/adaptability/work/01KNQ2MN48CNF3279F2DYAZTE2)), (3) setup eval infra. (in-loop should be fine for now)

**consuming NEMO**: Need to setup the verifiers from [`NVIDIA-NeMo/RL`](https://github.com/NVIDIA-NeMo/RL) and [`NVIDIA-NeMo/Gym`](https://github.com/NVIDIA-NeMo/Gym)

| split | num. prompts | open-instruct verifier | nemo gym verifier | notes | libraries |
|---|---|---|---|---|---|
| `dapo_math` | 1.79M | `nemo_dapo_math` | `math_with_judge` — symbolic math equivalence via `math-verify` library (LaTeX/boxed extraction + SymPy comparison). LLM judge fallback if symbolic fails. | open-instruct `MathVerifier` already does this. Uses boxed/Minerva extraction + `is_equiv`. Nearly identical logic. | `math-verify` (HF), or open-instruct's built-in `math_utils.py` |
| `skywork_math` | 105K | `nemo_skywork_math` | Same as above | Same — already works | Same |
| `math_proofs` | 921K | `nemo_math_proofs` | Same as above | Same — already works | Same |
| `instruction_following` | 46K | `nemo_instruction_following` | `instruction_following` — looks up each `instruction_id` in Google's `verifiable-instructions` registry, calls `check_following()` per constraint. Binary or fractional reward. | open-instruct `IFEvalVerifier` does exactly this (same IFEval constraint functions). Already works. | `verifiable-instructions` (Google), `nltk`, `spacy` |
| `competitive_coding` | 16K | `nemo_competitive_coding` | `code_gen` — extracts code from response, runs against unit tests via Ray remote sandboxed execution. Reward = 1.0 if all pass, else 0.0. | open-instruct `CodeVerifier` calls an external code execution API with the same test format. Needs a running code sandbox API. | `ray`, LiveCodeBench `lcb_integration`, or open-instruct's external code API |
| `mcqa` | 617K | `nemo_mcqa` | `mcqa` — regex extraction of `\boxed{X}` or `Answer: X`, exact letter match against gold. | open-instruct `StringMatcherVerifier` does normalized `<answer>` tag extraction + match. Slightly different extraction format but same idea. | stdlib `re` only |
| `reasoning_gym` | 14K | `nemo_reasoning_gym` | `reasoning_gym` — extracts `<answer>` tags, looks up per-task scoring function via `reasoning_gym.get_score_answer_fn(task_name)`. 100+ task types with custom scorers. | open-instruct `PuzzleMatcherVerifier` does normalized exact match. Simpler but covers most cases. For full fidelity, use the `reasoning-gym` library directly. | [`reasoning-gym`](https://github.com/open-thought/reasoning-gym) |
| `calendar` | 10K | `nemo_calendar` | `calendar` — parses assistant's JSON calendar output, checks event count, no time conflicts, all constraints satisfied (before/after/at/between, duration, min/max time). Fully deterministic. | ~100 lines of pure Python. Parse JSON, check constraints per event. No open-instruct equivalent exists. Could add as custom verifier or extract from `Gym/resources_servers/calendar/utils.py`. | stdlib only (`json`, `re`) |
| `structured_outputs` | 9K | `nemo_structured_outputs` | `structured_outputs` — parses response as JSON/YAML/XML, validates against OpenAPI schema. Reward = 1.0 if valid, 0.0 if not. | ~50 lines. Parse response, validate against schema string. No open-instruct equivalent. | `openapi-schema-validator`, `pyyaml`, `xmltodict` |
| `workplace_assistant` | 1.3K | `nemo_workplace_assistant` | `workplace_assistant` — executes predicted AND ground-truth tool calls against an in-memory simulated environment (email, calendar, analytics, PM, CRM). Compares resulting DataFrame states. Reward = 1.0 if states match. | ~200 lines + the tool environment code. Needs the full workplace tool simulation from `Gym/resources_servers/workplace_assistant/`. No open-instruct equivalent. | `pandas` (for DataFrame state comparison) |
| `agentic_tool_use` | 97K | `nemo_agentic_tool_use` | `single_step_tool_use_with_argument_comparison` — extracts predicted tool call from response, compares function name + arguments against `expected_action` using `ToolCallComparator`. For message-type actions, any chat response = reward 1.0. | ~100 lines. Parse response for tool calls, compare name + args against expected. No open-instruct equivalent. Could extract `ToolCallComparator` from Gym. | stdlib only |
| `swe_pivot` | 51K | `nemo_swe_pivot` | Same as `agentic_tool_use` — `ToolCallComparator` on `expected_action` | Same implementation | stdlib only |
| `safety` | 89K | `nemo_safety` | `xstest` / `jailbreak_detection` / `over_refusal_detection` — prefix-based refusal detection (checks ~20 known refusal prefixes like "I'm sorry", "I cannot"). Optional LLM judge for more nuanced classification. | Without judge: ~30 lines of prefix matching. With judge: needs LLM API. open-instruct `LMJudgeVerifier("safety")` uses an LLM judge (different prompt but similar intent). | stdlib for prefix matching; LLM API for judge mode |
| `identity_following` | 22K | `nemo_identity_following` | No dedicated NeMo Gym verifier — uses GenRM or LLM judge with the `principle` as rubric | Needs LLM judge. open-instruct `LMJudgeVerifier("quality")` is the closest match. | LLM API endpoint |
| `multiturn_chat` | 2K | `nemo_multiturn_chat` | No dedicated verifier — uses LLM judge with `rubric` field | Needs LLM judge. open-instruct `LMJudgeVerifier("quality")` is the closest match. | LLM API endpoint |
| `genrm` | 300K | `nemo_genrm` | `genrm_compare` — buffers rollouts per prompt, generates pairwise comparisons, sends to a dedicated Generative Reward Model, parses scores/rankings, aggregates. | Needs a GenRM model endpoint. open-instruct `LMJudgeVerifier("quality")` is a simpler approximation. Full fidelity requires a GenRM model. | GenRM model endpoint (**required**) |

In `open-instruct/` the datasets are called `nemo_{split_name}` for verification purposes.

**small vs. large scale**: Current plan is two setups:
- **param and tokens** - small scale runs are Qwen 3 1.7B at 128K episodes. large scale run is Qwen 3 14B at 1.5M epsiodes (so ~100K prompts in the large-scale run)
- **tokens only** - small scale runs are Qwen 3 14B at 16K episodes (so ~1K prompts), large scale is the same.
- **real config** - in practice, we need a 1,000x to 10,000x compute multiplier from 1 small-scale run to 1 large-scale run. Not 100x.

**evals**: Get GPQA, MMLU Pro, HumanEval, MBPP, LCB, IF Bench as in-loop eval. There's AIME one, and some RL Zero val sets folks have used. Should be easy to pipe into HF datasets

- [ ] `gpqa:0shot_cot::hamish_zs_reasoning_deepseek`
- [ ] `ifeval::hamish_zs_reasoning_deepseek`
- [ ] `mmlu:cot::hamish_zs_reasoning_deepseek`
- [ ] `codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek`
- [ ] `mbppplus:0-shot-chat::tulu-thinker_deepseek`
- [ ] `livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags`
- [ ] `ifeval_ood::tulu-thinker-deepseek`
- [ ] (mmlu pro would be nice, but no Olmo 3 instruct eval config)

I have a V1 here: https://huggingface.co/datasets/davidheineman/eval-openinstruct. I still need to test this (either parity, or just that the verifiers work). Seems like IF Bench and code eval may not work right now.

```sh
python ingest/convert_evals.py \
    --output-dir ingest/converted_evals \
    --push-to-hub davidheineman/eval-openinstruct
```

**chat template**: Qwen chat template adapted to thinking models (need to implement):

```python
"qwen_thinker": (
    "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
    "{% if not has_system %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant. You first think about the reasoning process in the <think> and </think> tags, and then provide your answer in the <answer> and </answer> tags.<|im_end|>\n' }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n<think>' }}"
    "{% endif %}"
    "{% endfor %}"
),
```

### setup

```bash
git clone https://github.com/davidheineman/rl-mixing
cd rl-mixing

# mixing 💛 open-instruct
git clone https://github.com/allenai/open-instruct.git
cd open-instruct
git checkout -b dhei/rl-mixing origin/main
cd ..

# install
uv sync
```

### usage

```bash
# experiments are defined through configs
python src/launch.py configs/examples/gsm_math_qwen_2b.yaml --dry-run
python src/launch.py configs/examples/gsm_math_qwen_2b.yaml -f # run + follow

# --sweep launches a sweep
python src/launch.py --sweep src/sweeps/example_gsm_if.py --dry-run
```

#### python api

```python
from experiments import Experiment, Mix, MixComponent, EvalMix, EvalMixComponent

# example: GSM / IF mixture sweep
for gsm_frac in [0.25, 0.5, 0.75]:
    exp = Experiment(
        name=f"gsm{gsm_frac:.0%}-if{1-gsm_frac:.0%}",
        model="allenai/Llama-3.1-Tulu-3-8B-DPO",
        mix=Mix([
            MixComponent("allenai/RLVR-GSM-MATH", gsm_frac),
            MixComponent("allenai/RLVR-IF-Mixed-Constraints", 1.0 - gsm_frac),
        ]),
        eval_mix=EvalMix([
            EvalMixComponent("allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", num_samples=16),
        ]),
    )
    exp.launch(dry_run=True)
```

### ingest data

Ingest [nvidia/Nemotron-RL-Super-Training-Blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends) -> [davidheineman/nemotron-super-stage-1-unmixed-openinstruct](https://huggingface.co/datasets/davidheineman/nemotron-super-stage-1-unmixed-openinstruct)

```sh
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'nvidia/Nemotron-RL-Super-Training-Blends',
    repo_type='dataset',
    local_dir='ingest/raw',
)
print('Download complete.')
"

python ingest/raw/fill_placeholders.py \
    --input-dir ingest/raw \
    --output-dir ingest/filled

python ingest/convert_nemotron.py \
    --input-dir ingest/filled \
    --output-dir ingest/converted \
    --splits rlvr1 rlvr2 rlvr3 \
    --high-fidelity-only \
    --push-to-hub davidheineman/nemotron-rlvr-openinstruct \
    --max-shard-size 50MB
```