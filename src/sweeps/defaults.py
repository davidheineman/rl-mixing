from experiments import (
    Experiment,
    Mix,
    EvalMix,
    EvalMixComponent,
    TrainingConfig,
    InfraConfig,
    BeakerConfig,
)


DATASET = "davidheineman/nemotron-super-stage-1-unmixed-openinstruct"

EVAL_MIX = EvalMix([
    EvalMixComponent("mnoukhov/aime_2025_openinstruct", num_samples=1.0, split="train"),
    EvalMixComponent("mnoukhov/brumo_2025_openinstruct", num_samples=1.0, split="train"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=1.0, split="gpqa"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=1.0, split="ifeval"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=1.0, split="mmlu"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=1.0, split="humanevalplus"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=1.0, split="mbppplus"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=1.0, split="livecodebench"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=1.0, split="ifeval_ood"),
])

# rl-mixing repo root (for locating scripts/)
_REPO_URL = "https://github.com/davidheineman/rl-mixing.git"

# Setup: clone open-instruct, install deps, apply patches (multi-remap + lean verifier),
# start code execution API and Lean sandbox.
_SETUP_OI = (
    'git clone --depth 1 -b dhei/rl-mixing https://github.com/allenai/open-instruct.git /tmp/oi'
    ' && cp -r /tmp/oi/open_instruct/* open_instruct/'
    ' && cp -r /tmp/oi/configs/beaker_configs/* configs/beaker_configs/ 2>/dev/null || true'
    ' && mkdir -p oe-eval-internal'
    ' && uv pip install openapi-schema-validator pyyaml xmltodict spacy langdetect emoji syllapy nltk httpx -q 2>/dev/null || true'
    ' && python -m nltk.downloader stopwords -q 2>/dev/null || true'
)

_SETUP_PATCHES = (
    'git clone --depth 1 ' + _REPO_URL + ' /tmp/rl-mixing'
    ' && python /tmp/rl-mixing/scripts/patch_open_instruct.py'
)

_SETUP_CODE_API = 'source configs/beaker_configs/code_api_setup.sh'

_SETUP_LEAN_SANDBOX = 'source /tmp/rl-mixing/scripts/lean_sandbox_setup.sh'


def base_experiment(name: str, mix: Mix) -> Experiment:
    """ Default small-scale training config """
    return Experiment(
        name=name,
        model="Qwen/Qwen3-1.7B-Base",
        chat_template="qwen_thinker",
        mix=mix,
        eval_mix=EVAL_MIX,
        training=TrainingConfig(
            learning_rate=1e-6,
            beta=0.0,
            total_episodes=65536,
            temperature=1.0,
            response_length=6144,
            max_prompt_token_length=2048,
            pack_length=8192,
            num_samples_per_prompt_rollout=16,
            num_unique_prompts_rollout=8,
            async_steps=1,
            per_device_train_batch_size=1,
            kl_estimator=2,
            lr_scheduler_type="constant",
            deepspeed_stage=3,
            gradient_checkpointing=True,
            seed=1,
            eval_at_end=True,
            save_freq=100,
            non_stop_penalty=False,
            apply_verifiable_reward=True,
        ),
        infra=InfraConfig(
            num_learners_per_node=[4],
            vllm_num_engines=4,
            vllm_tensor_parallel_size=1,
        ),
        beaker=BeakerConfig(
            cluster="ai2/jupiter",
            workspace="ai2/adaptability",
            budget="ai2/oe-adapt",
            num_nodes=1,
            gpus=8,
            priority="urgent",
            preemptible=True,
            image="michaeln/open_instruct",
            mount_docker_socket=True,
            setup_commands=[
                _SETUP_OI,
                _SETUP_PATCHES,
                _SETUP_CODE_API,
                _SETUP_LEAN_SANDBOX,
            ],
            env={
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
                "BEAKER_ALLOW_SUBCONTAINERS": "1",
            },
        ),
        extra_args=[
            "--inflight_updates",
            "--vllm_enable_prefix_caching",
            "--clip_higher", "0.272",
            "--mask_truncated_completions", "False",
            "--load_ref_policy", "True",
            "--num_mini_batches", "1",
            "--stop_strings", "</answer>",
            "--code_api_url", "$CODE_API_URL/test_program",
            "--code_max_execution_time", "6.0",
            "--llm_judge_model", "gpt-4o-mini",
            "--remap_verifier", "nemo_competitive_coding=code_stdio,nemo_math_proofs=lean",
            "--checkpoint_state_freq", "100",
            "--checkpoint_state_dir", "/weka/oe-adapt-default/allennlp/deletable_checkpoint/davidh/v2/",
            "--wandb_project", "rl-mixing",
        ],
    )
