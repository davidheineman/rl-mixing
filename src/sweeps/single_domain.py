from experiments import (
    Experiment,
    Mix,
    MixComponent,
    EvalMix,
    EvalMixComponent,
    TrainingConfig,
    InfraConfig,
    BeakerConfig,
)


DATASET = "davidheineman/nemotron-super-stage-1-unmixed-openinstruct"

USEFUL_DOMAINS = [
    "dapo_math",
    "skywork_math",
    "math_proofs",
    "multiturn_chat",
    "reasoning_gym",
    "competitive_coding",
    "structured_outputs",
    "instruction_following",
    "mcqa",
]

EVAL_MIX = EvalMix([
    EvalMixComponent("mnoukhov/aime_2025_openinstruct", num_samples=32, split="train"),
    EvalMixComponent("mnoukhov/brumo_2025_openinstruct", num_samples=32, split="train"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=32, split="gpqa"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=32, split="ifeval"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=32, split="mmlu"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=32, split="humanevalplus"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=32, split="mbppplus"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=32, split="livecodebench"),
    EvalMixComponent("davidheineman/eval-openinstruct", num_samples=32, split="ifeval_ood"),
])


def get_experiments() -> list[Experiment]:
    base = Experiment(
        name="single-domain-qwen3-1.7b",
        model="Qwen/Qwen3-1.7B-Base",
        chat_template="qwen_thinker",
        mix=Mix([]),
        eval_mix=EVAL_MIX,
        training=TrainingConfig(
            learning_rate=1e-6,
            beta=0.0,
            total_episodes=131072,
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
            local_eval_every=100,
            save_freq=100,
            non_stop_penalty=False,
            apply_verifiable_reward=True,
        ),
        infra=InfraConfig(
            num_learners_per_node=4,
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
            setup_commands=[
                'git clone --depth 1 -b dhei/rl-mixing https://github.com/allenai/open-instruct.git /tmp/oi && cp /tmp/oi/open_instruct/nemo_verifiers.py /tmp/oi/open_instruct/ground_truth_utils.py open_instruct/ && uv pip install openapi-schema-validator pyyaml xmltodict -q 2>/dev/null || true',
                'python -c "p=__import__(\'transformers\').__file__.replace(\'__init__.py\',\'modeling_rope_utils.py\');t=open(p).read().replace(\'received_keys -= ignore_keys\',\'received_keys -= set(ignore_keys)\');open(p,\'w\').write(t)"',
                'python -c "p=__import__(\'transformers\').__file__.replace(\'__init__.py\',\'modeling_utils.py\');t=open(p).read().replace(\'model = cls(config, *model_args, **model_kwargs)\',\'model_kwargs.pop(\\"use_cache\\", None); model = cls(config, *model_args, **model_kwargs)\');open(p,\'w\').write(t)"',
            ],
            env={
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
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
            "--code_api_url", "https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program",
            "--code_max_execution_time", "6.0",
            "--checkpoint_state_freq", "100",
            "--checkpoint_state_dir", "/weka/oe-adapt-default/allennlp/deletable_checkpoint/davidh/",
            "--wandb_project", "rl-mixing",
        ],
    )

    experiments = []
    for domain in USEFUL_DOMAINS:
        mix = Mix([MixComponent(DATASET, 1.0, split=domain)])
        exp = base.vary_mix(mix, name_suffix=domain)
        experiments.append(exp)

    return experiments
