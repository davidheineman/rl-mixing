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


def get_experiments() -> list[Experiment]:
    base = Experiment(
        name="gsm-if-proportion-sweep",
        model="allenai/Llama-3.1-Tulu-3-8B-DPO",
        mix=Mix([]),  # overridden below
        eval_mix=EvalMix([
            EvalMixComponent("allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", num_samples=16),
        ]),
        training=TrainingConfig(
            learning_rate=5e-7,
            beta=0.01,
            total_episodes=10_000_000,
            num_samples_per_prompt_rollout=16,
            seed=1,
            local_eval_every=100,
        ),
        infra=InfraConfig(
            num_learners_per_node=4,
            vllm_tensor_parallel_size=1,
            vllm_num_engines=4,
        ),
        beaker=BeakerConfig(
            cluster="ai2/jupiter",
            budget="ai2/oe-adapt",
            num_nodes=1,
            gpus=8,
            preemptible=True,
        ),
    )

    experiments = []
    for gsm_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if_frac = 1.0 - gsm_frac

        components = []
        if gsm_frac > 0:
            components.append(MixComponent("allenai/RLVR-GSM-MATH", gsm_frac))
        if if_frac > 0:
            components.append(MixComponent("allenai/RLVR-IF-Mixed-Constraints", if_frac))

        exp = base.vary_mix(
            Mix(components),
            name_suffix=f"gsm{gsm_frac:.0%}-if{if_frac:.0%}",
        )
        experiments.append(exp)

    return experiments
