from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
OPEN_INSTRUCT_DIR = REPO_ROOT / "open-instruct"
MASON_PY = OPEN_INSTRUCT_DIR / "mason.py"
MASON_PY = OPEN_INSTRUCT_DIR / "mason.py"
TRAINING_SCRIPT = "open_instruct/grpo_fast.py"
RAY_SETUP_SCRIPT = "configs/beaker_configs/ray_node_setup.sh"


# ---------------------------------------------------------------------------
# Data mix specification
# ---------------------------------------------------------------------------

@dataclass
class MixComponent:
    """A single dataset + weight in a mix."""
    dataset: str
    proportion: float
    split: str = "train"


@dataclass
class Mix:
    """A weighted collection of datasets."""
    components: list[MixComponent]

    def to_mixer_list(self) -> list[str]:
        """Produce the flat alternating list for --dataset_mixer_list."""
        result: list[str] = []
        for c in self.components:
            result.extend([c.dataset, str(c.proportion)])
        return result

    def splits(self) -> list[str]:
        # open-instruct auto-expands a single split to all datasets;
        # passing N splits for N datasets (but 2N mixer_list elements) errors.
        splits = [c.split for c in self.components]
        if len(set(splits)) == 1:
            return splits[:1]
        return splits

    def summary(self) -> str:
        parts = [f"{c.dataset.split('/')[-1]}={c.proportion}" for c in self.components]
        return "+".join(parts)


@dataclass
class EvalMix:
    """Evaluation datasets.  `num_samples` per dataset instead of proportion."""
    components: list[EvalMixComponent]

    def to_mixer_list(self) -> list[str]:
        result: list[str] = []
        for c in self.components:
            result.extend([c.dataset, str(c.num_samples)])
        return result

    def splits(self) -> list[str]:
        splits = [c.split for c in self.components]
        if len(set(splits)) == 1:
            return splits[:1]
        return splits


@dataclass
class EvalMixComponent:
    dataset: str
    num_samples: int = 16
    split: str = "train"


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # GRPOExperimentConfig fields
    learning_rate: float = 5e-7
    total_episodes: int = 10_000_000
    beta: float = 0.01
    kl_estimator: int = 3
    lr_scheduler_type: str = "constant"
    deepspeed_stage: int = 2
    gradient_checkpointing: bool = True
    per_device_train_batch_size: int = 1
    seed: int = 1
    save_freq: int = 100
    local_eval_every: int = 100
    warmup_ratio: float = 0.0

    # StreamingDataLoaderConfig fields
    response_length: int = 2048
    max_prompt_token_length: int = 2048
    pack_length: int = 4096
    temperature: float = 1.0
    num_samples_per_prompt_rollout: int = 16
    num_unique_prompts_rollout: int = 16
    async_steps: int = 1
    non_stop_penalty: bool = True
    non_stop_penalty_value: float = 0.0
    apply_verifiable_reward: bool = True
    verification_reward: float = 10.0

    def to_args(self) -> list[str]:
        args = [
            # GRPOExperimentConfig
            "--learning_rate", str(self.learning_rate),
            "--total_episodes", str(self.total_episodes),
            "--beta", str(self.beta),
            "--kl_estimator", str(self.kl_estimator),
            "--lr_scheduler_type", self.lr_scheduler_type,
            "--deepspeed_stage", str(self.deepspeed_stage),
            "--per_device_train_batch_size", str(self.per_device_train_batch_size),
            "--seed", str(self.seed),
            "--save_freq", str(self.save_freq),
            "--local_eval_every", str(self.local_eval_every),
            "--warmup_ratio", str(self.warmup_ratio),
            # StreamingDataLoaderConfig
            "--response_length", str(self.response_length),
            "--max_prompt_token_length", str(self.max_prompt_token_length),
            "--pack_length", str(self.pack_length),
            "--temperature", str(self.temperature),
            "--num_samples_per_prompt_rollout", str(self.num_samples_per_prompt_rollout),
            "--num_unique_prompts_rollout", str(self.num_unique_prompts_rollout),
            "--async_steps", str(self.async_steps),
            "--verification_reward", str(self.verification_reward),
        ]
        if self.gradient_checkpointing:
            args.append("--gradient_checkpointing")
        if self.non_stop_penalty:
            args.extend(["--non_stop_penalty", "--non_stop_penalty_value", str(self.non_stop_penalty_value)])
        if self.apply_verifiable_reward:
            args.append("--apply_verifiable_reward")
        return args


# ---------------------------------------------------------------------------
# GPU / vLLM layout
# ---------------------------------------------------------------------------

@dataclass
class InfraConfig:
    num_learners_per_node: int = 8
    vllm_tensor_parallel_size: int = 1
    vllm_num_engines: int = 8

    def to_args(self) -> list[str]:
        return [
            "--num_learners_per_node", str(self.num_learners_per_node),
            "--vllm_tensor_parallel_size", str(self.vllm_tensor_parallel_size),
            "--vllm_num_engines", str(self.vllm_num_engines),
        ]


# ---------------------------------------------------------------------------
# Beaker launch configuration
# ---------------------------------------------------------------------------

@dataclass
class BeakerConfig:
    cluster: str = "ai2/jupiter"
    workspace: str | None = None
    budget: str = "ai2/oe-adapt"
    num_nodes: int = 1
    gpus: int = 8
    priority: str = "normal"
    preemptible: bool = True
    image: str = "nathanl/open_instruct_auto"
    shared_memory: str = "10.24gb"
    setup_commands: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def to_mason_args(self) -> list[str]:
        args = [
            "--cluster", self.cluster,
            "--budget", self.budget,
            "--num_nodes", str(self.num_nodes),
            "--gpus", str(self.gpus),
            "--priority", self.priority,
            "--image", self.image,
            "--shared_memory", self.shared_memory,
            "--pure_docker_mode",
            "--no_auto_dataset_cache",
        ]
        if self.workspace:
            args.extend(["--workspace", self.workspace])
        if self.preemptible:
            args.append("--preemptible")
        for k, v in self.env.items():
            args.extend(["--env", f"{k}={v}"])
        return args


# ---------------------------------------------------------------------------
# Full experiment definition
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    name: str
    model: str
    mix: Mix
    eval_mix: EvalMix | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    beaker: BeakerConfig = field(default_factory=BeakerConfig)
    chat_template: str | None = "tulu"
    output_base: str = "/weka/oe-adapt-default"
    with_tracking: bool = True
    extra_args: list[str] = field(default_factory=list)

    # ----- derived helpers -----

    @property
    def output_dir(self) -> str:
        user = os.environ.get("USER", "unknown")
        return f"{self.output_base}/{user}/rl-mixing/{self.name}"

    def build_training_command(self) -> list[str]:
        """Build the training script invocation (runs inside the Beaker container)."""
        cmd = [
            "python", TRAINING_SCRIPT,
            "--exp_name", self.name,
            "--output_dir", self.output_dir,
            "--model_name_or_path", self.model,
            "--dataset_mixer_list", *self.mix.to_mixer_list(),
            "--dataset_mixer_list_splits", *self.mix.splits(),
        ]
        if self.chat_template:
            cmd.extend(["--chat_template_name", self.chat_template])

        if self.eval_mix:
            cmd.extend(["--dataset_mixer_eval_list", *self.eval_mix.to_mixer_list()])
            cmd.extend(["--dataset_mixer_eval_list_splits", *self.eval_mix.splits()])

        cmd.extend(self.training.to_args())
        cmd.extend(self.infra.to_args())

        if self.with_tracking:
            cmd.append("--with_tracking")

        cmd.extend(self.extra_args)

        return cmd

    def validate(self) -> bool:
        """Validate training args locally against the open-instruct parser.

        Returns True if validation passed (or was skipped due to missing deps).
        Returns False if args are invalid.
        """
        training_cmd = self.build_training_command()
        training_args = training_cmd[2:]  # skip ['python', 'open_instruct/grpo_fast.py']

        validation_script = textwrap.dedent("""\
            import sys
            from open_instruct.utils import ArgumentParserPlus
            from open_instruct import grpo_utils, data_loader as data_loader_lib
            from open_instruct.dataset_transformation import TokenizerConfig
            from open_instruct.model_utils import ModelConfig
            from open_instruct.environments.tools.utils import EnvsConfig
            parser = ArgumentParserPlus((
                grpo_utils.GRPOExperimentConfig,
                TokenizerConfig,
                ModelConfig,
                data_loader_lib.StreamingDataLoaderConfig,
                data_loader_lib.VLLMConfig,
                EnvsConfig,
            ))
            parser.set_defaults(
                exp_name="grpo", warmup_ratio=0.0,
                max_grad_norm=1.0, per_device_train_batch_size=1,
            )
            parser.parse_args_into_dataclasses()
        """)

        try:
            result = subprocess.run(
                [sys.executable, "-c", validation_script, *training_args],
                cwd=str(OPEN_INSTRUCT_DIR),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            print(f"  [validate] timed out for {self.name}, skipping")
            return True

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
                print(f"  [validate] skipped (missing local deps)")
                return True
            print(f"\n  [validate] FAILED for {self.name}:")
            for line in stderr.splitlines()[-5:]:
                print(f"    {line}")
            print()
            return False

        print(f"  [validate] OK for {self.name}")
        return True

    def build_full_command(self) -> list[str]:
        """Build the full mason.py command for Beaker launch."""
        training_cmd = self.build_training_command()

        setup_prefix = ""
        if self.beaker.setup_commands:
            setup_prefix = " && ".join(self.beaker.setup_commands) + " && "

        if self.beaker.num_nodes > 1:
            inner = setup_prefix + f"source {RAY_SETUP_SCRIPT} && " + " ".join(training_cmd)
        else:
            inner = setup_prefix + " ".join(training_cmd)

        mason_cmd = [
            sys.executable, str(MASON_PY),
            *self.beaker.to_mason_args(),
            "--description", self.name,
            "--task_name", self.name,
            "--", *inner.split(),
        ]
        return mason_cmd

    def launch(self, dry_run: bool = False) -> subprocess.CompletedProcess | None:
        """Launch on Beaker. If dry_run, just print the command."""
        cmd = self.build_full_command()
        cmd_str = " \\\n    ".join(cmd)

        self.beaker_experiment_id: str | None = None

        if dry_run:
            print(f"\n{'='*60}")
            print(f"[DRY RUN] {self.name}")
            print(f"{'='*60}")
            print(cmd_str)
            print()
            return None

        print(f"\nLaunching: {self.name}")
        print(f"  mix: {self.mix.summary()}")
        print(f"  model: {self.model}")
        print(f"  output: {self.output_dir}")

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(OPEN_INSTRUCT_DIR) + (f":{existing}" if existing else "")

        result = subprocess.run(
            cmd,
            cwd=str(OPEN_INSTRUCT_DIR),
            env=env,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)

        match = re.search(r'https://beaker\.org/ex/([A-Z0-9]+)', result.stdout + result.stderr)
        if match:
            self.beaker_experiment_id = match.group(1)

        result.check_returncode()
        return result

    # ----- serialization -----

    def to_yaml(self) -> str:
        return yaml.dump(self._to_dict(), default_flow_style=False, sort_keys=False)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml())

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        data = yaml.safe_load(Path(path).read_text())
        return cls._from_dict(data)

    def _to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "chat_template": self.chat_template,
            "reward_model_multiplier": self.reward_model_multiplier,
            "apply_verifiable_reward": self.apply_verifiable_reward,
            "non_stop_penalty": self.non_stop_penalty,
            "stop_token": self.stop_token,
            "penalty_reward_value": self.penalty_reward_value,
            "output_base": self.output_base,
            "with_tracking": self.with_tracking,
        }
        d["mix"] = [asdict(c) for c in self.mix.components]
        if self.eval_mix:
            d["eval_mix"] = [asdict(c) for c in self.eval_mix.components]
        d["training"] = asdict(self.training)
        d["infra"] = asdict(self.infra)
        d["beaker"] = asdict(self.beaker)
        if self.extra_args:
            d["extra_args"] = self.extra_args
        return d

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> Experiment:
        mix = Mix([MixComponent(**c) for c in d.pop("mix")])
        eval_mix = None
        if "eval_mix" in d:
            eval_mix = EvalMix([EvalMixComponent(**c) for c in d.pop("eval_mix")])
        training = TrainingConfig(**d.pop("training", {}))
        infra = InfraConfig(**d.pop("infra", {}))
        beaker = BeakerConfig(**d.pop("beaker", {}))
        extra_args = d.pop("extra_args", [])
        return cls(
            mix=mix,
            eval_mix=eval_mix,
            training=training,
            infra=infra,
            beaker=beaker,
            extra_args=extra_args,
            **d,
        )

    def vary_mix(self, new_mix: Mix, name_suffix: str | None = None) -> Experiment:
        """Return a copy with a different mix (useful for sweeps)."""
        new = copy.deepcopy(self)
        new.mix = new_mix
        if name_suffix:
            new.name = f"{self.name}_{name_suffix}"
        else:
            new.name = f"{self.name}_{new_mix.summary()}"
        return new

    def vary(self, **overrides: Any) -> Experiment:
        """Return a copy with arbitrary field overrides."""
        new = copy.deepcopy(self)
        for k, v in overrides.items():
            if not hasattr(new, k):
                raise ValueError(f"Experiment has no attribute '{k}'")
            setattr(new, k, v)
        return new


# ---------------------------------------------------------------------------
# Batch launcher helpers
# ---------------------------------------------------------------------------

def launch_sweep(
    experiments: list[Experiment],
    dry_run: bool = False,
) -> list[subprocess.CompletedProcess | None]:
    """Launch a list of experiments, printing a summary first."""
    print(f"\n{'='*60}")
    print(f"Launching {len(experiments)} experiments")
    print(f"{'='*60}")
    for i, exp in enumerate(experiments):
        print(f"  [{i+1}] {exp.name:<40s}  mix: {exp.mix.summary()}")
    print()

    results = []
    for exp in experiments:
        results.append(exp.launch(dry_run=dry_run))
    return results


def load_experiments(path: str | Path) -> list[Experiment]:
    """Load one or more experiments from a YAML file.

    The YAML can be either a single experiment dict or a list of dicts.
    """
    data = yaml.safe_load(Path(path).read_text())
    if isinstance(data, list):
        return [Experiment._from_dict(d) for d in data]
    return [Experiment._from_dict(data)]
