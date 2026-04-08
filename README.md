data mixing for RL.

🚦 **Current status:** I have a nice default in `configs/examples/gsm_math_qwen_2b.yaml`. Need to (1) setup the Nemotron dataset to get mix domains, (2) set small / large experiment scales, (3) setup eval infra.

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

Ingest [nvidia/Nemotron-RL-Super-Training-Blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends) -> [davidheineman/nemotron-rlvr-openinstruct](https://huggingface.co/datasets/davidheineman/nemotron-rlvr-openinstruct)

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