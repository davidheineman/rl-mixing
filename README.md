data mixing for RL. 

Data mixing for pretraining predicts a single "optimal" mix. However, we know the "best" data for RL changes depending on the capability of the model. My idea is to predict *the trajectory along the simplex during training*, by simply adding a timestep parameter to the fitting procedure.

### notes

🚦 **Current status:** Data + eval is implemented. Debug runs are training now...

- In `src/sweeps/single_domain.py`, I have "debug" runs where I train using 100% of the mix domain on Qwen 3 1.7B. Each takes ~6 hours on 1 node. See runs: at [wandb.ai/ai2-llm/rl-mixing](https://wandb.ai/ai2-llm/rl-mixing?nw=4u44z0eam48).
    - https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends
    - Need to check that all domains *actually work*
    - Can we make runtime 50% shorter for small-scale run?
- In-loop eval is implemented 
    - https://huggingface.co/datasets/davidheineman/eval-openinstruct
    - but logging is confusing (needs to show *individual task pass rate*). Also, needs a flag to eval on the full sample at the end of training.
- Fixes are implemented
    - but it's unclear whether I'm actually running on my `open-instruct` branch... need to figure out the monkey patch that cursor implemented.

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
python src/launch.py configs/examples/small_scale.yaml -f
python src/launch.py configs/examples/eval_only.yaml -f

# --sweep launches a sweep
python src/launch.py --sweep src/sweeps/single_domain.py --dry-run
```

#### python api

```python
from rl_mixing.experiments import Experiment, Mix, MixComponent, EvalMix, EvalMixComponent

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
# convert nemotron datasets
python ingest/convert_individual.py \
    --output-dir ingest/converted_nemotron \
    --push-to-hub davidheineman/nemotron-super-stage-1-unmixed-openinstruct

# convert from oe-eval-internal
python ingest/convert_evals.py \
    --output-dir ingest/converted_evals \
    --push-to-hub davidheineman/eval-openinstruct
```
