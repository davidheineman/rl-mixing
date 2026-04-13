data mixing for RL. 

Data mixing for pretraining predicts a single "optimal" mix. However, we know the "best" data for RL changes depending on the capability of the model. My idea is to predict *the trajectory along the simplex during training*, by simply adding a timestep parameter to the fitting procedure.

### notes

- `open-instruct/` - https://github.com/allenai/open-instruct/commits/dhei/rl-mixing

🚦 **Current status:** Data + eval is implemented. Debug runs are training now...

- In `src/sweeps/single_domain.py`, I have "debug" runs where I train using 100% of the mix domain on Qwen 3 1.7B. Each takes ~6 hours on 1 node. See runs: at [wandb.ai/ai2-llm/rl-mixing](https://wandb.ai/ai2-llm/rl-mixing?nw=4u44z0eam48).
    - https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends
    - Need to check that all domains *actually work* (`competitive_coding`, `math_proofs` might be broken. both require code execution APIs)
    - Can we make runtime 50% shorter for small-scale run?
- In-loop eval is implemented 
    - https://huggingface.co/datasets/davidheineman/eval-openinstruct
    - but logging is confusing (needs to show *individual task pass rate*). Also, needs a flag to eval on the full sample at the end of training.
- Fixes are implemented
    - but it's unclear whether I'm actually running on my `open-instruct` branch... need to figure out the monkey patch that cursor implemented.

**Result 1:** These are Qwen 3 1.7B Base models trained on a few stateless Nemotron RL env types. As you can see, the "correct rate" training curves are completely different depending on the environment type:

<p align="center">
<img width="600" alt="Screenshot 2026-04-13 at 11 56 49 AM" src="https://github.com/user-attachments/assets/3261f419-b5b4-41dd-a561-0ed524ff2f8d" />
</p>

I'm concerned that I don't understand properties of environments well enough. In text pretraining (for mixing), we assume that the optimal mix at small scales (e.g. 30M param) is the same at larger scales (e.g. 7B param). However, for RL, the optimal environments depends on the capability of the base model, *and that capabilitiy chagnes during training*. It feels like this would create a bound on how far we could extrapolate performance (unlike pretraining, where 10,000x compute extrapolations are suprisingly reasonable).

This problem may have been addressed in prior literature ([epiplexity](https://arxiv.org/abs/2601.03220v1)? [coffee automation](https://arxiv.org/abs/1405.6903)? maybe there's some paper on Atari games or something)

It could be the case that mixing is the wrong tool. Instead, you could try to train a sampler (say, $g_i(\tau)$) which predicts the marginal gain from training on source $i$ at capability level $\tau$. That's a form of curriculum learning.

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
