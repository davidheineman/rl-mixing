## Using Compute

You have access to run on a compute cluster!

You can run in `ai2/adaptability` using the `ai2/oe-base` budget. PLEASE RUN ON `urgent` ALWAYS!

## Beaker and Cuvette

I have a pip library, `cuvette` which has some useful utilities:

```sh
# manage jobs
uv run -w cuvette bstream <job_or_experiment_id> # stream logs for job
uv run -w cuvette bpriority <job_or_experiment_id> # change priority for a job
beaker experiment stop <job_or_experiment_id> # stop a job

# manage workspace
uv run -w cuvette blist # list secrets in workspace
uv run -w cuvette bcopy -f <from_workspace> -t <to_workspace> -s <secret> -n <new_secret_name> # copy secret
```

When launching a job, you might find `bstream` useful to monitor that job before it ends.

## Gantry

Gantry is the Python SDK for running on Gantry. You can see an example of this in-action here: https://github.com/davidheineman/tinking/blob/main/tinking/beaker/launch.py.

You can use that above logic again, if it's helpful.

## Repo-specific details

I've cloned some internal Ai2 repos for your reference:

- `open-instruct/` - The post-training library
- `oe-eval-internal/` - The eval library

Here are some external codebases

- `RL/` - Nemotron RL (NVIDIA)
- `Gym/` - Nemotron Gym (NVIDIA)

You can use `open-instruct/` with Gantry! Just enter the folder, implement your method, push to GitHub, and run the command and it will wrok with gantry.

## Cursor Cloud specific instructions

### Project overview

`rl-mixing` is a Python job launcher and data pipeline for RL data mixing experiments. It generates and launches Beaker jobs using `open-instruct`. Actual GPU training runs remotely on Beaker, not locally.

### Environment setup

- Requires Python 3.12 and `uv`.
- `open-instruct/` must be cloned at the repo root (it is a local editable path dependency).
- The `pyproject.toml` `environments` field is set to `sys_platform == 'darwin'`, so `uv sync` will fail on Linux. Use `uv pip install` with `--override` instead:
  ```sh
  uv venv --python 3.12 --clear
  source .venv/bin/activate
  echo 'transformers>=5.4.0' > /tmp/overrides.txt
  uv pip install --override /tmp/overrides.txt -e "./open-instruct" pyyaml cuvette ruff
  uv pip install --override /tmp/overrides.txt -e "."
  ```
- The `transformers>=5.4.0` override is needed because `open-instruct` requires `transformers>=5.4.0` but `vllm` pins `transformers<5`. The `open-instruct` repo uses `override-dependencies` in its own `pyproject.toml` for this.

### Running the application

- See `README.md` for usage. Key commands:
  - `python src/launch.py configs/examples/small_scale.yaml --dry-run` — validate and print command without launching
  - `python src/launch.py --sweep src/sweeps/example_gsm_if.py --dry-run` — sweep dry-run
  - `python src/launch.py configs/examples/small_scale.yaml --list` — list experiments (note: minor bug with `number_samples_per_prompt` attribute)
- Launching real jobs requires Beaker credentials and cluster access.

### Linting

- No project-specific lint config; use `ruff check src/ ingest/` for basic linting.