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
