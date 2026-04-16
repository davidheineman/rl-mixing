import argparse
import csv
import io
import re
import wandb
from rich.console import Console
from rich.table import Table

ENTITY = "ai2-llm"
PROJECT = "rl-mixing"
RUN_PATTERN = re.compile(r"^(?:nvidia|natural)-mix-")


def get_eval_runs() -> list[dict]:
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    rows = []
    for run in runs:
        if not RUN_PATTERN.match(run.name):
            continue

        summary = dict(run.summary)
        eval_keys = {k: v for k, v in summary.items() if k.startswith("eval/")}
        if not eval_keys:
            continue

        rows.append({"run_name": run.name, "run_id": run.id, "state": run.state, **eval_keys})

    return rows


def fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    rows = get_eval_runs()

    if not rows:
        print("No matching runs with eval/ keys found.")
        raise SystemExit(0)

    eval_keys = sorted({k for row in rows for k in row if k.startswith("eval/")})
    header = ["metric"] + [r["run_name"] for r in rows]
    table_rows = [
        [k.removeprefix("eval/")] + [fmt(row.get(k, "")) for row in rows]
        for k in eval_keys
    ]

    if args.csv:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        writer.writerows(table_rows)
        print(buf.getvalue(), end="")
    else:
        table = Table(title="eval/ metrics", show_lines=True, highlight=True)
        table.add_column(header[0], style="bold", no_wrap=True)
        for col in header[1:]:
            table.add_column(col, no_wrap=True)
        for tr in table_rows:
            table.add_row(*tr)

        console = Console()
        console.print(table)
        console.print(f"\n[bold]{len(rows)}[/bold] runs, [bold]{len(eval_keys)}[/bold] eval metrics")
