"""Download recent CliqueAI validator rounds from W&B into a JSON-lines file.

Standalone: needs only the `wandb` package (keep it out of the miner's venv):
    python3.12 -m venv ~/wandbenv && ~/wandbenv/bin/pip install wandb
    ~/wandbenv/bin/wandb login --host https://api.wandb.ai
    ~/wandbenv/bin/python scripts/fetch_rounds.py --rounds 300 --out ~/rounds.jsonl

Then replay them anywhere with:  python scripts/replay_rounds.py --rows rounds.jsonl
"""

import argparse
import json
import sys

# Rows must have REQUIRED; runs that also log the rest give coldkey and own-reward analysis.
REQUIRED = ["encoded_matrix", "miner_ans", "time_limit", "difficulty", "miner_hotkeys"]
KEYS = REQUIRED + ["number_of_nodes", "miner_coldkeys", "miner_uids", "miner_rewards", "timestamp", "uuid"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", default="toptensor-ai/CliqueAI")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--runs", type=int, default=12, help="most recent validator runs to read")
    ap.add_argument("--out", default="rounds.jsonl")
    args = ap.parse_args()

    import wandb

    api = wandb.Api(timeout=120)
    runs = list(api.runs(args.project, order="-created_at", per_page=args.runs))[: args.runs]
    print(f"{len(runs)} recent runs in {args.project}", flush=True)
    per_run = max(1, args.rounds // max(1, len(runs)))
    written = 0
    with open(args.out, "w") as out:
        for run in runs:
            got = 0
            try:
                for keys in (KEYS, REQUIRED):
                    for row in run.scan_history(keys=keys, page_size=25):
                        record = {k: row.get(k) for k in KEYS}
                        record["run"] = run.name
                        out.write(json.dumps(record) + "\n")
                        got += 1
                        written += 1
                        if got >= per_run or written >= args.rounds:
                            break
                    if got:
                        break
            except Exception as e:  # one bad run should not stop the download
                print(f"  run {run.name}: {type(e).__name__}: {e}", flush=True)
            if got == 0:
                keys = sorted(run.summary.keys())[:40]
                print(f"  run {run.name}: no rows with all fields; summary keys: {keys}", flush=True)
            else:
                print(f"  run {run.name}: {got} rounds", flush=True)
            if written >= args.rounds:
                break
    print(f"wrote {written} rounds to {args.out}")
    if written == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
