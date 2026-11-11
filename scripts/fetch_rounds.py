"""Download recent CliqueAI validator rounds from W&B into a JSON-lines file.

Standalone: needs only the `wandb` package (keep it out of the miner's venv):
    python3.12 -m venv ~/wandbenv && ~/wandbenv/bin/pip install wandb
    ~/wandbenv/bin/python scripts/fetch_rounds.py --rounds 150 --out ~/rounds.jsonl
(If W&B asks for an API key, run `~/wandbenv/bin/wandb login` once.)

Then replay them anywhere with:  python scripts/replay_rounds.py --rows rounds.jsonl

A validator keeps one W&B run open for as long as its process lives (often weeks), and W&B scans
a run's history forward from its first step. So this reads the *newest* rounds of the runs that
are still logging, and prints how old they are.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

# Rows must have REQUIRED; runs that also log the rest give coldkey and own-reward analysis.
REQUIRED = ["encoded_matrix", "miner_ans", "time_limit", "difficulty", "miner_hotkeys"]
KEYS = REQUIRED + ["number_of_nodes", "miner_coldkeys", "miner_uids", "miner_rewards", "timestamp", "uuid"]


def unix_time(value) -> float | None:
    """Unix time of a W&B ISO-8601 timestamp (UTC when no zone is given), or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def hours_ago(rows: list[dict], now: float | None = None) -> str:
    """'a-b h ago' for the logged `timestamp`s of `rows`, or '' when they carry none."""
    stamps = [r["timestamp"] for r in rows if isinstance(r.get("timestamp"), (int, float))]
    if not stamps:
        return ""
    now = time.time() if now is None else now
    return f"logged {(now - max(stamps)) / 3600:.1f}-{(now - min(stamps)) / 3600:.1f} h ago"


def active_runs(api, project: str, count: int, max_age_hours: float, log=print) -> list:
    """Up to `count` runs that logged most recently, minus those idle for over `max_age_hours`.
    (Creation time says little: a validator's run stays open until its process restarts.)"""
    now = time.time()
    runs = []
    for run in list(api.runs(project, order="-heartbeat_at", per_page=count))[:count]:
        beat = unix_time(getattr(run, "heartbeat_at", None))
        if beat is not None and now - beat > max_age_hours * 3600:
            log(f"  run {run.name}: idle for {(now - beat) / 3600:.0f} h, skipped")
            continue
        runs.append(run)
    return runs


def newest_rows(run, count: int, page_size: int = 25, max_widen: int = 16) -> list[dict]:
    """The newest `count` rows of `run` holding every REQUIRED key (and all of KEYS when the run
    logs them), oldest first.

    W&B scans history forward from `min_step`, so read a window at the end of the run and widen
    it (up to `max_widen` x `count` steps) while it holds too few rounds."""
    last = run.lastHistoryStep
    if last is None or last < 0 or count <= 0:
        return []
    window, keys, rows = count, None, []
    while True:
        start = max(0, last + 1 - window)
        for ks in [keys] if keys else (KEYS, REQUIRED):
            rows = [dict(row) for row in run.scan_history(keys=ks, page_size=page_size, min_step=start)
                    if all(row.get(k) is not None for k in ks)]  # fmt: skip
            if rows:
                keys = ks
                break
        if len(rows) >= count or start == 0 or window >= max_widen * count:
            return rows[-count:]
        window *= 4


def history_length(run) -> int:
    try:
        return run.lastHistoryStep + 1
    except Exception:
        return 0


def fetch_newest(api, project: str, rounds: int, runs: int, max_age_hours: float, log=print):
    """Yield up to `rounds` of the newest logged rounds, split over the active runs. Shortest runs
    go first, so a just-restarted validator's short run leaves the rest of its share to the
    others. Each row gets its run's name under "run"."""
    active = sorted(active_runs(api, project, runs, max_age_hours, log), key=history_length)
    log(f"{len(active)} run(s) in {project} logged within {max_age_hours:g} h")
    left = rounds
    for i, run in enumerate(active):
        share = -(-left // (len(active) - i))  # ceil
        try:
            rows = newest_rows(run, share)
        except Exception as e:  # one bad run should not stop the download
            log(f"  run {run.name}: {type(e).__name__}: {e}")
            continue
        if not rows:
            log(f"  run {run.name}: no rows with all of {REQUIRED}")
            continue
        log(f"  run {run.name}: {len(rows)} rounds, {hours_ago(rows) or 'no timestamps'}")
        for row in rows:
            row["run"] = run.name
            yield row
        left -= len(rows)
        if left <= 0:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", default="toptensor-ai/CliqueAI")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--runs", type=int, default=12, help="most recently active runs to consider")
    ap.add_argument("--max-age-hours", type=float, default=48,
                    help="skip runs that have not logged for this long")  # fmt: skip
    ap.add_argument("--out", default="rounds.jsonl")
    args = ap.parse_args()

    import wandb

    api = wandb.Api(timeout=120)
    written = []
    with open(args.out, "w") as out:
        for row in fetch_newest(api, args.project, args.rounds, args.runs, args.max_age_hours):
            out.write(json.dumps({k: row.get(k) for k in KEYS + ["run"]}) + "\n")
            written.append({"timestamp": row.get("timestamp")})
    print(f"wrote {len(written)} rounds to {args.out} {hours_ago(written)}".rstrip())
    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
