"""Replay real CliqueAI validator rounds against this miner's solver and score it like a validator.

Validators log every round to W&B (https://wandb.ai/toptensor-ai/CliqueAI): the graph, the time
limit, every selected miner's answer and reward. For each round this script runs our solver with
the miner's exact time budget, inserts our answer into the real field, and rescores everyone with
the validator's own CliqueScoreCalculator. It reports, per tier:

- size: how often we match / beat / trail the best clique any miner returned
- reward: our reward vs the field (percentile), and our mean reward vs the best real hotkeys
- strategy: what returning a unique clique one vertex smaller (--near) would have scored

Needs a (free) W&B account key for reading even this public project:
    pip install wandb && wandb login
    python scripts/replay_rounds.py --rounds 200
Offline (rows saved as JSON lines with the WandbRunLogData fields):
    python scripts/replay_rounds.py --rows rounds.jsonl
"""

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from CliqueAI.clique_algorithms import fast_solver  # noqa: E402
from CliqueAI.graph.codec import GraphCodec  # noqa: E402
from CliqueAI.graph.model import LambdaGraph  # noqa: E402
from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator  # noqa: E402

KEYS = ["encoded_matrix", "miner_ans", "time_limit", "difficulty", "number_of_nodes", "miner_hotkeys"]
CODEC = GraphCodec()


def wandb_rows(project: str, limit: int):
    import wandb

    api = wandb.Api(timeout=120)
    got = 0
    for run in api.runs(project, order="-created_at"):
        rows = 0
        for row in run.scan_history(keys=KEYS, page_size=50):
            rows += 1
            yield row
            got += 1
            if got >= limit:
                return
        if rows == 0:  # schema differs from what we expect: show what is there
            print(f"run {run.name}: no rows with {KEYS}; history keys: {sorted(run.history_keys.get('keys', {}))[:40]}")


def file_rows(path: str, limit: int):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                return
            yield json.loads(line)


def score(graph, difficulty, answers):
    return CliqueScoreCalculator(graph, difficulty, answers).get_scores()[-1]


def tier(n: int) -> str:
    return f"{round(n / 100) * 100}v"


def replay(row, threads: int, margin: float, rng: random.Random) -> dict:
    enc, field = row["encoded_matrix"], [list(a or []) for a in row["miner_ans"]]
    d, tl = float(row["difficulty"]), float(row["time_limit"])
    adj = CODEC.matrix_to_list(CODEC.decode_matrix(enc))
    graph = LambdaGraph(uuid="r", label="general", number_of_nodes=len(adj), adjacency_list=adj)
    calc = CliqueScoreCalculator(graph, d, field)
    valid_sizes = [len(a) if calc.is_valid_maximum_clique(a) else 0 for a in field]
    field_best = max(valid_sizes, default=0)

    budget = max(0.2, tl - margin - 0.05 * tl)
    res = fast_solver.solve(enc, budget, threads=threads, extra_args=["--near"])
    ours = fast_solver.choose_clique(res, "rarest", rng)
    near = []
    if res.near_cliques:
        fewest = min(res.near_hits)
        near = rng.choice([c for c, h in zip(res.near_cliques, res.near_hits) if h == fewest])

    rewards = score(graph, d, field + [ours])
    near_rewards = score(graph, d, field + [near]) if near else None
    max_answers = {tuple(sorted(a)) for a, s in zip(field, valid_sizes) if s == field_best and s}
    return {
        "tier": tier(len(adj)),
        "ours_size": len(ours),
        "field_best": field_best,
        "f": sum(s == field_best for s in valid_sizes) / max(1, len(field)),
        "distinct_field_max": len(max_answers),
        "collides": tuple(sorted(ours)) in {tuple(sorted(a)) for a in field},
        "our_reward": float(rewards[-1]),
        "near_reward": float(near_rewards[-1]) if near_rewards is not None else None,
        "field_rewards": [float(x) for x in rewards[:-1]],
        "hotkeys": list(row.get("miner_hotkeys") or [None] * len(field)),
        "our_distinct_max": len(res.cliques),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", default="toptensor-ai/CliqueAI")
    ap.add_argument("--rows", help="JSON-lines file of logged rounds instead of W&B")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--safety-margin", type=float, default=1.0)
    args = ap.parse_args()

    fast_solver.ensure_built()
    rng = random.Random(0)
    source = file_rows(args.rows, args.rounds) if args.rows else wandb_rows(args.project, args.rounds)
    results = []
    for i, row in enumerate(source, 1):
        r = replay(row, args.threads, args.safety_margin, rng)
        results.append(r)
        print(f"[{i}] {r['tier']} ours={r['ours_size']} field_best={r['field_best']} "
              f"f={r['f']:.2f} our_reward={r['our_reward']:.2f} "
              f"field_max_reward={max(r['field_rewards'], default=0):.2f}", flush=True)
    if not results:
        print("no rounds replayed")
        return

    print("\nper tier:")
    by_tier = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)
    for t in sorted(by_tier, key=lambda s: int(s[:-1])):
        rs = by_tier[t]
        n = len(rs)
        pct = [sum(x <= r["our_reward"] for x in r["field_rewards"]) / max(1, len(r["field_rewards"]))
               for r in rs]
        near = [r["near_reward"] - r["our_reward"] for r in rs if r["near_reward"] is not None]
        print(f"  {t}: rounds={n} size beat/match/trail="
              f"{sum(r['ours_size'] > r['field_best'] for r in rs)}/"
              f"{sum(r['ours_size'] == r['field_best'] for r in rs)}/"
              f"{sum(r['ours_size'] < r['field_best'] for r in rs)} "
              f"median_f={statistics.median(r['f'] for r in rs):.2f} "
              f"collisions={sum(r['collides'] for r in rs)} "
              f"reward_percentile_median={statistics.median(pct):.2f} "
              f"near_strategy_gain_mean={statistics.mean(near) if near else float('nan'):+.3f}")

    # Mean reward per real hotkey (as rescored with us in the field) vs ours: the EMA proxy.
    per_key = defaultdict(list)
    for r in results:
        for key, rew in zip(r["hotkeys"], r["field_rewards"]):
            if key:
                per_key[key].append(rew)
    ours = statistics.mean(r["our_reward"] for r in results)
    means = sorted((statistics.mean(v) for v in per_key.values() if len(v) >= 3), reverse=True)
    if means:
        rank = 1 + sum(m > ours for m in means)
        print(f"\nour mean reward {ours:.3f}; would rank {rank} of {len(means) + 1} hotkeys "
              f"(with >=3 rounds); best real hotkey mean {means[0]:.3f}, median {statistics.median(means):.3f}")


if __name__ == "__main__":
    main()
