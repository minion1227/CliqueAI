"""Replay logged CliqueAI validator rounds against this miner's solver and tune its answer policy.

Validators log every round to W&B (https://wandb.ai/toptensor-ai/CliqueAI): the graph, the time
limit, every selected miner's answer, coldkey and reward. For each round this script runs our
solver with the miner's time budget, puts each candidate answer into the real field (replacing
our own hotkey's logged answer if it is there) and rescores everyone exactly like the validator:

    reward = optimality * (1 + difficulty) + diversity

It compares the two answers the miner can give - a maximum clique ("max") and a unique maximal
clique one vertex smaller ("near") - per round, and sweeps the miner's --solver.near_max_cliques
threshold (answer "near" when the search found at most that many distinct maximum cliques).

On a machine that can reach W&B (e.g. the miner's VPS: runs at nice 19, so a live miner keeps
priority; a round takes about its time limit):
    python scripts/replay_rounds.py --rounds 150 --hotkey <your hotkey> --save replay.jsonl
From rows saved by scripts/fetch_rounds.py:
    python scripts/replay_rounds.py --rows rounds.jsonl --save replay.jsonl
Re-analyse a saved replay instantly (no solving; the file is small enough to copy anywhere):
    python scripts/replay_rounds.py --load replay.jsonl
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_rounds import fetch_newest, hours_ago  # noqa: E402  (W&B access lives there)

THRESHOLDS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 256]
K_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128),
             (129, 256), (257, 10**9)]  # fmt: skip


# ------------------------------------------------------------------------------------ sources


def wandb_rows(project: str, limit: int, runs: int, max_age_hours: float):
    import wandb

    log = lambda msg: print(msg, flush=True)  # noqa: E731
    yield from fetch_newest(wandb.Api(timeout=120), project, limit, runs, max_age_hours, log)


def file_rows(path: str, limit: int):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                return
            yield json.loads(line)


# ------------------------------------------------------------------------------------ scoring


class RoundScorer:
    """The validator's CliqueScoreCalculator, rewritten with adjacency sets so a candidate can be
    rescored against ~250 logged answers in milliseconds. Checked against the original."""

    def __init__(self, adjacency: list[list[int]], difficulty: float, field: list[list[int]]):
        self.n = len(adjacency)
        self.adj = [set(a) for a in adjacency]
        self.difficulty = difficulty
        self.field = field
        self.sizes = [len(a) if self.is_valid(a) else 0 for a in field]
        self.canon = Counter(tuple(sorted(a)) for a in field)

    def is_valid(self, nodes) -> bool:
        nodes = list(nodes)
        members = set(nodes)
        if not members or len(members) != len(nodes):
            return False
        if not all(isinstance(v, int) and 0 <= v < self.n for v in members):
            return False
        for v in nodes:
            if not members - {v} <= self.adj[v]:
                return False
        # Maximal iff no vertex is adjacent to every member (members are not their own neighbours).
        return not set.intersection(*(self.adj[v] for v in nodes))

    def rewards(self, answer: list[int] | None = None) -> list[float]:
        """Rewards of the field plus `answer` (appended last, if given)."""
        sizes = list(self.sizes)
        canon = self.canon
        answers = list(self.field)
        if answer is not None:
            sizes.append(len(answer) if self.is_valid(answer) else 0)
            canon = canon + Counter([tuple(sorted(answer))])
            answers.append(answer)
        total = len(sizes)
        if total == 0:
            return []
        max_size = max(sizes)
        if max_size <= 0:
            return [0.0] * total
        ordered = sorted(sizes)
        omega = []
        for s in sizes:
            if s <= 0:
                omega.append(0.0)
                continue
            larger = total - _bisect_right(ordered, s)
            omega.append(math.exp(-(larger / total) / (s / max_size)))
        max_omega = max(omega)
        delta = [(1.0 / canon[tuple(sorted(a))]) if s > 0 else 0.0 for a, s in zip(answers, sizes)]
        max_delta = max(delta)
        return [
            (o / max_omega if max_omega else 0.0) * (1 + self.difficulty)
            + (dl / max_delta if max_delta else 0.0)
            for o, dl in zip(omega, delta)
        ]

    def score(self, answer: list[int]) -> tuple[float, int]:
        """Our reward with `answer`, and how many field answers are identical to it."""
        return self.rewards(answer)[-1], self.canon[tuple(sorted(answer))]


def _bisect_right(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def check_scorer(adjacency, difficulty, field, answer) -> None:
    """Compare with the validator's own (slow) calculator on a 30-answer sample of the field."""
    from CliqueAI.graph.model import LambdaGraph
    from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator

    field = field[:15] + [a for a in field if a == answer][:3] + field[-12:]

    graph = LambdaGraph(uuid="r", label="general", number_of_nodes=len(adjacency),
                        adjacency_list=adjacency)  # fmt: skip
    expected = CliqueScoreCalculator(graph, difficulty, field + [answer]).get_scores()[-1]
    got = RoundScorer(adjacency, difficulty, field).rewards(answer)
    worst = max((abs(a - b) for a, b in zip(expected, got)), default=0.0)
    if len(expected) != len(got) or worst > 1e-9:
        raise SystemExit(f"fast scorer disagrees with CliqueScoreCalculator (max diff {worst})")


# ------------------------------------------------------------------------------------- replay


def tier(n: int) -> str:
    return f"{round(n / 100) * 100}v"


def replay(row, args, rng: random.Random, check: bool) -> dict:
    from CliqueAI.clique_algorithms import fast_solver
    from CliqueAI.graph.codec import GraphCodec

    codec = GraphCodec()
    enc = row["encoded_matrix"]
    d, tl = float(row["difficulty"]), float(row["time_limit"])
    field = [[int(v) for v in (a or [])] for a in row["miner_ans"]]
    hotkeys = list(row.get("miner_hotkeys") or [None] * len(field))
    coldkeys = list(row.get("miner_coldkeys") or [None] * len(field))
    logged = [float(x) for x in (row.get("miner_rewards") or [])]
    adjacency = codec.matrix_to_list(codec.decode_matrix(enc))

    ours_at = hotkeys.index(args.hotkey) if args.hotkey and args.hotkey in hotkeys else None
    actual = None
    if ours_at is not None:  # our real answer this round; the counterfactual replaces it
        actual = {
            "size": len(field[ours_at]),
            "reward": logged[ours_at] if ours_at < len(logged) else None,
        }
        field = field[:ours_at] + field[ours_at + 1:]
        hotkeys = hotkeys[:ours_at] + hotkeys[ours_at + 1:]
        coldkeys = coldkeys[:ours_at] + coldkeys[ours_at + 1:]
        logged = logged[:ours_at] + logged[ours_at + 1:]
    scorer = RoundScorer(adjacency, d, field)
    if actual is not None:
        actual["identical"] = scorer.canon[tuple(sorted(row["miner_ans"][ours_at] or []))]

    budget = max(0.2, tl - args.safety_margin - 0.05 * tl - args.transit)
    res = fast_solver.solve(enc, budget, threads=args.threads, extra_args=["--near"])
    k = fast_solver.distinct_best(res)
    max_answer = fast_solver.choose_clique(res, "rarest", rng)
    near_answer, _ = fast_solver.choose_answer(res, "rarest", 10**9, rng)
    if check:
        check_scorer(adjacency, d, field, max_answer)
    max_reward, max_same = scorer.score(max_answer)
    near = None
    if len(near_answer) == res.best_size - 1 and res.near_cliques:
        near_reward, near_same = scorer.score(near_answer)
        near = {"reward": near_reward, "identical": near_same}

    sizes = scorer.sizes
    field_best = max(sizes, default=0)
    at_best = [tuple(sorted(a)) for a, s in zip(field, sizes) if s == field_best and s > 0]
    max_same_cold = sorted({str(c)[:8] for a, c in zip(field, coldkeys)
                            if tuple(sorted(a)) == tuple(sorted(max_answer))})  # fmt: skip
    by_cold = defaultdict(list)
    for c, r in zip(coldkeys, logged):
        by_cold[str(c)[:8]].append(r)
    return {
        "run": row.get("run"),
        "timestamp": row.get("timestamp"),
        "uuid": row.get("uuid"),
        "tier": tier(len(adjacency)),
        "n": len(adjacency),
        "difficulty": d,
        "time_limit": tl,
        "budget": round(budget, 3),
        "field_size": len(field),
        "field_best": field_best,
        "field_at_best": len(at_best),
        "field_distinct_at_best": len(set(at_best)),
        "field_one_below": sum(s == field_best - 1 for s in sizes),
        "field_invalid": sum(s == 0 for s in sizes),
        "ours_best": res.best_size,
        "k": k,
        "near_found": len(res.near_cliques),
        "time_to_best_ms": res.time_to_best_ms,
        "complete": res.complete,
        "max": {"reward": max_reward, "identical": max_same, "same_as_coldkeys": max_same_cold},
        "near": near,
        "actual": actual,
        "field_rewards": sorted(logged),
        "coldkey_rewards": {c: [sum(v), len(v)] for c, v in by_cold.items()},
    }


# ----------------------------------------------------------------------------------- analysis


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def policy_reward(r: dict, threshold: int) -> float:
    if r["near"] is not None and r["k"] <= threshold:
        return r["near"]["reward"]
    return r["max"]["reward"]


def analyse(results: list[dict]) -> None:
    n = len(results)
    print(f"\n=== {n} rounds replayed {hours_ago(results)}".rstrip())
    tiers = Counter(r["tier"] for r in results)
    print("tiers:", dict(sorted(tiers.items(), key=lambda t: int(t[0][:-1]))))
    beat = sum(r["ours_best"] > r["field_best"] for r in results)
    trail = sum(r["ours_best"] < r["field_best"] for r in results)
    print(f"our best size vs field best: beat {beat}, match {n - beat - trail}, trail {trail}")
    print(f"field reaching its best size: median {statistics.median(r['field_at_best'] for r in results)}"
          f" of {statistics.median(r['field_size'] for r in results)} miners")  # fmt: skip
    collide = sum(r["max"]["identical"] > 0 for r in results)
    print(f"our max-clique answer identical to someone else's: {collide}/{n} rounds")

    print("\nby distinct maximum cliques our search found (K):")
    print("   K        rounds  max_reward  near_reward  near_wins  max_shared  near_shared"
          "  field_at_best")
    for lo, hi in K_BUCKETS:
        rs = [r for r in results if lo <= r["k"] <= hi]
        if not rs:
            continue
        nr = [r for r in rs if r["near"] is not None]
        label = f"{lo}" if lo == hi else f"{lo}-{hi if hi < 10**9 else ''}"
        print(f"  {label:<9}{len(rs):>6}  {mean(r['max']['reward'] for r in rs):>10.3f}"
              f"  {mean(r['near']['reward'] for r in nr):>11.3f}"
              f"  {sum(r['near']['reward'] > r['max']['reward'] for r in nr):>4}/{len(nr):<4}"
              f"  {mean(r['max']['identical'] > 0 for r in rs):>10.2f}"
              f"  {mean(r['near']['identical'] > 0 for r in nr):>11.2f}"
              f"  {mean(r['field_at_best'] for r in rs):>13.1f}")  # fmt: skip

    # Hard rounds (best size found late) are where few miners reach it and "near" loses least.
    print("\nK <= 16, split by when our search first reached its best size:")
    for name, test in (("< 50 ms", lambda ms: ms < 50), (">= 50 ms", lambda ms: ms >= 50)):
        rs = [r for r in results if r["k"] <= 16 and r["near"] is not None
              and r["time_to_best_ms"] is not None and test(r["time_to_best_ms"])]  # fmt: skip
        if rs:
            print(f"  {name:>8}: rounds {len(rs)}, max {mean(r['max']['reward'] for r in rs):.3f}, "
                  f"near {mean(r['near']['reward'] for r in rs):.3f}, "
                  f"field_at_best {mean(r['field_at_best'] for r in rs):.1f}")  # fmt: skip

    print("\n--solver.near_max_cliques sweep (mean reward per round):")
    sweep = [(t, mean(policy_reward(r, t) for r in results)) for t in THRESHOLDS]
    best_t, best_v = max(sweep, key=lambda tv: (round(tv[1], 4), -tv[0]))
    for t, v in sweep:
        print(f"  {t:>4}: {v:.4f}{'   <- best' if t == best_t else ''}")
    oracle = mean(max(r["max"]["reward"], r["near"]["reward"] if r["near"] else 0) for r in results)
    print(f"  oracle (best answer every round): {oracle:.4f}")
    for t_name in sorted(tiers, key=lambda s: int(s[:-1])):
        rs = [r for r in results if r["tier"] == t_name]
        if len(rs) >= 10:
            tb = max(THRESHOLDS, key=lambda t: (round(mean(policy_reward(r, t) for r in rs), 4), -t))
            print(f"  {t_name}: best threshold {tb} ({mean(policy_reward(r, tb) for r in rs):.3f} vs "
                  f"{mean(policy_reward(r, 4) for r in rs):.3f} at 4, {len(rs)} rounds)")  # fmt: skip

    medians = [statistics.median(r["field_rewards"]) for r in results if r["field_rewards"]]
    if medians:
        print(f"\nfield median reward per round, averaged: {mean(medians):.3f}")
    cold = defaultdict(lambda: [0.0, 0])
    for r in results:
        for c, (s, cnt) in r["coldkey_rewards"].items():
            cold[c][0] += s
            cold[c][1] += cnt
    print("biggest coldkeys (answers in these rounds, mean logged reward):")
    for c, (s, cnt) in sorted(cold.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"  {c}  {cnt:>6}  {s / cnt:.3f}")
    with_actual = [r for r in results if r["actual"] and r["actual"]["reward"] is not None]
    if with_actual:
        print(f"\nour hotkey in {len(with_actual)} rounds: logged mean reward "
              f"{mean(r['actual']['reward'] for r in with_actual):.3f}, replayed max-policy "
              f"{mean(r['max']['reward'] for r in with_actual):.3f}, replayed with threshold 4 "
              f"{mean(policy_reward(r, 4) for r in with_actual):.3f}")  # fmt: skip


# --------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", default="toptensor-ai/CliqueAI")
    ap.add_argument("--rows", help="JSON-lines rows from scripts/fetch_rounds.py instead of W&B")
    ap.add_argument("--load", help="analyse a replay saved with --save (no solving)")
    ap.add_argument("--save", help="write one JSON line of features per replayed round")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--runs", type=int, default=12, help="most recently active W&B runs to read")
    ap.add_argument("--max-age-hours", type=float, default=48,
                    help="skip W&B runs that have not logged for this long")  # fmt: skip
    ap.add_argument("--hotkey", help="your miner hotkey: compare with its logged rewards")
    ap.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--safety-margin", type=float, default=1.0)
    ap.add_argument("--transit", type=float, default=0.3, help="seconds lost to the upload")
    ap.add_argument("--no-nice", action="store_true", help="do not lower this process's priority")
    args = ap.parse_args()

    if args.load:
        with open(args.load) as f:
            analyse([json.loads(line) for line in f if line.strip()])
        return
    if not args.no_nice:
        os.nice(19)  # a live miner on the same machine keeps the CPU when it needs it

    from CliqueAI.clique_algorithms import fast_solver

    fast_solver.ensure_built()
    rng = random.Random(0)
    source = (file_rows(args.rows, args.rounds) if args.rows
              else wandb_rows(args.project, args.rounds, args.runs, args.max_age_hours))  # fmt: skip
    out = open(args.save, "w") if args.save else None
    results = []
    started = time.time()
    try:
        for i, row in enumerate(source, 1):
            try:
                r = replay(row, args, rng, check=i <= 2)
            except SystemExit:
                raise
            except Exception as e:  # a malformed logged row must not end the replay
                print(f"[{i}] skipped: {type(e).__name__}: {e}", flush=True)
                continue
            results.append(r)
            if out:
                out.write(json.dumps(r) + "\n")
                out.flush()
            near = f"{r['near']['reward']:.2f}" if r["near"] else "-"
            print(f"[{i}] {r['tier']} limit={r['time_limit']}s ours={r['ours_best']} "
                  f"field_best={r['field_best']} ({r['field_at_best']}/{r['field_size']} reach it) "
                  f"K={r['k']} max={r['max']['reward']:.2f} (shared with {r['max']['identical']}) "
                  f"near={near} [{time.time() - started:.0f}s]", flush=True)  # fmt: skip
    finally:
        if out:
            out.close()
    if results:
        analyse(results)
    else:
        print("no rounds replayed")


if __name__ == "__main__":
    main()
