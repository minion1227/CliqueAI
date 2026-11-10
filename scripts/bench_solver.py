"""Benchmark the compiled clique solver on CliqueAI-sized graphs.

For each graph it runs a long multi-threaded reference search ("best known"), then repeated runs at
the budget a miner actually gets, and reports how often the short runs match the reference, how
fast they get there, and how many distinct best cliques they find (diversity score fuel).

    python scripts/bench_solver.py                      # synthetic tiers, default budgets
    python scripts/bench_solver.py --sizes 900 --budgets 4.7 --seeds 10
    python scripts/bench_solver.py --test-data          # bundled test_data graphs
    python scripts/bench_solver.py --baseline           # also time the stock networkx miner

Synthetic graphs are uniform G(n, p). The bundled test graphs have the clique numbers G(n, p)
theory predicts at their density (~0.735), so this is a fair stand-in for the "general" label.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from CliqueAI.clique_algorithms import fast_solver  # noqa: E402
from CliqueAI.graph.codec import GraphCodec  # noqa: E402

CODEC = GraphCodec()


def gnp_encoded(n: int, p: float, seed: int) -> str:
    rng = random.Random(seed)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        row = matrix[i]
        for j in range(i + 1, n):
            if rng.random() < p:
                row[j] = matrix[j][i] = 1
    return CODEC.encode_matrix(matrix)


def baseline_networkx(encoded: str) -> tuple[int, float]:
    from CliqueAI.clique_algorithms import networkx_algorithm

    adj = CODEC.matrix_to_list(CODEC.decode_matrix(encoded))
    start = time.monotonic()
    clique = networkx_algorithm(len(adj), adj)
    return len(clique), time.monotonic() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 500, 700, 900])
    ap.add_argument("--density", type=float, default=0.735)
    ap.add_argument("--graphs", type=int, default=2, help="graphs per size")
    ap.add_argument("--test-data", action="store_true", help="use test_data/*.json instead")
    # Miner budgets for the 6 s / 10 s / 30 s limits with the default 1 s + 5% safety margin.
    ap.add_argument("--budgets", type=float, nargs="+", default=[4.7, 8.5, 27.5])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--reference-time", type=float, default=30.0)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--schedule", default="", help="solver phase mix R,P,G (default: built-in)")
    args = ap.parse_args()
    extra = ["--schedule", args.schedule] if args.schedule else None

    fast_solver.ensure_built()
    if args.test_data:
        graphs = [
            (path.stem, json.loads(path.read_text())["encoded_matrix"])
            for path in sorted((ROOT / "test_data").glob("general_*.json"))
        ]
    else:
        graphs = [
            (f"G({n},{args.density})#{g}", gnp_encoded(n, args.density, seed=1000 * n + g))
            for n in args.sizes
            for g in range(args.graphs)
        ]

    print(f"threads={args.threads} seeds={args.seeds} reference={args.reference_time}s")
    header = f"{'graph':<20}{'ref':>5}{'budget':>8}{'hit':>7}{'sizes':>14}{'t_best ms':>22}{'distinct':>14}"
    print(header + ("   networkx" if args.baseline else ""))
    for name, encoded in graphs:
        ref = fast_solver.solve(
            encoded, args.reference_time, threads=args.threads, seed=12345, extra_args=extra
        )
        base = ""
        if args.baseline:
            size, secs = baseline_networkx(encoded)
            base = f"   {size} in {secs:.1f}s"
        for budget in args.budgets:
            runs = [
                fast_solver.solve(
                    encoded, budget, threads=args.threads, seed=seed, extra_args=extra
                )
                for seed in range(args.seeds)
            ]
            sizes = [r.best_size for r in runs]
            hits = sum(s >= ref.best_size for s in sizes)
            ttb = [r.time_to_best_ms for r in runs if r.time_to_best_ms is not None]
            distinct = [len(r.cliques) for r in runs]
            print(
                f"{name:<20}{ref.best_size:>5}{budget:>7.1f}s{hits:>4}/{len(runs):<2}"
                f"{min(sizes):>7}-{max(sizes):<6}"
                f"{statistics.median(ttb):>10.1f} (max {max(ttb):>7.0f})"
                f"{statistics.median(distinct):>8.0f} (ref {len(ref.cliques)})"
                + base
            )
            base = ""
        sys.stdout.flush()


if __name__ == "__main__":
    main()
