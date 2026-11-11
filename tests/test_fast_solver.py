"""Tests for the compiled clique solver (CliqueAI/clique_algorithms/fast_solver).

Run from the repo root:  python -m pytest tests/test_fast_solver.py
The first run compiles the solver (needs g++/clang++, or `pip install ziglang`).
"""

import asyncio
import json
import random
import subprocess
import time
from pathlib import Path

import pytest
from CliqueAI.clique_algorithms import fast_solver
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.graph.model import LambdaGraph
from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator

ROOT = Path(__file__).resolve().parents[1]
CODEC = GraphCodec()
# Best sizes known for the bundled graphs; 19 is proven optimal for the 90-node graph.
KNOWN_BEST = {"general_0.1": 19, "general_0.2": 22, "general_0.4": 25}


def load(name: str) -> str:
    return json.loads((ROOT / "test_data" / f"{name}.json").read_text())["encoded_matrix"]


def adjacency(encoded: str) -> list[list[int]]:
    return CODEC.matrix_to_list(CODEC.decode_matrix(encoded))


def random_encoded(n: int, p: float, seed: int) -> str:
    rng = random.Random(seed)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                matrix[i][j] = matrix[j][i] = 1
    return CODEC.encode_matrix(matrix)


def validator_accepts(encoded: str, clique: list[int]) -> bool:
    adj = adjacency(encoded)
    graph = LambdaGraph(uuid="t", label="general", number_of_nodes=len(adj), adjacency_list=adj)
    return CliqueScoreCalculator(graph, 0.8, [clique]).is_valid_maximum_clique(clique)


@pytest.fixture(scope="module", autouse=True)
def built():
    return fast_solver.ensure_built()


def dump(encoded: str) -> list[list[int]]:
    out = subprocess.run(
        [str(fast_solver.BINARY), "--mode", "dump"],
        input=encoded.encode(),
        capture_output=True,
        timeout=60,
        check=True,
    ).stdout.decode()
    return [[int(c) for c in line[2:]] for line in out.splitlines() if line.startswith("A ")]


@pytest.mark.parametrize(
    "encoded",
    [load("sample"), load("general_0.1"), load("general_0.2")]
    # 50*49/2 < 1246 < 51*50/2: sizes around chunk boundaries and word boundaries.
    + [random_encoded(n, p, n) for n, p in [(0, 0.5), (1, 0.5), (2, 1.0), (50, 0.7), (51, 0.7),
                                            (64, 0.3), (65, 0.9), (158, 0.735), (300, 0.5)]],
)
def test_decoder_matches_graph_codec(encoded):
    assert dump(encoded) == CODEC.decode_matrix(encoded)


@pytest.mark.parametrize("name", sorted(KNOWN_BEST))
def test_cliques_are_valid_and_large(name):
    encoded = load(name)
    result = fast_solver.solve(encoded, time_limit=1.0, threads=4)
    assert result.complete and result.error is None
    assert result.cliques, "no clique returned"
    assert len({tuple(c) for c in result.cliques}) == len(result.cliques), "duplicates reported"
    for clique in result.cliques:
        assert len(clique) == result.best_size
        assert validator_accepts(encoded, clique)
    # Local search reaches these in milliseconds; allow one vertex of slack for slow machines.
    assert result.best_size >= KNOWN_BEST[name] - 1


def test_respects_time_limit():
    start = time.monotonic()
    result = fast_solver.solve(load("general_0.4"), time_limit=1.0, threads=2)
    assert result.complete
    assert time.monotonic() - start < 1.0 + fast_solver.KILL_GRACE


def test_partial_result_survives_kill():
    encoded = load("general_0.4")
    result = fast_solver.solve(encoded, time_limit=30.0, threads=1, hard_timeout=0.7)
    assert not result.complete
    assert result.elapsed < 2.0
    assert len(result.cliques) == 1
    assert validator_accepts(encoded, result.cliques[0])


def test_solve_async_runs_requests_in_parallel():
    encoded = load("general_0.2")

    async def main():
        start = time.monotonic()
        results = await asyncio.gather(
            *[fast_solver.solve_async(encoded, 0.8, threads=1) for _ in range(3)]
        )
        return results, time.monotonic() - start

    results, wall = asyncio.run(main())
    assert wall < 0.8 * 2, "requests were serialised instead of running in parallel"
    for r in results:
        assert r.complete and r.cliques
        assert all(validator_accepts(encoded, c) for c in r.cliques[:5])


def test_cancelled_request_kills_solver():
    encoded = load("general_0.4")

    async def main():
        task = asyncio.create_task(fast_solver.solve_async(encoded, 29.5, threads=1))
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    leftover = subprocess.run(
        ["pgrep", "-f", "clique_solver --time 29.500"], capture_output=True, text=True
    ).stdout
    assert leftover.strip() == "", f"orphaned solver processes: {leftover}"


def test_choose_clique_policies():
    result = fast_solver.SolverResult(cliques=[[1, 2], [3, 4], [5, 6]], hits=[5, 1, 1])
    rng = random.Random(0)
    assert {tuple(fast_solver.choose_clique(result, "rarest", rng)) for _ in range(50)} == {
        (3, 4),
        (5, 6),
    }
    assert len({tuple(fast_solver.choose_clique(result, "random", rng)) for _ in range(200)}) == 3
    assert fast_solver.choose_clique(fast_solver.SolverResult()) == []


def test_greedy_fallback_is_valid():
    encoded = load("general_0.2")
    clique = fast_solver.greedy_maximal_clique(adjacency(encoded), random.Random(1))
    assert validator_accepts(encoded, clique)


def test_tiny_graphs():
    single = fast_solver.solve(random_encoded(1, 0.5, 0), time_limit=0.2)
    assert single.cliques == [[0]]
    triangle = fast_solver.solve(CODEC.encode_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), 0.2)
    assert triangle.cliques == [[0, 1, 2]]


def test_corrupt_input_reports_error():
    result = fast_solver.solve(load("general_0.2")[:-1], time_limit=0.5)
    assert result.error and "length mismatch" in result.error
    assert result.cliques == []


def test_zero_budget_still_answers():
    # Even if decoding uses the whole budget, one expansion runs: any valid answer beats none.
    encoded = load("general_0.4")
    result = fast_solver.solve(encoded, time_limit=0.0, threads=4)
    assert result.complete and result.cliques
    assert all(validator_accepts(encoded, c) for c in result.cliques)


def test_parse_drops_truncated_and_smaller_lines():
    text = "N 9\nB 3 0.1 0 1 2\nB 4 0.2 0 1 2\nR 1 0 1 2\nR 2 3 4 5 6\nR 1 3 4 5 7\nR 1 3 4"
    result = fast_solver.parse_output(text)  # killed mid-line: no trailing newline, no E
    assert not result.complete
    assert result.cliques == [[3, 4, 5, 6], [3, 4, 5, 7]]  # truncated + smaller ones dropped
    assert result.hits == [2, 1]
    assert result.time_to_best_ms is None  # B 4 claimed 4 vertices but listed 3
    partial = fast_solver.parse_output("N 9\nB 3 0.1 0 1 2\nB 5 0.2 1 2 3 4 5\nB 6 0.3 1 2")
    assert partial.cliques == [[1, 2, 3, 4, 5]] and partial.best_size == 5


def test_request_path_never_compiles(monkeypatch, tmp_path):
    monkeypatch.setattr(fast_solver, "BINARY", tmp_path / "missing_solver")
    start = time.monotonic()
    with pytest.raises(fast_solver.SolverError):
        asyncio.run(fast_solver.solve_async(load("general_0.1"), 1.0))
    assert time.monotonic() - start < 0.5


def test_self_test_passes():
    fast_solver.self_test()


def test_near_best_cliques_are_valid():
    encoded = load("general_0.2")
    result = fast_solver.solve(encoded, 1.0, threads=4, extra_args=["--near"])
    assert result.near_cliques, "--near reported no size best-1 cliques"
    assert all(len(c) == result.best_size - 1 for c in result.near_cliques)
    assert all(validator_accepts(encoded, c) for c in result.near_cliques[:20])
    assert not fast_solver.solve(encoded, 0.3, threads=2).near_cliques  # off by default


def test_choose_answer_near_rule():
    result = fast_solver.SolverResult(
        best_size=3,
        cliques=[[0, 1, 2], [0, 1, 3]],
        hits=[9, 4],
        near_cliques=[[4, 5], [6, 7], [8, 9]],
        near_hits=[3, 1, 1],
        stats={"distinct": "2"},
    )
    rng = random.Random(0)
    assert fast_solver.distinct_best(result) == 2
    picks = {tuple(fast_solver.choose_answer(result, "rarest", 2, rng)[0]) for _ in range(50)}
    assert picks == {(6, 7), (8, 9)}  # least-visited near cliques only
    assert fast_solver.choose_answer(result, "rarest", 2, rng)[1] == "near"
    assert fast_solver.choose_answer(result, "rarest", 1, rng) == ([0, 1, 3], "max")  # too many
    assert fast_solver.choose_answer(result, "rarest", 0, rng) == ([0, 1, 3], "max")  # disabled
    # More found than reported: the S line count decides, not the reported list.
    result.stats["distinct"] = "300"
    assert fast_solver.choose_answer(result, "rarest", 4, rng)[1] == "max"
    # Killed before the S line: fall back to the reported count.
    result.stats = {}
    assert fast_solver.choose_answer(result, "rarest", 2, rng)[1] == "near"
    # No near clique (e.g. complete graph, or killed before the Q lines): a maximum clique.
    result.near_cliques, result.near_hits = [], []
    assert fast_solver.choose_answer(result, "rarest", 4, rng) == ([0, 1, 3], "max")
    assert fast_solver.choose_answer(fast_solver.SolverResult(), "rarest", 4, rng) == ([], "none")


def test_near_answer_is_valid_and_one_smaller():
    encoded = random_encoded(300, 0.7, 300)
    result = fast_solver.solve(encoded, 1.0, threads=4, extra_args=["--near"])
    k = fast_solver.distinct_best(result)
    assert result.near_cliques and k >= 1
    clique, kind = fast_solver.choose_answer(result, "rarest", k, random.Random(1))
    assert kind == "near" and len(clique) == result.best_size - 1
    assert validator_accepts(encoded, clique)
    assert fast_solver.choose_answer(result, "rarest", k - 1, random.Random(1))[1] == "max"


def test_complete_graph_has_no_near_answer():
    encoded = random_encoded(12, 1.0, 0)  # K12: the only maximal clique is everything
    result = fast_solver.solve(encoded, 0.2, threads=2, extra_args=["--near"])
    assert result.cliques == [list(range(12))] and result.near_cliques == []
    assert fast_solver.choose_answer(result, "rarest", 4) == (list(range(12)), "max")


def test_miner_defaults_and_budget():
    import argparse
    import types

    from CliqueAI.miner import MAX_TRANSIT, Miner
    from CliqueAI.protocol import MaximumCliqueOfLambdaGraph

    parser = argparse.ArgumentParser()
    Miner.add_args(parser)
    args = vars(parser.parse_args([]))
    assert args["neuron.autoupdate"] == 0, "a modified miner must not auto-update"
    assert args["solver.pick"] == "rarest"
    assert args["solver.near_max_cliques"] == 4

    miner = Miner.__new__(Miner)
    miner.config = types.SimpleNamespace(solver=types.SimpleNamespace(safety_margin=1.0))
    syn = MaximumCliqueOfLambdaGraph(timeout=6.0)
    full = 6.0 - 1.0 - 0.05 * 6.0
    assert miner.time_budget(syn) == pytest.approx(full, abs=0.05)  # no nonce
    syn.dendrite.nonce = time.time_ns() - int(0.4e9)  # 0.4 s in transit: credited
    assert miner.time_budget(syn) == pytest.approx(full - 0.4, abs=0.05)
    syn.dendrite.nonce = time.time_ns() - int(4e9)  # validator clock 4 s behind: capped
    assert miner.time_budget(syn) == pytest.approx(full - MAX_TRANSIT, abs=0.05)
    syn.dendrite.nonce = time.time_ns() + int(3e9)  # validator clock ahead: ignored
    assert miner.time_budget(syn) == pytest.approx(full, abs=0.05)
