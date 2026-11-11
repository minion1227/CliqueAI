"""Fast compiled maximum-clique solver for CliqueAI miners.

The solver is a standalone C++ program (``clique_solver.cpp``). It is compiled on first use with
whatever C++ compiler is available: ``$CLIQUE_SOLVER_CXX``, ``g++``, ``clang++``, ``c++``, or the
pip-installable ``ziglang`` toolchain, so no root access is needed.

Running it as a subprocess keeps the miner's asyncio event loop free while it searches, lets
overlapping validator requests solve in parallel on different cores, and lets us hard-kill a run
that overshoots its deadline while still keeping the best clique it had printed.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

_DIR = Path(__file__).resolve().parent
SOURCE = _DIR / "clique_solver.cpp"
BINARY = _DIR / "clique_solver"
CXXFLAGS = ["-O3", "-std=c++17", "-pthread"]
NATIVE_FLAGS = ["-march=native"]

# Extra wall-clock time the process gets past its own time limit before it is killed.
KILL_GRACE = 0.3


class SolverError(RuntimeError):
    pass


@dataclass
class SolverResult:
    n: int = 0
    best_size: int = 0
    cliques: list[list[int]] = field(default_factory=list)  # distinct best cliques, rarest first
    hits: list[int] = field(default_factory=list)  # visits per clique (index-aligned)
    near_cliques: list[list[int]] = field(default_factory=list)  # --near: size best - 1
    near_hits: list[int] = field(default_factory=list)
    stats: dict[str, str] = field(default_factory=dict)
    complete: bool = False  # solver exited normally (False: killed, partial output)
    elapsed: float = 0.0
    time_to_best_ms: float | None = None  # when the best size was first reached
    error: str | None = None


# --------------------------------------------------------------------------------------- build


def _compilers() -> list[list[str]]:
    env = os.environ.get("CLIQUE_SOLVER_CXX")
    if env:
        return [env.split()]
    found = [[path] for name in ("g++", "clang++", "c++") if (path := shutil.which(name))]
    try:
        import ziglang  # noqa: F401

        found.append([sys.executable, "-m", "ziglang", "c++"])
    except ImportError:
        pass
    return found


def _is_fresh() -> bool:
    return BINARY.exists() and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime


def ensure_built(force: bool = False) -> Path:
    """Compile the solver if the binary is missing or older than its source. Returns its path."""
    if not force and _is_fresh():
        return BINARY
    with open(_DIR / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # several miner processes may start at once
        if not force and _is_fresh():
            return BINARY
        compilers = _compilers()
        if not compilers:
            raise SolverError(
                "No C++ compiler found. Install one (`sudo apt install g++`) or "
                "`pip install ziglang`, or set CLIQUE_SOLVER_CXX."
            )
        errors = []
        for compiler in compilers:
            for flags in (CXXFLAGS + NATIVE_FLAGS, CXXFLAGS):
                fd, tmp = tempfile.mkstemp(prefix=".clique_solver.", dir=_DIR)
                os.close(fd)
                cmd = [*compiler, *flags, "-o", tmp, str(SOURCE)]
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                except (OSError, subprocess.TimeoutExpired) as e:
                    errors.append(f"{' '.join(cmd)}: {e}")
                    os.unlink(tmp)
                    continue
                if proc.returncode == 0:
                    os.chmod(tmp, 0o755)
                    os.replace(tmp, BINARY)
                    return BINARY
                os.unlink(tmp)
                errors.append(f"{' '.join(cmd)}:\n{proc.stderr[-1500:]}")
        raise SolverError("Compiling clique_solver failed:\n" + "\n".join(errors))


def binary() -> Path:
    """The built solver, for the request path: never compiles, never blocks."""
    if not BINARY.exists():
        raise SolverError(f"solver binary missing: {BINARY} (run fast_solver.ensure_built())")
    return BINARY


# K4 minus one edge: maximum cliques are exactly {0,1,2} and {0,1,3}.
_SELF_TEST_MATRIX = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]


def self_test() -> None:
    """Run the binary on a known graph; raises SolverError if it is broken (e.g. SIGILL from a
    -march=native build copied to another CPU)."""
    from CliqueAI.graph.codec import GraphCodec

    encoded = GraphCodec().encode_matrix(_SELF_TEST_MATRIX)
    result = solve(encoded, time_limit=0.05, threads=2, hard_timeout=10.0)
    found = sorted(map(tuple, result.cliques))
    if not result.complete or result.error or found != [(0, 1, 2), (0, 1, 3)]:
        raise SolverError(
            f"solver self-test failed: complete={result.complete} error={result.error} "
            f"cliques={found}"
        )


# ----------------------------------------------------------------------------------------- run


def _command(time_limit: float, threads: int, seed: int | None, report: int, extra: list[str]):
    seed = random.getrandbits(63) if seed is None else seed
    return [
        str(binary()),
        "--time", f"{max(0.0, time_limit):.3f}",
        "--threads", str(max(1, threads)),
        "--seed", str(seed),
        "--report", str(report),
        *extra,
    ]  # fmt: skip


def parse_output(text: str) -> SolverResult:
    """Parse the solver's line protocol. Falls back to the best `B` line if output is partial.

    A process killed mid-write can leave a truncated last line, which would be a non-maximal
    "clique" scoring 0. So unterminated lines are dropped, B lines must match their declared
    size, and only the largest R cliques are kept.
    """
    result = SolverResult()
    best_partial: list[int] = []
    first_seen_ms: dict[int, float] = {}
    lines = text.split("\n")[:-1]  # the last element is "" or an unterminated fragment
    for line in lines:
        tag, _, rest = line.partition(" ")
        try:
            if tag == "N":
                result.n = int(rest)
            elif tag == "B":
                parts = rest.split()
                size, ms = int(parts[0]), float(parts[1])
                vertices = [int(v) for v in parts[2:]]
                if len(vertices) != size:
                    continue
                first_seen_ms[size] = min(ms, first_seen_ms.get(size, ms))
                if size > len(best_partial):
                    best_partial = vertices
            elif tag in ("R", "Q"):
                parts = [int(v) for v in rest.split()]
                if len(parts) >= 2:
                    hits, cliques = (
                        (result.hits, result.cliques)
                        if tag == "R"
                        else (result.near_hits, result.near_cliques)
                    )
                    hits.append(parts[0])
                    cliques.append(parts[1:])
            elif tag == "S":
                for item in rest.split():
                    key, _, value = item.partition("=")
                    result.stats[key] = value
            elif tag == "E":
                result.complete = True
            elif tag == "X":
                result.error = rest
        except (ValueError, IndexError):
            continue  # malformed line: ignore rather than lose the whole answer
    if result.cliques:
        best = max(len(c) for c in result.cliques)
        kept = [(c, h) for c, h in zip(result.cliques, result.hits) if len(c) == best]
        result.cliques, result.hits = [c for c, _ in kept], [h for _, h in kept]
    if not result.cliques and best_partial:
        result.cliques, result.hits = [best_partial], [1]
    result.best_size = max((len(c) for c in result.cliques), default=0)
    result.time_to_best_ms = first_seen_ms.get(result.best_size)
    kept_near = [
        (c, h)
        for c, h in zip(result.near_cliques, result.near_hits)
        if len(c) == result.best_size - 1
    ]
    result.near_cliques = [c for c, _ in kept_near]
    result.near_hits = [h for _, h in kept_near]
    return result


def solve(
    encoded_matrix: str,
    time_limit: float,
    threads: int = 1,
    seed: int | None = None,
    report: int = 256,
    extra_args: list[str] | None = None,
    hard_timeout: float | None = None,
) -> SolverResult:
    """Blocking solve (tests, benchmarks). `hard_timeout` defaults to time_limit + KILL_GRACE."""
    cmd = _command(time_limit, threads, seed, report, extra_args or [])
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=encoded_matrix.encode("ascii"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=hard_timeout if hard_timeout is not None else time_limit + KILL_GRACE,
        )
        out = proc.stdout
    except subprocess.TimeoutExpired as e:  # killed: keep whatever it printed
        out = e.stdout or b""
    result = parse_output(out.decode("ascii", "replace"))
    result.elapsed = time.monotonic() - start
    return result


async def solve_async(
    encoded_matrix: str,
    time_limit: float,
    threads: int = 1,
    seed: int | None = None,
    report: int = 256,
    extra_args: list[str] | None = None,
    hard_timeout: float | None = None,
) -> SolverResult:
    """Non-blocking solve for the miner's event loop; kills the process at `hard_timeout`."""
    cmd = _command(time_limit, threads, seed, report, extra_args or [])
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    lines: list[bytes] = []

    async def communicate():
        try:
            proc.stdin.write(encoded_matrix.encode("ascii"))
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass  # solver exited early (error); its X line is still readable
        async for line in proc.stdout:
            lines.append(line)

    try:
        await asyncio.wait_for(
            communicate(), hard_timeout if hard_timeout is not None else time_limit + KILL_GRACE
        )
        await proc.wait()
    except asyncio.TimeoutError:
        pass  # keep the lines read so far; the process is killed below
    finally:
        if proc.returncode is None:  # timed out, or the request was cancelled: no orphans
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
    result = parse_output(b"".join(lines).decode("ascii", "replace"))
    result.elapsed = time.monotonic() - start
    return result


# ---------------------------------------------------------------------------------- selection


def choose_clique(
    result: SolverResult, policy: str = "rarest", rng: random.Random | None = None
) -> list[int]:
    """Pick which best-size clique to answer with.

    CliqueAI splits the diversity score between miners returning the identical clique, so we
    avoid the cliques our own search keeps landing on. Those have the biggest basins of
    attraction and are the likeliest to be found by other miners too.
    "rarest": uniform among the least-visited; "random": uniform among all.
    """
    rng = rng or random
    if not result.cliques:
        return []
    if policy == "random":
        return list(rng.choice(result.cliques))
    fewest = min(result.hits)
    return list(rng.choice([c for c, h in zip(result.cliques, result.hits) if h == fewest]))


def distinct_best(result: SolverResult) -> int:
    """How many distinct maximum cliques the search found (more than it may have reported)."""
    try:
        return max(int(result.stats["distinct"]), len(result.cliques))
    except (KeyError, ValueError):
        return len(result.cliques)  # killed before the S line


def choose_answer(
    result: SolverResult,
    pick: str = "rarest",
    near_max_cliques: int = 0,
    rng: random.Random | None = None,
) -> tuple[list[int], str]:
    """Pick the answer to send: (clique, "max" | "near" | "none").

    CliqueAI's reward is optimality * (1 + difficulty) + diversity, and the diversity part is split
    between miners returning the identical clique. When a graph has only a few maximum cliques,
    the other ~250 miners cover all of them, so a maximum clique shares its diversity with at least
    one more miner. A maximal clique one vertex smaller is almost always unique; it keeps the full
    diversity point and loses little optimality when few miners reach the maximum. A replay of
    logged validator rounds (Nov 2026) favoured it when the search found at most 4 maximum
    cliques; re-tune with scripts/replay_rounds.py.

    `near_max_cliques` (0 = off) is that threshold. Near cliques come from the solver's `--near`
    run, are verified maximal by it, and the least visited one is chosen (rarest = least likely to
    be found by anyone else). Without one, or above the threshold, this returns a maximum clique.
    """
    rng = rng or random
    if (
        near_max_cliques > 0
        and result.cliques
        and result.near_cliques
        and distinct_best(result) <= near_max_cliques
    ):
        fewest = min(result.near_hits)
        rarest = [c for c, h in zip(result.near_cliques, result.near_hits) if h == fewest]
        return list(rng.choice(rarest)), "near"
    clique = choose_clique(result, pick, rng)
    return clique, ("max" if clique else "none")


def greedy_maximal_clique(
    adjacency_list: list[list[int]], rng: random.Random | None = None
) -> list[int]:
    """Pure-Python fallback: a quick greedy maximal clique (valid for the validator, not big)."""
    rng = rng or random
    neighbours = [set(a) for a in adjacency_list]
    candidates = set(range(len(adjacency_list)))
    clique: list[int] = []
    while candidates:
        v = max(candidates, key=lambda u: (len(neighbours[u] & candidates), rng.random()))
        clique.append(v)
        candidates &= neighbours[v]
    return sorted(clique)
