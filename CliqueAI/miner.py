import asyncio
import os
import random
import re
import time
import typing

import bittensor as bt
from CliqueAI.clique_algorithms import fast_solver
from CliqueAI.commit_reveal import CommitRevealResponder
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph, MaximumCliqueSealedGraph
from common.base.miner import BaseMinerNeuron

# Most upload + axon-verification time credited from the validator's nonce (seconds). Caps the
# damage when a validator's clock runs behind ours.
MAX_TRANSIT = 1.5
MIN_BUDGET = 0.2

Challenge = typing.Union[MaximumCliqueOfLambdaGraph, MaximumCliqueSealedGraph]


class Miner(BaseMinerNeuron):
    """
    CliqueAI miner backed by the compiled local-search solver in
    CliqueAI/clique_algorithms/fast_solver.

    Each request runs the solver in its own subprocess until just before the validator's
    timeout, so the asyncio event loop stays free and overlapping requests from different
    validators solve in parallel. Answers both the current protocol (MaximumCliqueOfLambdaGraph)
    and upstream v0.0.18's commit-reveal one (MaximumCliqueSealedGraph), so the miner keeps
    scoring when validators upgrade. See fast_solver.choose_answer for which clique is returned.
    """

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        # Auto-update restarts the miner whenever HEAD differs from origin/main, which is always
        # true on this branch: with it on, the miner restart-loops. Pass --neuron.autoupdate 1
        # only when running unmodified upstream code.
        parser.set_defaults(**{"neuron.autoupdate": 0})
        parser.add_argument(
            "--solver.threads",
            type=int,
            default=0,
            help="Solver threads per request. 0 = all CPU cores, split between overlapping requests.",
        )
        parser.add_argument(
            "--solver.safety_margin",
            type=float,
            default=1.0,
            help="Seconds kept in reserve before the validator's timeout (plus 5%% of the timeout) "
            "for sending the answer back.",
        )
        parser.add_argument(
            "--solver.pick",
            type=str,
            choices=["rarest", "random"],
            default="rarest",
            help="Which of the distinct maximum cliques found to return.",
        )
        parser.add_argument(
            "--solver.near_max_cliques",
            type=int,
            default=4,
            help="Return a (practically unique) maximal clique one vertex smaller when the search "
            "finds at most this many distinct maximum cliques: those are shared with other "
            "miners, which splits the diversity reward. 0 = always return a maximum clique.",
        )
        parser.add_argument(
            "--solver.schedule",
            type=str,
            default="",
            help="Local-search phase mix 'random,penalty,greedy' (iterations per cycle). "
            "Empty = solver default (random only, best on CliqueAI's random graphs). "
            "Benchmark changes with scripts/bench_solver.py.",
        )

    def __init__(self, config=None):
        super().__init__(config=config)
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.blacklist_graph,
            priority_fn=self.priority_graph,
            verify_fn=self.verify_graph,
        )
        self.commit_reveal = CommitRevealResponder(solve=self.solve_revealed)
        self.axon.attach(
            forward_fn=self.forward_sealed_graph,
            blacklist_fn=self.blacklist_sealed_graph,
            priority_fn=self.priority_sealed_graph,
            verify_fn=self.verify_sealed_graph,
        )
        schedule = self.config.solver.schedule
        if schedule and (
            not re.fullmatch(r"\d+,\d+,\d+", schedule)
            or not any(int(x) for x in schedule.split(","))
        ):
            raise ValueError(
                f"--solver.schedule must look like 'R,P,G' with a non-zero phase (e.g. 1,0,0), "
                f"got {schedule!r}"
            )
        if self.config.solver.near_max_cliques < 0:
            raise ValueError("--solver.near_max_cliques must be >= 0")
        if self.config.neuron.autoupdate:
            bt.logging.warning(
                "Auto-update is on: the miner restarts whenever HEAD differs from origin/main."
            )
        self.rng = random.Random()
        self.cpu_count = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        ) or 1
        self.active_requests = 0
        # Fail fast: a miner stuck on the greedy fallback looks alive but scores near zero.
        solver_path = fast_solver.ensure_built()
        fast_solver.self_test()
        bt.logging.info(f"Clique solver ready and self-tested: {solver_path}")

    # ------------------------------------------------------------------------------- budgets

    def time_budget(self, synapse: Challenge) -> float:
        """Seconds the solver may run so the answer reaches the validator before its timeout."""
        now = time.time()
        timeout = float(synapse.timeout or 12.0)
        time_limit = float(getattr(synapse, "time_limit", 0.0) or 0.0)
        if 0 < time_limit < timeout:  # sealed reveal: the stated limit, if it is the tighter one
            timeout = time_limit
        start = now
        nonce = synapse.dendrite.nonce if synapse.dendrite is not None else None
        if nonce:
            # The nonce is the validator's time.time_ns() just before sending. Credit the upload
            # and axon verification already spent, capped so clock skew can't eat the budget.
            in_transit = now - nonce / 1e9
            if in_transit > 0:
                start = now - min(in_transit, MAX_TRANSIT)
        margin = self.config.solver.safety_margin + 0.05 * timeout
        return max(MIN_BUDGET, start + timeout - margin - now)

    def solver_threads(self) -> int:
        if self.config.solver.threads > 0:
            return self.config.solver.threads
        return max(1, self.cpu_count // max(1, self.active_requests))

    def caller(self, synapse: bt.Synapse) -> str:
        hotkey = synapse.dendrite.hotkey if synapse.dendrite is not None else None
        if not hotkey:
            return "?"
        try:
            return f"uid{self.metagraph.hotkeys.index(hotkey)}"
        except (ValueError, AttributeError):
            return hotkey[:8]

    # ------------------------------------------------------------------------------ solving

    async def answer(self, synapse: Challenge, encoded_matrix: str) -> list[int]:
        """Solve one challenge within its time budget and log the outcome."""
        started = time.time()
        budget = self.time_budget(synapse)
        self.active_requests += 1
        try:
            threads = self.solver_threads()
            clique, detail = await self.solve(encoded_matrix, budget, threads)
        finally:
            self.active_requests -= 1
        bt.logging.info(
            f"Maximum clique found: size={len(clique)} {detail} "
            f"validator={self.caller(synapse)} protocol={type(synapse).__name__} "
            f"timeout={synapse.timeout}s budget={budget:.2f}s threads={threads} "
            f"took={time.time() - started:.2f}s uuid={synapse.uuid}"
        )
        return clique

    async def solve(self, encoded_matrix: str, budget: float, threads: int) -> tuple[list[int], str]:
        schedule = self.config.solver.schedule
        near_max = self.config.solver.near_max_cliques
        extra = ["--near"] if near_max > 0 else []
        if schedule:
            extra += ["--schedule", schedule]
        try:
            result = await fast_solver.solve_async(
                encoded_matrix, budget, threads=threads, extra_args=extra
            )
            clique, kind = fast_solver.choose_answer(
                result, self.config.solver.pick, near_max, self.rng
            )
            if clique:
                return clique, (
                    f"answer={kind} best={result.best_size} n={result.n} "
                    f"distinct_best={fast_solver.distinct_best(result)} "
                    f"near_found={len(result.near_cliques)} "
                    f"time_to_best={result.time_to_best_ms or 0:.0f}ms "
                    f"iterations={result.stats.get('iterations', '?')} complete={result.complete}"
                )
            reason = result.error or "solver returned no clique"
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
        bt.logging.warning(f"Clique solver failed ({reason}); using Python greedy fallback.")
        try:
            clique = await asyncio.to_thread(self.fallback_clique, encoded_matrix)
        except Exception as e:  # undecodable graph: nothing valid to send
            bt.logging.error(f"Greedy fallback failed: {type(e).__name__}: {e}")
            return [], "answer=none"
        return clique, "answer=fallback"

    def fallback_clique(self, encoded_matrix: str) -> list[int]:
        codec = GraphCodec()
        adjacency_list = codec.matrix_to_list(codec.decode_matrix(encoded_matrix))
        return fast_solver.greedy_maximal_clique(adjacency_list, self.rng)

    # ------------------------------------------------- protocol: MaximumCliqueOfLambdaGraph

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        synapse.maximum_clique = await self.answer(synapse, synapse.encoded_matrix)
        # The validator keeps its own copy of the graph; echoing ~62 KB back only slows the
        # response down (and validators reject responses over 100 KB).
        synapse.encoded_matrix = ""
        return synapse

    async def blacklist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)

    async def verify_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> None:
        await self.verify(synapse)

    # ------------------------------------------ protocol: MaximumCliqueSealedGraph (v0.0.18)

    async def forward_sealed_graph(
        self, synapse: MaximumCliqueSealedGraph
    ) -> MaximumCliqueSealedGraph:
        return await self.commit_reveal.handle(synapse)

    async def solve_revealed(
        self, synapse: MaximumCliqueSealedGraph, encoded_matrix: str
    ) -> list[int]:
        return await self.answer(synapse, encoded_matrix)

    async def blacklist_sealed_graph(
        self, synapse: MaximumCliqueSealedGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_sealed_graph(self, synapse: MaximumCliqueSealedGraph) -> float:
        return await self.priority(synapse)

    async def verify_sealed_graph(self, synapse: MaximumCliqueSealedGraph) -> None:
        await self.verify(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Miner has started running.")
        while True:
            if miner.should_exit:
                bt.logging.info("Miner is exiting.")
                break
            time.sleep(1)
