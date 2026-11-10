import asyncio
import os
import random
import re
import time
import typing

import bittensor as bt
from CliqueAI.clique_algorithms import fast_solver
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron

# Longest the validator's clock may plausibly run ahead of ours in the request nonce (seconds).
MAX_NONCE_AGE = 5.0
MIN_BUDGET = 0.2


class Miner(BaseMinerNeuron):
    """
    CliqueAI miner backed by the compiled local-search solver in
    CliqueAI/clique_algorithms/fast_solver.

    Each request runs the solver in its own subprocess until just before the validator's
    timeout, so the asyncio event loop stays free and overlapping requests from different
    validators solve in parallel. Among the distinct maximum cliques found it returns a
    rarely-visited one, because CliqueAI shares the diversity score between identical answers.
    """

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
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
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )
        schedule = self.config.solver.schedule
        if schedule and not re.fullmatch(r"\d+,\d+,\d+", schedule):
            raise ValueError(f"--solver.schedule must look like 'R,P,G' (e.g. 1,0,0), got {schedule!r}")
        self.rng = random.Random()
        self.cpu_count = os.cpu_count() or 1
        self.active_requests = 0
        try:
            bt.logging.info(f"Clique solver ready: {fast_solver.ensure_built()}")
        except Exception as e:
            bt.logging.error(f"Clique solver unavailable, answers will come from the slow fallback: {e}")

    def time_budget(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        """Seconds the solver may run so the answer reaches the validator before its timeout."""
        now = time.time()
        timeout = float(synapse.timeout or 12.0)
        start = now
        nonce = synapse.dendrite.nonce if synapse.dendrite is not None else None
        if nonce:
            sent = nonce / 1e9  # validator's time.time_ns() just before sending
            if now - MAX_NONCE_AGE <= sent <= now:
                start = sent  # account for upload and axon verification time already spent
        margin = self.config.solver.safety_margin + 0.05 * timeout
        return max(MIN_BUDGET, start + timeout - margin - now)

    def solver_threads(self) -> int:
        if self.config.solver.threads > 0:
            return self.config.solver.threads
        return max(1, self.cpu_count // max(1, self.active_requests))

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        started = time.time()
        budget = self.time_budget(synapse)
        self.active_requests += 1
        try:
            threads = self.solver_threads()
            clique, detail = await self.solve(synapse, budget, threads)
        finally:
            self.active_requests -= 1
        synapse.maximum_clique = clique
        bt.logging.info(
            f"Maximum clique found: size={len(clique)} n={synapse.number_of_nodes} "
            f"timeout={synapse.timeout}s budget={budget:.2f}s threads={threads} "
            f"took={time.time() - started:.2f}s {detail} uuid={synapse.uuid}"
        )
        return synapse

    async def solve(
        self, synapse: MaximumCliqueOfLambdaGraph, budget: float, threads: int
    ) -> tuple[list[int], str]:
        schedule = self.config.solver.schedule
        try:
            result = await fast_solver.solve_async(
                synapse.encoded_matrix,
                budget,
                threads=threads,
                extra_args=["--schedule", schedule] if schedule else None,
            )
            clique = fast_solver.choose_clique(result, self.config.solver.pick, self.rng)
            if clique:
                return clique, (
                    f"distinct_best={len(result.cliques)} "
                    f"iterations={result.stats.get('iterations', '?')} complete={result.complete}"
                )
            reason = result.error or "solver returned no clique"
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
        bt.logging.warning(f"Clique solver failed ({reason}); using Python greedy fallback.")
        clique = await asyncio.to_thread(self.fallback_clique, synapse.encoded_matrix)
        return clique, "fallback=greedy"

    def fallback_clique(self, encoded_matrix: str) -> list[int]:
        codec = GraphCodec()
        adjacency_list = codec.matrix_to_list(codec.decode_matrix(encoded_matrix))
        return fast_solver.greedy_maximal_clique(adjacency_list, self.rng)

    async def backlist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Miner has started running.")
        while True:
            if miner.should_exit:
                bt.logging.info("Miner is exiting.")
                break
            time.sleep(1)
