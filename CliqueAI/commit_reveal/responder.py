"""Miner side of CliqueAI's commit-reveal challenges (MaximumCliqueSealedGraph, upstream v0.0.18).

A validator first sends the graph sealed (commit), then the key and the time limit (reveal); the
time limit only starts at the reveal. Adapted from upstream's CliqueAI/commit_reveal/responder.py
with the same checks, pending table, per-hotkey quota and TTL. Differences:

- The unsealed graph goes to `solve` still base92-encoded. The compiled solver decodes a 900-node
  graph in milliseconds; upstream decodes it in Python first, which takes ~0.4 s of the limit.
- `solve(synapse, encoded_matrix)` owns the time budget: it sees the reveal's nonce and timeout,
  so the same budget rules apply to both protocols.
"""

import math
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Awaitable, Callable

import bittensor as bt
from CliqueAI.commit_reveal.sealer import SealError, unseal
from CliqueAI.graph.codec import DEFAULT_BASE92_ALPHABET
from CliqueAI.protocol import MAX_SEALED_CHARS, MaximumCliqueSealedGraph

DEFAULT_MAX_PENDING = 64
DEFAULT_MAX_PENDING_PER_HOTKEY = 10
DEFAULT_PENDING_TTL = 120.0

_BASE92 = frozenset(DEFAULT_BASE92_ALPHABET)
_HEX = frozenset("0123456789abcdef")

Solve = Callable[[MaximumCliqueSealedGraph, str], Awaitable[list[int]]]


def _is_hex64(value: str) -> bool:
    return len(value) == 64 and set(value) <= _HEX


@dataclass
class PendingChallenge:
    sealed_matrix: str
    digest: str
    created_at: float


class CommitRevealResponder:
    """Store sealed challenges per validator hotkey and solve them after the reveal.

    Pending commits are keyed by (validator hotkey, uuid), so two validators using the same uuid
    cannot collide, and a reveal only opens a commit made by the same signature-verified hotkey.
    A hotkey over its quota displaces its own oldest pending commit, never another hotkey's.
    """

    def __init__(
        self,
        solve: Solve,
        max_sealed_chars: int = MAX_SEALED_CHARS,
        max_pending: int = DEFAULT_MAX_PENDING,
        max_pending_per_hotkey: int = DEFAULT_MAX_PENDING_PER_HOTKEY,
        pending_ttl: float = DEFAULT_PENDING_TTL,
    ):
        if not 0 < max_sealed_chars <= MAX_SEALED_CHARS:
            raise ValueError(f"max_sealed_chars must be between 1 and {MAX_SEALED_CHARS}")
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if max_pending_per_hotkey <= 0:
            raise ValueError("max_pending_per_hotkey must be positive")
        if not math.isfinite(pending_ttl) or pending_ttl <= 0:
            raise ValueError("pending_ttl must be positive and finite")
        self.solve = solve
        self.max_sealed_chars = max_sealed_chars
        self.max_pending = max_pending
        self.max_pending_per_hotkey = max_pending_per_hotkey
        self.pending_ttl = pending_ttl
        self._pending: OrderedDict[tuple[str, str], PendingChallenge] = OrderedDict()
        self._pending_per_hotkey: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._pending)

    async def handle(self, synapse: MaximumCliqueSealedGraph) -> MaximumCliqueSealedGraph:
        synapse.maximum_clique = []
        self._prune()
        if synapse.sealed_matrix and synapse.key:
            self._clear(synapse)
            bt.logging.warning(f"Challenge {synapse.uuid}: invalid commit-reveal phase")
            return synapse
        if synapse.sealed_matrix:
            return self._commit(synapse)
        if synapse.key:
            return await self._reveal(synapse)
        bt.logging.warning(f"Challenge {synapse.uuid}: missing commit and reveal data")
        return synapse

    @staticmethod
    def _clear(synapse: MaximumCliqueSealedGraph) -> None:
        # Nothing the validator sent needs to travel back; keeps responses small.
        synapse.sealed_matrix = ""
        synapse.digest = ""
        synapse.key = ""

    def _commit(self, synapse: MaximumCliqueSealedGraph) -> MaximumCliqueSealedGraph:
        payload, digest = synapse.sealed_matrix, synapse.digest
        self._clear(synapse)
        hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not hotkey:
            bt.logging.warning(f"Rejecting commit {synapse.uuid}: missing dendrite hotkey")
            return synapse
        if not synapse.uuid or len(synapse.uuid) > 64:
            bt.logging.warning("Rejecting commit: invalid uuid")
            return synapse
        if len(payload) > self.max_sealed_chars:
            bt.logging.warning(
                f"Rejecting commit {synapse.uuid}: sealed_matrix too large, "
                f"{len(payload)} chars > {self.max_sealed_chars}"
            )
            return synapse
        if not set(payload) <= _BASE92:
            bt.logging.warning(f"Rejecting commit {synapse.uuid}: invalid sealed_matrix")
            return synapse
        if not _is_hex64(digest):
            bt.logging.warning(f"Rejecting commit {synapse.uuid}: invalid digest")
            return synapse

        pending_key = (hotkey, synapse.uuid)
        if pending_key in self._pending:
            bt.logging.warning(f"Rejecting commit {synapse.uuid}: already committed")
            return synapse
        if self._pending_per_hotkey[hotkey] >= self.max_pending_per_hotkey:
            self._evict_oldest(hotkey)
        if len(self._pending) >= self.max_pending:
            bt.logging.warning(f"Rejecting commit {synapse.uuid}: pending limit reached")
            return synapse
        self._pending[pending_key] = PendingChallenge(payload, digest, time.monotonic())
        self._pending_per_hotkey[hotkey] += 1
        bt.logging.info(f"Committed challenge {synapse.uuid} from {hotkey} ({len(payload)} chars)")
        return synapse

    async def _reveal(self, synapse: MaximumCliqueSealedGraph) -> MaximumCliqueSealedGraph:
        key = synapse.key
        self._clear(synapse)
        hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not hotkey:
            bt.logging.warning(f"Rejecting reveal {synapse.uuid}: missing dendrite hotkey")
            return synapse
        if not synapse.uuid or len(synapse.uuid) > 64:
            bt.logging.warning("Rejecting reveal: invalid uuid")
            return synapse
        if not _is_hex64(key):
            bt.logging.warning(f"Reveal {synapse.uuid}: invalid key")
            return synapse
        if not math.isfinite(synapse.time_limit) or synapse.time_limit <= 0:
            bt.logging.warning(f"Reveal {synapse.uuid}: invalid time limit")
            return synapse

        pending_key = (hotkey, synapse.uuid)
        challenge = self._pending.get(pending_key)
        if challenge is None:
            bt.logging.warning(f"Reveal {synapse.uuid}: no committed challenge from {hotkey}")
            return synapse
        try:
            encoded_matrix = unseal(
                uuid=synapse.uuid,
                payload=challenge.sealed_matrix,
                key=key,
                digest=challenge.digest,
            )
        except (SealError, ValueError) as error:
            # Keep the commit (until its TTL): only a key that opens it may consume it.
            bt.logging.error(f"Reveal {synapse.uuid}: unseal failed: {error}")
            return synapse
        self._pop(pending_key)

        try:
            clique = await self.solve(synapse, encoded_matrix)
        except Exception as error:  # never let one bad challenge take the request down
            bt.logging.error(f"Reveal {synapse.uuid}: solve failed: {type(error).__name__}: {error}")
            return synapse
        synapse.maximum_clique = [int(v) for v in clique or []]
        return synapse

    def _pop(self, pending_key: tuple[str, str]) -> None:
        if self._pending.pop(pending_key, None) is not None:
            self._release(pending_key[0])

    def _evict_oldest(self, hotkey: str) -> None:
        for pending_key in self._pending:
            if pending_key[0] == hotkey:
                del self._pending[pending_key]
                self._release(hotkey)
                bt.logging.warning(
                    f"Evicting pending commit {pending_key[1]} from {hotkey}: "
                    "per-hotkey pending limit reached"
                )
                return

    def _release(self, hotkey: str) -> None:
        self._pending_per_hotkey[hotkey] -= 1
        if self._pending_per_hotkey[hotkey] <= 0:
            del self._pending_per_hotkey[hotkey]

    def _prune(self) -> None:
        expires_before = time.monotonic() - self.pending_ttl
        while self._pending:
            pending_key, challenge = next(iter(self._pending.items()))
            if challenge.created_at > expires_before:
                break
            del self._pending[pending_key]
            self._release(pending_key[0])
