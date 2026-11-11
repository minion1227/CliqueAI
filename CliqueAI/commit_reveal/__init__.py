"""Commit-reveal challenge distribution for CliqueAI (upstream protocol v0.0.18).

`sealer` is upstream's file unchanged, so seals made by v0.0.18 validators open here.
"""

from .responder import CommitRevealResponder, PendingChallenge
from .sealer import (
    KEY_BYTES,
    SealedChallenge,
    SealError,
    commitment,
    keystream,
    seal,
    unseal,
)

__all__ = [
    "CommitRevealResponder",
    "KEY_BYTES",
    "PendingChallenge",
    "SealError",
    "SealedChallenge",
    "commitment",
    "keystream",
    "seal",
    "unseal",
]
