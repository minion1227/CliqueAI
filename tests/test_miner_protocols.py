"""Tests for the miner's request handling: both protocols, the blacklist and its config.

Run from the repo root:  python -m pytest tests/test_miner_protocols.py
The last test serves a real axon on 127.0.0.1 and queries it with a real dendrite.
"""

import argparse
import asyncio
import json
import random
import socket
import time
import types
from pathlib import Path

import bittensor as bt
import pytest
from CliqueAI.commit_reveal import CommitRevealResponder, seal
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.graph.model import LambdaGraph
from CliqueAI.miner import Miner
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph, MaximumCliqueSealedGraph
from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator

ROOT = Path(__file__).resolve().parents[1]
CODEC = GraphCodec()
VALIDATOR = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"  # dev key Alice
NO_PERMIT = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"  # dev key Bob
STRANGER = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"  # dev key Charlie, unregistered


def load(name: str) -> str:
    return json.loads((ROOT / "test_data" / f"{name}.json").read_text())["encoded_matrix"]


def validator_accepts(encoded: str, clique: list[int]) -> bool:
    adj = CODEC.matrix_to_list(CODEC.decode_matrix(encoded))
    graph = LambdaGraph(uuid="t", label="general", number_of_nodes=len(adj), adjacency_list=adj)
    return CliqueScoreCalculator(graph, 0.8, [clique]).is_valid_maximum_clique(clique)


def make_miner(validator: str = VALIDATOR, near_max_cliques: int = 4) -> Miner:
    """A Miner with stubbed chain state: no subtensor, wallet or axon needed."""
    miner = Miner.__new__(Miner)
    miner.config = types.SimpleNamespace(
        solver=types.SimpleNamespace(
            threads=2, safety_margin=1.0, pick="rarest", near_max_cliques=near_max_cliques, schedule=""
        ),
        blacklist=types.SimpleNamespace(force_validator_permit=True, allow_non_registered=False),
    )
    miner.metagraph = types.SimpleNamespace(
        hotkeys=[validator, NO_PERMIT], validator_permit=[True, False], S=[1000.0, 1.0]
    )
    miner.rng = random.Random(0)
    miner.cpu_count = 2
    miner.active_requests = 0
    miner.commit_reveal = CommitRevealResponder(solve=miner.solve_revealed)
    return miner


def sealed(uuid: str, hotkey: str | None = VALIDATOR, **fields) -> MaximumCliqueSealedGraph:
    syn = MaximumCliqueSealedGraph(uuid=uuid, **fields)
    if hotkey:
        syn.dendrite = bt.TerminalInfo(hotkey=hotkey, nonce=time.time_ns())
    return syn


def test_forward_graph_answers_and_drops_the_graph():
    encoded = load("general_0.2")
    syn = MaximumCliqueOfLambdaGraph(uuid="u1", number_of_nodes=158, encoded_matrix=encoded, timeout=2.0)
    syn.dendrite = bt.TerminalInfo(hotkey=VALIDATOR, nonce=time.time_ns())
    out = asyncio.run(make_miner().forward_graph(syn))
    assert out.encoded_matrix == "", "the 62 KB graph must not be echoed back"
    assert validator_accepts(encoded, out.maximum_clique)


def test_sealed_commit_reveal_round_trip():
    encoded = load("general_0.2")
    challenge = seal("graph-1", encoded)
    miner = make_miner()

    async def main():
        commit = await miner.forward_sealed_graph(
            sealed("graph-1", sealed_matrix=challenge.payload, digest=challenge.digest, timeout=30.0)
        )
        assert commit.maximum_clique == [] and commit.sealed_matrix == "" and commit.digest == ""
        assert len(miner.commit_reveal) == 1
        return await miner.forward_sealed_graph(
            sealed("graph-1", key=challenge.key, time_limit=2.0, timeout=2.0)
        )

    reveal = asyncio.run(main())
    assert reveal.key == "" and len(miner.commit_reveal) == 0
    assert validator_accepts(encoded, reveal.maximum_clique)


def test_sealed_rejections_keep_the_commit_for_the_right_key():
    encoded = load("general_0.1")
    good = seal("g", encoded)
    wrong = seal("g", encoded)  # same graph, different key and pad
    miner = make_miner()
    pending = miner.commit_reveal
    handle = miner.forward_sealed_graph

    async def main():
        out = await handle(sealed("g", key=good.key, time_limit=1.0, timeout=1.0))
        assert out.maximum_clique == []  # reveal before any commit
        await handle(sealed("g", sealed_matrix=good.payload, digest=good.digest))
        await handle(sealed("g", sealed_matrix=good.payload, digest=good.digest))  # duplicate
        assert len(pending) == 1
        out = await handle(sealed("g", hotkey=NO_PERMIT, key=good.key, time_limit=1.0, timeout=1.0))
        assert out.maximum_clique == [] and len(pending) == 1  # another hotkey can't open it
        out = await handle(sealed("g", key=wrong.key, time_limit=1.0, timeout=1.0))
        assert out.maximum_clique == [] and len(pending) == 1  # digest mismatch
        out = await handle(
            sealed("g", sealed_matrix=good.payload, digest=good.digest, key=good.key, time_limit=1.0)
        )
        assert out.maximum_clique == [] and out.sealed_matrix == "" and out.key == ""
        await handle(sealed("h", sealed_matrix=good.payload, digest="not-hex"))
        await handle(sealed("i", sealed_matrix=good.payload[:-1] + "é", digest=good.digest))
        await handle(sealed("j", hotkey=None, sealed_matrix=good.payload, digest=good.digest))
        assert len(pending) == 1
        out = await handle(sealed("k", key=good.key, time_limit=0.0, timeout=1.0))
        assert out.maximum_clique == []
        return await handle(sealed("g", key=good.key, time_limit=1.0, timeout=1.0))

    out = asyncio.run(main())
    assert validator_accepts(encoded, out.maximum_clique) and len(pending) == 0


def test_sealed_pending_quota_and_ttl():
    async def never(synapse, encoded):
        raise AssertionError("not revealed in this test")

    payload = seal("x", load("general_0.1"))
    r = CommitRevealResponder(solve=never, max_pending=3, max_pending_per_hotkey=2, pending_ttl=0.3)

    async def commit(uuid, hotkey):
        await r.handle(sealed(uuid, hotkey=hotkey, sealed_matrix=payload.payload, digest=payload.digest))

    async def main():
        for uuid in "abc":
            await commit(uuid, VALIDATOR)  # quota 2: "c" evicts this hotkey's oldest, "a"
        assert set(r._pending) == {(VALIDATOR, "b"), (VALIDATOR, "c")}
        await commit("d", NO_PERMIT)
        await commit("e", STRANGER)  # table full: rejected, nobody else's commit evicted
        assert len(r) == 3 and (STRANGER, "e") not in r._pending
        await asyncio.sleep(0.35)
        await commit("f", STRANGER)  # expired commits are pruned first
        assert set(r._pending) == {(STRANGER, "f")}

    asyncio.run(main())
    with pytest.raises(ValueError):
        CommitRevealResponder(solve=never, pending_ttl=float("inf"))


def test_undecodable_graph_returns_empty_not_exception():
    # Unseals fine but is not a valid GraphCodec string: solver and fallback both fail.
    bad = seal("y", "!!!")
    miner = make_miner()

    async def main():
        await miner.forward_sealed_graph(sealed("y", sealed_matrix=bad.payload, digest=bad.digest))
        return await miner.forward_sealed_graph(sealed("y", key=bad.key, time_limit=1.0, timeout=1.0))

    assert asyncio.run(main()).maximum_clique == []


def test_blacklist_and_priority():
    miner = make_miner()

    def call(fn, hotkey):
        syn = MaximumCliqueOfLambdaGraph()
        syn.dendrite = bt.TerminalInfo(hotkey=hotkey)
        return asyncio.run(fn(syn))

    assert call(miner.blacklist, STRANGER) == (True, "Unrecognized hotkey")  # used to raise
    assert call(miner.blacklist, NO_PERMIT)[0] is True
    assert call(miner.blacklist, VALIDATOR)[0] is False
    assert call(miner.priority, STRANGER) == 0.0
    assert call(miner.priority, VALIDATOR) == 1000.0
    miner.config.blacklist.force_validator_permit = False
    assert call(miner.blacklist, NO_PERMIT)[0] is False
    assert call(miner.blacklist, STRANGER)[0] is True
    miner.config.blacklist.allow_non_registered = True
    assert call(miner.blacklist, STRANGER)[0] is False


def test_unsigned_request_is_not_verified():
    miner = make_miner()
    syn = MaximumCliqueSealedGraph(uuid="u")
    syn.dendrite = bt.TerminalInfo(hotkey=VALIDATOR, nonce=time.time_ns())
    with pytest.raises(Exception, match="signature"):
        asyncio.run(miner.verify_sealed_graph(syn))


def test_config_defaults():
    parser = argparse.ArgumentParser()
    Miner.add_args(parser)
    args = vars(parser.parse_args([]))
    assert args["blacklist.force_validator_permit"] is True
    assert args["solver.near_max_cliques"] == 4
    off = vars(parser.parse_args(["--no-blacklist.force_validator_permit", "--solver.near_max_cliques", "0"]))
    assert off["blacklist.force_validator_permit"] is False and off["solver.near_max_cliques"] == 0
    assert vars(parser.parse_args(["--blacklist.force_validator_permit"]))["blacklist.force_validator_permit"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_axon_serves_both_protocols(tmp_path):
    """Signed requests through a real axon: body hashes, signatures, verify and blacklist."""
    wallets = {}
    for name in ("miner", "validator"):
        wallets[name] = bt.Wallet(name=name, hotkey="h", path=str(tmp_path))
        wallets[name].create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    miner = make_miner(validator=wallets["validator"].hotkey.ss58_address)
    port = free_port()
    axon = bt.Axon(wallet=wallets["miner"], ip="127.0.0.1", port=port, external_ip="127.0.0.1")
    miner.axon = axon
    axon.attach(
        forward_fn=miner.forward_graph,
        blacklist_fn=miner.blacklist_graph,
        priority_fn=miner.priority_graph,
        verify_fn=miner.verify_graph,
    )
    axon.attach(
        forward_fn=miner.forward_sealed_graph,
        blacklist_fn=miner.blacklist_sealed_graph,
        priority_fn=miner.priority_sealed_graph,
        verify_fn=miner.verify_sealed_graph,
    )
    target = bt.AxonInfo(
        version=1, ip="127.0.0.1", port=port, ip_type=4,
        hotkey=wallets["miner"].hotkey.ss58_address, coldkey=wallets["miner"].coldkeypub.ss58_address,
    )  # fmt: skip
    encoded = load("general_0.2")
    challenge = seal("e2e-1", encoded)
    axon.start()
    try:

        async def main():
            dendrite = bt.Dendrite(wallet=wallets["validator"])
            try:
                old = await dendrite.forward(
                    axons=[target], timeout=3.0, deserialize=False,
                    synapse=MaximumCliqueOfLambdaGraph(uuid="e2e-0", number_of_nodes=158, encoded_matrix=encoded),
                )  # fmt: skip
                commit = await dendrite.forward(
                    axons=[target], timeout=10.0, deserialize=False,
                    synapse=MaximumCliqueSealedGraph(uuid="e2e-1", sealed_matrix=challenge.payload, digest=challenge.digest),
                )  # fmt: skip
                reveal = await dendrite.forward(
                    axons=[target], timeout=3.0, deserialize=False,
                    synapse=MaximumCliqueSealedGraph(uuid="e2e-1", key=challenge.key, time_limit=3.0),
                )  # fmt: skip
                stranger = bt.Dendrite(wallet=wallets["miner"])  # registered nowhere in the stub
                try:
                    refused = await stranger.forward(
                        axons=[target], timeout=3.0, deserialize=False,
                        synapse=MaximumCliqueOfLambdaGraph(uuid="e2e-2", number_of_nodes=158, encoded_matrix=encoded),
                    )  # fmt: skip
                finally:
                    await stranger.aclose_session()
                return old[0], commit[0], reveal[0], refused[0]
            finally:
                await dendrite.aclose_session()

        old, commit, reveal, refused = asyncio.run(main())
    finally:
        axon.stop()
    assert old.dendrite.status_code == 200, old.dendrite.status_message
    assert validator_accepts(encoded, old.maximum_clique)
    assert old.encoded_matrix == ""
    assert commit.dendrite.status_code == 200, commit.dendrite.status_message
    assert reveal.dendrite.status_code == 200, reveal.dendrite.status_message
    assert validator_accepts(encoded, reveal.maximum_clique)
    assert refused.dendrite.status_code != 200 and refused.maximum_clique == []
