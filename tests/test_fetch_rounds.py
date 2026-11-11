"""scripts/fetch_rounds.py must read the *newest* rounds of the W&B runs that are still logging.

W&B scans a run's history forward from a start step, and a validator keeps one run open for as
long as its process lives, so a plain scan returns rounds that can be weeks old.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_rounds import (  # noqa: E402
    KEYS,
    REQUIRED,
    fetch_newest,
    hours_ago,
    newest_rows,
    unix_time,
)


def iso(hours_ago_: float) -> str:
    """W&B-style naive UTC ISO timestamp."""
    when = datetime.fromtimestamp(time.time() - hours_ago_ * 3600, tz=timezone.utc)
    return when.replace(tzinfo=None).isoformat(timespec="seconds")


def a_round(step: int, keys=KEYS) -> dict:
    return {k: (time.time() - (10_000 - step) * 60 if k == "timestamp" else [step]) for k in keys}


class FakeRun:
    """Mimics wandb's Run: a key-filtered scan yields, in step order from `min_step`, only the
    rows that contain every requested key."""

    def __init__(self, name: str, rows: dict[int, dict], idle_hours: float = 0.1):
        self.name = name
        self.rows = rows
        self.heartbeat_at = iso(idle_hours)
        self.scans: list[int] = []

    @property
    def lastHistoryStep(self) -> int:
        return max(self.rows, default=-1)

    def scan_history(self, keys=None, page_size=1000, min_step=0, max_step=None):
        self.scans.append(min_step)
        for step in sorted(self.rows):
            if step >= min_step and all(k in self.rows[step] for k in keys or []):
                yield {"_step": step, **self.rows[step]}


class FakeApi:
    def __init__(self, runs: list[FakeRun]):
        self._runs = runs

    def runs(self, project, order="+created_at", per_page=50):
        assert order == "-heartbeat_at"
        return sorted(self._runs, key=lambda r: unix_time(r.heartbeat_at), reverse=True)


def steps(rows: list[dict]) -> list[int]:
    return [r["_step"] for r in rows]


def test_reads_the_end_of_a_long_run():
    run = FakeRun("long", {s: a_round(s) for s in range(5000)})
    assert steps(newest_rows(run, 10)) == list(range(4990, 5000))
    assert run.scans == [4990]  # one scan of the tail, not of the whole run


def test_widens_past_rows_that_are_not_rounds():
    rows = {s: (a_round(s) if s % 3 == 0 else {"other": 1}) for s in range(1000)}
    got = steps(newest_rows(FakeRun("mixed", rows), 10))
    assert got == [s for s in range(1000) if s % 3 == 0][-10:]


def test_falls_back_to_required_keys():
    run = FakeRun("old-validator", {s: a_round(s, REQUIRED) for s in range(100)})
    got = newest_rows(run, 5)
    assert steps(got) == [95, 96, 97, 98, 99]
    assert all(set(REQUIRED) <= set(r) for r in got)


def test_gives_up_on_a_run_without_rounds_after_bounded_scans():
    run = FakeRun("no-rounds", {s: {"other": 1} for s in range(10_000)})
    assert newest_rows(run, 10, max_widen=16) == []
    assert min(run.scans) >= 10_000 - 160  # never scans further back than 16 x 10 steps


def test_empty_run():
    assert newest_rows(FakeRun("empty", {}), 10) == []


def test_skips_idle_runs_and_spreads_rounds_over_active_ones():
    big = FakeRun("big", {s: a_round(s) for s in range(1000)}, idle_hours=0.1)
    small = FakeRun("restarted", {s: a_round(s) for s in range(5)}, idle_hours=0.5)
    idle = FakeRun("idle", {s: a_round(s) for s in range(1000)}, idle_hours=100)
    log: list[str] = []
    got = list(fetch_newest(FakeApi([big, small, idle]), "p", 30, 12, 48, log.append))
    assert len(got) == 30
    assert sum(r["run"] == "restarted" for r in got) == 5  # all it has; "big" fills the rest
    assert steps([r for r in got if r["run"] == "big"]) == list(range(975, 1000))
    assert not idle.scans and any("idle" in line and "skipped" in line for line in log)


def test_unix_time():
    assert unix_time("2026-11-11T15:00:00") == unix_time("2026-11-11T15:00:00Z")
    assert unix_time("2026-11-11T15:00:00") == unix_time("2026-11-11T16:00:00+01:00")
    assert unix_time("not a time") is None and unix_time(None) is None


def test_hours_ago():
    now = time.time()
    rows = [{"timestamp": now - 3600}, {"timestamp": now - 3 * 3600}, {"timestamp": None}]
    assert hours_ago(rows, now) == "logged 1.0-3.0 h ago"
    assert hours_ago([{}], now) == ""
