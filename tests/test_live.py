"""
Live session tests - the human-in-the-loop path.

The test that matters most here is `test_live_matches_the_batch_run`. It runs
the same weeks twice, once through the batch runner with a simulated manager
and once through the live session answering identically, and asserts the two
produce the same numbers to the conversion. If they ever diverge, then the
figures in the report were produced by different code from the prototype in
the demonstration, and neither can be trusted.

Run either way:
    python -m pytest tests -q
    python tests/test_live.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.graph import AgenticProcess, ManagerPolicy  # noqa: E402
from src.agents.live import AWAITING, SEASON_COMPLETE, WEEK_COMPLETE, LiveSession  # noqa: E402
from src.allocator import ThompsonSamplingAllocator  # noqa: E402
from src.policy import Rulebook  # noqa: E402
from src.world import CHANNELS, SEED, generate_latent, run_allocation  # noqa: E402

START_SHARES = {"google": 0.40, "meta": 0.35, "amazon": 0.25}
WEEKS = 14
SCENARIO = {6: {"action": "modify"}, 11: {"action": "reject"}}


def _rulebook() -> Rulebook:
    return Rulebook.load()


def _session(weeks: int = WEEKS, **kwargs) -> LiveSession:
    latent = generate_latent(seed=SEED, n_weeks=weeks)
    return LiveSession(latent, _rulebook(), concentration=3.0, discount=0.0,
                       allocator_seed=7, world_seed=SEED,
                       start_shares=START_SHARES, **kwargs)


def _advance_to_a_pause(session: LiveSession) -> dict:
    """Run weeks until the graph stops for a human."""
    while not session.finished:
        step = session.start_next_week()
        if step["state"] == AWAITING:
            return step
    raise AssertionError("no week ever asked for a decision")


# ---------------------------------------------------------------------------
# Pausing and resuming
# ---------------------------------------------------------------------------

def test_the_graph_pauses_for_the_manager():
    session = _session()
    step = _advance_to_a_pause(session)

    assert step["state"] == AWAITING
    assert session.awaiting
    assert session.pending["week"] == session.week


def test_the_pause_carries_what_the_manager_needs_to_decide():
    session = _session()
    payload = _advance_to_a_pause(session)["payload"]

    for key in ("case_id", "week", "brief", "previous_shares", "proposal",
                "reasons", "shift_pp", "options"):
        assert key in payload, f"{key} missing from the approval payload"

    assert payload["brief"].strip(), "an approval with no brief is not oversight"
    assert set(payload["proposal"]) == set(CHANNELS)
    assert payload["options"] == ["approve", "modify", "reject"]


def test_approving_completes_the_week():
    session = _session()
    _advance_to_a_pause(session)
    proposed = dict(session.pending["proposal"])

    step = session.submit({"action": "approve"})

    assert step["state"] == WEEK_COMPLETE
    assert not session.awaiting
    assert step["case"]["status"] == "executed"
    for c in CHANNELS:
        assert abs(step["case"]["executed"][c] - proposed[c]) < 1e-6


def test_rejecting_pauses_again_with_a_revised_proposal():
    """E4 through the live path: one week, two decisions."""
    session = _session()
    _advance_to_a_pause(session)
    first = dict(session.pending["proposal"])

    step = session.submit({"action": "reject", "note": "too aggressive"})

    assert step["state"] == AWAITING, "a rejection must come back for a decision"
    assert session.pending["replans"] == 1
    revised = session.pending["proposal"]
    assert any(abs(revised[c] - first[c]) > 1e-6 for c in CHANNELS), \
        "the re-plan returned the identical proposal"


def test_a_modification_is_honoured_but_still_checked():
    session = _session()
    _advance_to_a_pause(session)
    illegal = {"google": 0.90, "meta": 0.05, "amazon": 0.05}

    step = session.submit({"action": "modify", "shares": illegal})
    while step["state"] == AWAITING:
        step = session.submit({"action": "approve"})

    executed = step["case"]["executed"]
    assert not session.rulebook.check(executed, step["case"]["previous_shares"]), \
        "a manager override must not bypass the rulebook"
    assert executed["google"] <= 0.55 + 1e-6


# ---------------------------------------------------------------------------
# Guard rails on the driver itself
# ---------------------------------------------------------------------------

def test_cannot_start_a_week_while_a_decision_is_outstanding():
    session = _session()
    _advance_to_a_pause(session)
    try:
        session.start_next_week()
    except RuntimeError:
        return
    raise AssertionError("started a new week with an approval still open")


def test_cannot_submit_when_nothing_is_waiting():
    session = _session(weeks=3)
    try:
        session.submit({"action": "approve"})
    except RuntimeError:
        return
    raise AssertionError("accepted a decision nobody asked for")


def test_the_season_finishes():
    session = _session(weeks=4)
    policy = ManagerPolicy()
    session.auto_run(4, policy)

    assert session.finished
    assert session.start_next_week()["state"] == SEASON_COMPLETE
    assert len(session.cases) == 4


# ---------------------------------------------------------------------------
# The parity test
# ---------------------------------------------------------------------------

def test_live_matches_the_batch_run_given_the_same_answers():
    """
    The dashboard and the report must be the same prototype.

    Same world, same seeds, same manager answers - so the live session and the
    batch run have to agree exactly. If they do not, the numbers in the report
    did not come from the thing being demonstrated.
    """
    rulebook = _rulebook()
    latent = generate_latent(seed=SEED, n_weeks=WEEKS)

    allocator = ThompsonSamplingAllocator(
        rulebook, concentration=3.0, discount=0.0, seed=7,
        start_shares=START_SHARES)
    batch = AgenticProcess(rulebook, allocator,
                           manager_policy=ManagerPolicy(SCENARIO),
                           start_shares=START_SHARES)
    batch_run = run_allocation(latent, batch, seed=SEED)

    session = _session()
    session.auto_run(WEEKS, ManagerPolicy(SCENARIO))
    live_run = session.run_frame()

    assert len(live_run) == len(batch_run)
    assert int(live_run["conversions"].sum()) == int(batch_run["conversions"].sum()), \
        "live and batch disagree on conversions"
    assert np.allclose(live_run["share"].to_numpy(),
                       batch_run["share"].to_numpy(), atol=1e-9), \
        "live and batch allocated differently"

    batch_counts = batch.process_counts()
    live_counts = session.process_counts()
    for key in ("cycles", "auto_executed", "sent_for_approval", "held",
                "anomaly_weeks", "trimmed_weeks", "replan_cycles"):
        assert batch_counts[key] == live_counts[key], \
            f"{key}: batch {batch_counts[key]} vs live {live_counts[key]}"


def test_the_audit_trail_survives_the_live_path():
    session = _session(weeks=5)
    session.auto_run(5, ManagerPolicy())

    audit = session.audit_frame()
    assert not audit.empty
    assert set(audit["week"]) == set(range(1, 6))
    assert "budget_api" in set(audit["tool"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} live-session tests passed")
    sys.exit(1 if failed else 0)
