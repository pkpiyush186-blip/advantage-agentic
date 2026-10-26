"""
Graph tests: routing, every exception path, and the audit trail.

The 26-week run exercises E1, E2, E4 and E5 on its own, but E3 only fires
when the rulebook cannot be satisfied, which never happens in a normal week.
The prototype has to be able to demonstrate every exception path, so E3 gets
a constructed case here.

Two of these are regression tests for bugs this code actually had:

    test_routing_decisions_are_persisted
        The three decision points were originally computed inside the
        conditional-edge functions. LangGraph keeps what a node returns and
        discards what an edge writes, so D1/D2/D3 never reached the case
        record and never reached the audit log.

    test_repeated_rejection_stops_instead_of_looping
        The same bug left the re-plan counter permanently at zero, so a
        manager who kept rejecting was asked forever and the graph span until
        it hit the recursion limit.

Run either way:
    python -m pytest tests -q
    python tests/test_graph.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.graph import AgenticProcess, ManagerPolicy  # noqa: E402
from src.allocator import ThompsonSamplingAllocator  # noqa: E402
from src.policy import Rulebook  # noqa: E402
from src.world import CHANNELS  # noqa: E402

START_SHARES = {"google": 0.40, "meta": 0.35, "amazon": 0.25}
EQUAL = {c: 1 / 3 for c in CHANNELS}


def _rulebook() -> Rulebook:
    return Rulebook.load()


def _process(rulebook=None, scenario=None, concentration=3.0,
             start_shares=None) -> AgenticProcess:
    rulebook = rulebook or _rulebook()
    allocator = ThompsonSamplingAllocator(
        rulebook, concentration=concentration, discount=0.0, seed=7,
        start_shares=start_shares or START_SHARES)
    return AgenticProcess(rulebook, allocator,
                          manager_policy=ManagerPolicy(scenario or {}),
                          start_shares=start_shares or START_SHARES)


def make_history(n_weeks: int = 6, *, shares=None, cpc=20.0, cvr=0.025,
                 spike_last: dict | None = None,
                 missing_last: list[str] | None = None) -> pd.DataFrame:
    """A steady reported history, optionally broken in its final week."""
    shares = shares or START_SHARES
    budget = 1_150_000.0
    rows = []

    for week in range(1, n_weeks + 1):
        last = week == n_weeks
        for channel in CHANNELS:
            this_cpc, this_cvr = cpc, cvr
            missing = bool(last and missing_last and channel in missing_last)

            if last and spike_last and channel in spike_last:
                this_cpc *= spike_last[channel].get("cpc", 1.0)
                this_cvr *= spike_last[channel].get("cvr", 1.0)

            spend = shares[channel] * budget
            clicks = spend / this_cpc
            conversions = clicks * this_cvr
            rows.append({
                "week": week, "channel": channel, "share": shares[channel],
                "spend": spend, "clicks": clicks,
                "conversions": round(conversions), "revenue": conversions * 2000.0,
                "feed_missing": missing, "shock": "", "festive": False,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Happy path and routing
# ---------------------------------------------------------------------------

def test_clean_week_takes_the_happy_path():
    process = _process()
    process(7, make_history(6))
    case = process.cases[-1]

    assert case["d1"] == "propose", case["d1"]
    assert case["anomalies"] == []
    assert case["status"] == "executed"


def test_routing_decisions_are_persisted_in_the_case_record():
    """Regression: D1/D2/D3 were computed in edge functions and lost."""
    process = _process()
    process(7, make_history(6))
    case = process.cases[-1]

    for key in ("d1", "d2", "d3"):
        assert key in case and case[key], f"{key} missing from the case record"

    stages = {entry["stage"] for entry in case["audit"]}
    assert {"D1", "D2"} <= stages, stages


def test_shares_always_sum_to_one_whatever_path_is_taken():
    for scenario in ({}, {7: {"action": "modify"}}, {7: {"action": "reject"}},
                     {7: {"action": "timeout"}}):
        process = _process(scenario=scenario)
        shares = process(7, make_history(6))
        assert abs(sum(shares.values()) - 1.0) < 1e-6, (scenario, shares)


# ---------------------------------------------------------------------------
# E1 - anomaly freezes a channel
# ---------------------------------------------------------------------------

def test_cpc_spike_freezes_the_channel_and_takes_e1():
    process = _process()
    history = make_history(6, spike_last={"google": {"cpc": 1.8}})
    process(7, history)
    case = process.cases[-1]

    assert case["d1"] == "exception"
    assert any(a["type"] == "cpc_spike" and a["channel"] == "google"
               for a in case["anomalies"]), case["anomalies"]
    assert "google" in case["frozen_channels"]
    assert any("E1" in note for note in case["escalations"])


def test_a_frozen_channel_does_not_move():
    process = _process()
    process(7, make_history(6, spike_last={"google": {"cpc": 1.8}}))
    case = process.cases[-1]

    assert abs(case["executed"]["google"] - case["previous_shares"]["google"]) < 1e-6


def test_steady_data_raises_no_anomaly():
    """The detector must not fire on a quiet week - it cried wolf once."""
    process = _process()
    process(7, make_history(6))
    assert process.cases[-1]["anomalies"] == []


# ---------------------------------------------------------------------------
# E2 - missing feed
# ---------------------------------------------------------------------------

def test_missing_feed_marks_the_case_degraded_and_halves_the_cap():
    process = _process()
    process(7, make_history(6, missing_last=["amazon"]))
    case = process.cases[-1]

    assert case["degraded"] is True
    assert "amazon" in case["missing_feeds"]
    assert case["used_fallback_data"] is True, "should fall back to the last valid week"
    assert any("E2" in note for note in case["escalations"])

    cap = float(process.rulebook.data["hard_limits"]["max_weekly_shift_pp"])
    moved = max(abs(case["executed"][c] - case["previous_shares"][c]) * 100.0
                for c in CHANNELS)
    assert moved <= cap / 2 + 1e-6, f"moved {moved:.2f} pp on degraded data"


# ---------------------------------------------------------------------------
# E3 - nothing complies
# ---------------------------------------------------------------------------

def test_infeasible_rulebook_escalates_and_holds_e3():
    data = copy.deepcopy(_rulebook().data)
    data["hard_limits"]["min_channel_share"] = 0.40   # 3 x 40% = 120%
    rulebook = Rulebook(data)

    process = _process(rulebook=rulebook)
    shares = process(7, make_history(6))
    case = process.cases[-1]

    assert case["feasible"] is False
    assert case["d2"] == "escalate"
    assert case["status"] == "escalated"
    assert all(abs(shares[c] - START_SHARES[c] / sum(START_SHARES.values())) < 1e-6
               for c in CHANNELS), "an infeasible week must hold, not guess"
    assert any("E3" in note for note in case["escalations"])


# ---------------------------------------------------------------------------
# E4 - manager override
# ---------------------------------------------------------------------------

def test_manager_rejection_triggers_a_replan():
    process = _process(scenario={7: {"action": "reject"}})
    process(7, make_history(6))
    case = process.cases[-1]

    assert case["replans"] == 1
    assert case["status"] == "executed"
    assert any("E4" in entry["summary"] for entry in case["audit"])


def test_manager_modification_is_honoured_but_still_checked():
    """A human override does not get to bypass the rulebook."""
    illegal = {"google": 0.90, "meta": 0.05, "amazon": 0.05}
    process = _process(scenario={7: {"action": "modify", "shares": illegal}})
    process(7, make_history(6))
    case = process.cases[-1]

    assert case["replans"] >= 1
    assert not process.rulebook.check(case["executed"], case["previous_shares"]), \
        "executed an allocation the rulebook forbids"
    assert case["executed"]["google"] <= 0.55 + 1e-6


def test_repeated_rejection_stops_instead_of_looping():
    """Regression: the re-plan counter never incremented, so this never ended."""
    class AlwaysRejects(ManagerPolicy):
        def __call__(self, case):
            return {"action": "reject", "note": "no"}

    rulebook = _rulebook()
    allocator = ThompsonSamplingAllocator(rulebook, concentration=3.0, discount=0.0,
                                          seed=7, start_shares=START_SHARES)
    process = AgenticProcess(rulebook, allocator,
                             manager_policy=AlwaysRejects(),
                             start_shares=START_SHARES)
    shares = process(7, make_history(6))
    case = process.cases[-1]

    max_replans = int(rulebook.data["approval"]["max_replans"])
    assert case["replans"] == max_replans
    assert case["status"] == "held"
    assert all(abs(shares[c] - case["previous_shares"][c]) < 1e-6 for c in CHANNELS)


def test_an_override_is_remembered_for_later_weeks():
    process = _process(scenario={7: {"action": "modify"}})
    process(7, make_history(6))

    assert process.agents.overrides, "override history is one of the three memories"
    assert process.agents.overrides[-1]["week"] == 7


# ---------------------------------------------------------------------------
# E5 - no response
# ---------------------------------------------------------------------------

def test_timeout_escalates_and_holds_the_previous_allocation():
    process = _process(scenario={7: {"action": "timeout"}})
    shares = process(7, make_history(6))
    case = process.cases[-1]

    assert case["status"] == "held"
    assert all(abs(shares[c] - case["previous_shares"][c]) < 1e-6 for c in CHANNELS)
    assert any("E5" in note for note in case["escalations"])

    senior = process.rulebook.data["approval"]["escalation_after_sla_to"]
    assert any(m["to"] == senior for m in process.tools.notifications), \
        "E5 must escalate to the senior approver"


# ---------------------------------------------------------------------------
# Autonomy and oversight
# ---------------------------------------------------------------------------

def test_a_small_clean_shift_runs_without_a_human():
    """Act-then-report autonomy: no manager involved below the threshold."""
    process = _process(concentration=0.05)   # barely moves off the current split
    process(7, make_history(6))
    case = process.cases[-1]

    if case["route"] == "auto_execute":
        assert case["manager_decision"] == {}
        assert case["status"] == "executed"
    else:
        # Still acceptable, but it must say why a human was involved.
        assert case["approval_reasons"]


def test_a_large_shift_always_reaches_a_human():
    process = _process(concentration=12.0)
    process(7, make_history(6, spike_last=None))
    case = process.cases[-1]

    if case["d3"] == "execute":
        assert case["route"] == "auto_execute"
        moved = max(abs(case["executed"][c] - case["previous_shares"][c]) * 100.0
                    for c in CHANNELS)
        threshold = float(process.rulebook.data["approval"]["auto_execute_max_shift_pp"])
        assert moved <= threshold + 1e-6
    else:
        assert case["manager_decision"], "a large shift must be put to a human"


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

def test_the_audit_trail_records_every_tool_call():
    process = _process()
    process(7, make_history(6))
    case = process.cases[-1]

    tools_used = {entry["tool"] for entry in case["audit"]}
    for expected in ("performance_db", "policy_kb", "anomaly_checker", "allocator",
                     "constraint_calculator", "budget_api", "audit_logger"):
        assert expected in tools_used, f"{expected} missing from {sorted(tools_used)}"


def test_every_audit_entry_is_timestamped_and_attributed():
    process = _process()
    process(7, make_history(6))

    for entry in process.cases[-1]["audit"]:
        assert entry["at"] and entry["case_id"] and entry["stage"]
        assert entry["summary"], "an audit line with no summary explains nothing"


def test_execution_is_recorded_with_amounts():
    process = _process()
    process(7, make_history(6))

    assert process.tools.executions, "the budget API call must be recorded"
    confirmation = process.tools.executions[-1]
    assert set(confirmation["amounts_inr"]) == set(CHANNELS)
    assert abs(sum(confirmation["amounts_inr"].values()) - 1_150_000.0) < 1.0


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
    print(f"\n{len(tests) - failed}/{len(tests)} graph tests passed")
    sys.exit(1 if failed else 0)
