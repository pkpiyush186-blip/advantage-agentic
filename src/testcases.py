"""
The 16 test cases from Section 13 of the proposal, as executable scenarios.

Each case builds a world the agent has never seen, runs one real weekly cycle
through the real graph, and checks what the agent actually did against what
the proposal said it should do. Nothing is stubbed: the same nodes, the same
guardrail, the same allocator and the same audit trail as the 26-week run.

Two rules were followed while writing these, and both matter more than the
pass count:

*   A case asserts the *behaviour* the proposal specifies, not the exact
    numbers that happen to come out. TC03 says a large improvement should be
    routed for approval, so the check is "shift above the threshold and sent
    to a human", not "+9.0 pp". Numbers that are asserted because they were
    observed are not tests, they are transcription.
*   Where a scenario needed a capability the prototype did not have, the
    capability was built (per-week budgets for TC15, pacing on under-delivery
    for TC08) rather than the check being softened until it passed.

The measured shift, route and evidence are reported for every case, so the
table shows what happened and not merely that something passed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .agents.graph import AgenticProcess, ManagerPolicy
from .allocator import ThompsonSamplingAllocator
from .policy import Rulebook
from .world import CHANNELS, WEEKLY_BUDGET

EQUAL = {c: 1.0 / len(CHANNELS) for c in CHANNELS}
STEADY_CPC = 20.0
STEADY_CVR = 0.025
HISTORY_WEEKS = 6
RUN_WEEK = HISTORY_WEEKS + 1


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def build_history(shares: dict[str, float], *, n_weeks: int = HISTORY_WEEKS,
                  channel_mult: dict[str, dict] | None = None,
                  last_week_mult: dict[str, dict] | None = None,
                  missing_last: list[str] | None = None,
                  delivery: dict[str, float] | None = None,
                  budget: float = WEEKLY_BUDGET) -> pd.DataFrame:
    """
    A steady reported history, bent in whatever way a case needs.

    `channel_mult` applies for every week, so it reads as a settled fact about
    a channel and raises no anomaly. `last_week_mult` applies only to the
    final week, so it reads as a sudden change and is exactly what the
    detector is meant to catch. `delivery` makes a channel spend less than it
    was given, which is the pacing case.
    """
    rows = []
    for week in range(1, n_weeks + 1):
        last = week == n_weeks
        for channel in CHANNELS:
            cpc, cvr = STEADY_CPC, STEADY_CVR

            for source, active in ((channel_mult, True), (last_week_mult, last)):
                if active and source and channel in source:
                    cpc *= source[channel].get("cpc", 1.0)
                    cvr *= source[channel].get("cvr", 1.0)

            allocated = shares[channel] * budget
            spend = allocated
            if last and delivery and channel in delivery:
                spend = allocated * delivery[channel]

            clicks = spend / cpc
            rows.append({
                "week": week, "channel": channel, "share": shares[channel],
                "spend": spend, "clicks": clicks,
                "conversions": round(clicks * cvr),
                "revenue": clicks * cvr * 2000.0,
                "feed_missing": bool(last and missing_last and channel in missing_last),
                "shock": "", "festive": False,
            })
    return pd.DataFrame(rows)


def run_case(*, shares: dict[str, float], concentration: float = 3.0,
             rulebook: Rulebook | None = None,
             manager: dict | None = None,
             budget_schedule: dict[int, float] | None = None,
             festive_weeks=(), **history_kwargs) -> tuple[dict, AgenticProcess]:
    """Run exactly one weekly cycle through the real graph."""
    rulebook = rulebook or Rulebook.load()
    history = build_history(shares, **history_kwargs)

    allocator = ThompsonSamplingAllocator(
        rulebook, concentration=concentration, discount=0.0, seed=7,
        start_shares=shares)
    process = AgenticProcess(
        rulebook, allocator,
        manager_policy=ManagerPolicy({RUN_WEEK: manager} if manager else {}),
        start_shares=shares, budget_schedule=budget_schedule,
        festive_weeks=festive_weeks)

    process(RUN_WEEK, history)
    return process.cases[-1], process


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def shift_pp(case: dict) -> float:
    return max(abs(case["executed"][c] - case["previous_shares"][c]) * 100.0
               for c in CHANNELS)


def has_anomaly(case: dict, kind: str, channel: str | None = None) -> bool:
    return any(a["type"] == kind and (channel is None or a["channel"] == channel)
               for a in case["anomalies"])


def summary(case: dict) -> str:
    moves = ", ".join(f"{c} {case['executed'][c]:.0%}" for c in CHANNELS)
    return f"{moves}; shift {shift_pp(case):.1f} pp; route {case.get('route', '-')}"


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

@dataclass
class Case:
    tc: str
    kind: str
    scenario: str
    expected: str
    run: Callable[[], tuple[dict, AgenticProcess]]
    check: Callable[[dict, AgenticProcess], tuple[bool, str]]
    covers: str = ""
    notes: str = field(default="")


def _tc01():
    case, process = run_case(shares=EQUAL, concentration=0.5)

    def check(case, process):
        ok = (case["route"] == "auto_execute" and shift_pp(case) <= 5.0
              and case["status"] == "executed" and not case["anomalies"]
              and any(e["tool"] == "audit_logger" for e in case["audit"]))
        return ok, f"{summary(case)}; logged {len(case['audit'])} entries"

    return case, process, check


def _tc02():
    case, process = run_case(shares=EQUAL, concentration=0.5,
                             last_week_mult={"meta": {"cvr": 1.12}})

    def check(case, process):
        ok = (case["route"] == "auto_execute" and shift_pp(case) <= 5.0
              and case["executed"]["meta"] > case["previous_shares"]["meta"])
        return ok, summary(case)

    return case, process, check


def _tc03():
    # A moderate improvement: the shift must land above the auto-execute
    # threshold but clear of the weekly cap, so this case tests approval
    # routing and TC07 separately tests the cap. If both saturated the cap
    # they would be the same test written twice.
    case, process = run_case(shares=EQUAL, concentration=1.2,
                             last_week_mult={"google": {"cvr": 1.45}},
                             manager={"action": "approve"})

    def check(case, process):
        cap = float(process.rulebook.data["hard_limits"]["max_weekly_shift_pp"])
        moved = shift_pp(case)
        ok = (case["route"] == "approve" and 5.0 < moved < cap - 1e-6
              and case["manager_decision"].get("action") == "approve"
              and case["status"] == "executed"
              and case["executed"]["google"] > case["previous_shares"]["google"])
        return ok, f"{summary(case)}; above the 5 pp threshold, clear of the {cap:.0f} pp cap"

    return case, process, check


def _tc04():
    case, process = run_case(shares=EQUAL, concentration=3.0,
                             last_week_mult={"amazon": {"cvr": 1.45}},
                             manager={"action": "modify"})

    def check(case, process):
        overrides = process.agents.overrides
        ok = (case["replans"] >= 1 and overrides
              and overrides[-1]["action"] == "modify"
              and case["status"] == "executed"
              and any("E4" in e["summary"] for e in case["audit"]))
        return ok, (f"{summary(case)}; re-planned {case['replans']}x; "
                    f"override logged for week {overrides[-1]['week']}"
                    if overrides else summary(case))

    return case, process, check


def _tc05():
    starved = {"google": 0.40, "meta": 0.40, "amazon": 0.20}
    case, process = run_case(shares=starved, concentration=3.0,
                             channel_mult={"amazon": {"cvr": 0.5}},
                             manager={"action": "approve"})

    def check(case, process):
        floor = float(process.rulebook.data["hard_limits"]["min_channel_share"])
        share = case["executed"]["amazon"]
        ok = share >= floor - 1e-6 and not has_anomaly(case, "cvr_collapse", "amazon")
        bound = "floor" if abs(share - floor) < 1e-3 else "shift cap"
        return ok, (f"amazon held at {share:.1%} (floor {floor:.0%}, bound by the "
                    f"{bound}) despite the weakest conversions")

    return case, process, check


def _tc06():
    case, process = run_case(shares=EQUAL, concentration=3.0,
                             channel_mult={c: {"cvr": 1.30} for c in CHANNELS},
                             festive_weeks=(RUN_WEEK,),
                             manager={"action": "approve"})

    def check(case, process):
        violations = process.rulebook.check(
            case["executed"], case["previous_shares"], festive=True)
        ok = (case["festive"] and not violations and shift_pp(case) <= 5.0
              and case["status"] == "executed")
        return ok, (f"{summary(case)}; festive week, uniform uplift moved the split "
                    f"by {shift_pp(case):.1f} pp and breached nothing")

    return case, process, check


def _tc07():
    case, process = run_case(shares=EQUAL, concentration=6.0,
                             last_week_mult={"google": {"cvr": 1.9}},
                             manager={"action": "approve"})

    def check(case, process):
        cap = float(process.rulebook.data["hard_limits"]["max_weekly_shift_pp"])
        moved = shift_pp(case)
        ok = (5.0 < moved <= cap + 1e-6 and case["route"] == "approve"
              and case["status"] == "executed")
        return ok, f"{summary(case)}; within the {cap:.0f} pp weekly cap"

    return case, process, check


def _tc08():
    case, process = run_case(shares=EQUAL, concentration=3.0,
                             delivery={"google": 0.70},
                             manager={"action": "approve"})

    def check(case, process):
        flagged = has_anomaly(case, "under_delivery", "google")
        capped = case["executed"]["google"] < case["previous_shares"]["google"]
        ok = flagged and capped and case["status"] == "executed"
        cap = (case.get("delivery_caps") or {}).get("google")
        return ok, (f"google spent 70% of its budget; capped at "
                    f"{cap:.1%} and executed at {case['executed']['google']:.1%}"
                    if cap else summary(case))

    return case, process, check


def _tc09():
    case, process = run_case(shares=EQUAL, concentration=3.0,
                             last_week_mult={"google": {"cpc": 1.60}},
                             manager={"action": "approve"})

    def check(case, process):
        frozen = "google" in case["frozen_channels"]
        held = abs(case["executed"]["google"]
                   - case["previous_shares"]["google"]) < 1e-6
        ok = (has_anomaly(case, "cpc_spike", "google") and frozen and held
              and any("E1" in note for note in case["escalations"])
              and process.tools.notifications)
        return ok, (f"CPC spike flagged; google frozen at "
                    f"{case['executed']['google']:.1%}, budget re-planned across the "
                    f"other two, manager notified")

    return case, process, check


def _tc10():
    case, process = run_case(shares=EQUAL, concentration=3.0,
                             last_week_mult={"meta": {"cvr": 0.35}},
                             manager={"action": "approve"})

    def check(case, process):
        frozen = "meta" in case["frozen_channels"]
        ok = (has_anomaly(case, "cvr_collapse", "meta") and frozen
              and any("E1" in note for note in case["escalations"]))
        return ok, (f"conversion collapse flagged; meta held at "
                    f"{case['executed']['meta']:.1%} pending investigation")

    return case, process, check


def _tc11():
    case, process = run_case(shares=EQUAL, concentration=3.0,
                             missing_last=["amazon"],
                             manager={"action": "approve"})

    def check(case, process):
        cap = float(process.rulebook.data["hard_limits"]["max_weekly_shift_pp"])
        scale = float(process.rulebook.data["degraded_mode"]["shift_cap_multiplier"])
        ok = (case["degraded"] and case["used_fallback_data"]
              and shift_pp(case) <= cap * scale + 1e-6
              and any("E2" in note for note in case["escalations"]))
        return ok, (f"feed missing; fell back to the last valid week, cap halved to "
                    f"{cap * scale:.0f} pp, moved {shift_pp(case):.1f} pp")

    return case, process, check


def _tc12():
    # Google starts at 45%, so the 15 pp weekly cap would allow 60% and it is
    # the 55% ceiling that has to bite. Started from an equal split the cap
    # binds first and the ceiling is never tested - the case would pass
    # without exercising the rule it names.
    start = {"google": 0.45, "meta": 0.30, "amazon": 0.25}
    case, process = run_case(shares=start, concentration=50.0,
                             last_week_mult={"google": {"cvr": 1.6}},
                             manager={"action": "approve"})

    def check(case, process):
        ceiling = float(process.rulebook.data["hard_limits"]["max_channel_share"])
        proposed = max(case["target"].values())
        executed = case["executed"]["google"]
        violations = process.rulebook.check(case["executed"], case["previous_shares"])
        ok = (proposed > ceiling and case["was_trimmed"] and not violations
              and abs(executed - ceiling) < 1e-3)   # the ceiling, not the cap
        return ok, (f"allocator wanted google at {proposed:.0%}; guardrail trimmed to "
                    f"the {ceiling:.0%} ceiling exactly and re-checked clean")

    return case, process, check


def _tc13():
    case, process = run_case(shares=EQUAL, concentration=3.0,
                             last_week_mult={"google": {"cvr": 1.45}},
                             manager={"action": "reject"})

    def check(case, process):
        overrides = process.agents.overrides
        ok = (case["replans"] >= 1 and overrides
              and overrides[-1]["action"] == "reject"
              and any("E4" in e["summary"] for e in case["audit"]))
        return ok, (f"rejection logged; re-planned {case['replans']}x towards the "
                    f"status quo, finished {case['status']}")

    return case, process, check


def _tc14():
    case, process = run_case(shares=EQUAL, concentration=3.0,
                             last_week_mult={"google": {"cvr": 1.45}},
                             manager={"action": "timeout"})

    def check(case, process):
        senior = process.rulebook.data["approval"]["escalation_after_sla_to"]
        held = all(abs(case["executed"][c] - case["previous_shares"][c]) < 1e-6
                   for c in CHANNELS)
        sla = int(process.rulebook.data["approval"]["approval_sla_hours"])
        ok = (case["status"] == "held" and held
              and any(m["to"] == senior for m in process.tools.notifications)
              and any("E5" in note for note in case["escalations"]))
        return ok, (f"no reply in {sla}h; reminder sent, escalated to {senior}, "
                    f"previous split held unchanged")

    return case, process, check


def _tc15():
    cut = WEEKLY_BUDGET * 0.80
    case, process = run_case(shares=EQUAL, concentration=3.0,
                             budget_schedule={RUN_WEEK: cut},
                             last_week_mult={"google": {"cvr": 1.3}},
                             manager={"action": "approve"})

    def check(case, process):
        confirmation = process.tools.executions[-1] if process.tools.executions else None
        spent = sum(confirmation["amounts_inr"].values()) if confirmation else 0.0
        violations = process.rulebook.check(case["executed"], case["previous_shares"])
        # Tolerance of ten rupees on a 9.2 lakh budget: shares are recorded to
        # six decimals and the amounts are rounded to the rupee, so an exact
        # equality check here would be testing float arithmetic, not the agent.
        ok = (abs(spent - cut) < 10.0 and not violations
              and abs(case["weekly_budget"] - cut) < 1.0)
        return ok, (f"total cut to Rs {cut:,.0f}; agent re-planned and executed "
                    f"Rs {spent:,.0f} with every limit respected")

    return case, process, check


def _tc16():
    data = copy.deepcopy(Rulebook.load().data)
    for rule in data["brand_rules"]:
        if rule["id"] == "BR01":
            rule["active"] = True
    rulebook = Rulebook(data)

    case, process = run_case(shares=EQUAL, concentration=6.0, rulebook=rulebook,
                             last_week_mult={"meta": {"cvr": 1.9}},
                             manager={"action": "approve"})

    def check(case, process):
        cap = 0.40
        wanted = case["target"]["meta"]
        cited = any("BR01" in note for note in case["trim_notes"])
        ok = (wanted > cap                                   # it wanted to breach
              and case["executed"]["meta"] <= cap + 1e-6     # it did not
              and cited)                                     # and said which rule stopped it
        return ok, (f"new rule active without a code change; allocator wanted meta at "
                    f"{wanted:.0%}, executed at {case['executed']['meta']:.0%}"
                    + ("; BR01 cited in the trim notes" if cited
                       else "; BR01 NOT cited in the trim notes"))

    return case, process, check


SPECS = [
    ("TC01", "Normal", "Stable week, small differences",
     "Minimal shift, auto-execute, logged", _tc01, "S1-S7 happy path"),
    ("TC02", "Normal", "Meta conversion rate rises slightly",
     "Within threshold, auto-execute", _tc02, "autonomy: act then report"),
    ("TC03", "Normal", "Google improves markedly",
     "Routed for approval; manager approves", _tc03, "D3, autonomy: approve"),
    ("TC04", "Normal", "Amazon strong, manager wants less",
     "Manager modifies; override logged", _tc04, "E4"),
    ("TC05", "Normal", "A channel is the weakest performer",
     "Still funded at the floor, never starved", _tc05, "floor as exploration"),
    ("TC06", "Normal", "Festive uplift across all channels",
     "Proportional shift, no breach", _tc06, "festive calendar"),
    ("TC07", "Normal", "Very strong signal on one channel",
     "Large shift within the weekly cap, approved", _tc07, "shift cap"),
    ("TC08", "Normal", "A channel under-spends its budget",
     "Pacing adjustment proposed and approved", _tc08, "pacing caps"),
    ("TC09", "Exception", "Google CPC spikes 60%",
     "Channel frozen, re-plan, escalate with context", _tc09, "E1"),
    ("TC10", "Exception", "Meta conversion collapse",
     "Hold, escalate for investigation", _tc10, "E1"),
    ("TC11", "Exception", "Amazon data feed missing",
     "Use last valid data, flag degraded, halve cap", _tc11, "E2"),
    ("TC12", "Exception", "Proposal exceeds the 55% ceiling",
     "Guardrail trims and re-checks", _tc12, "E3"),
    ("TC13", "Exception", "Manager rejects the proposal",
     "Re-plan with the override as a constraint", _tc13, "E4"),
    ("TC14", "Exception", "No approval within 48 hours",
     "Reminder, escalate, hold previous allocation", _tc14, "E5"),
    ("TC15", "What-if", "Total budget cut 20% mid-cycle",
     "Re-run with the new total, limits respected", _tc15, "per-week budget"),
    ("TC16", "What-if", "New brand rule caps Meta at 40%",
     "Rulebook updated; agent complies at once", _tc16, "knowledge retrieval"),
]


def run_all() -> list[dict]:
    """Execute every case and return one result row each."""
    results = []
    for tc, kind, scenario, expected, factory, covers in SPECS:
        try:
            case, process, check = factory()
            passed, evidence = check(case, process)
            route = case.get("route", "-")
            status = case["status"]
            moved = shift_pp(case)
        except Exception as exc:  # a case that errors is a failure, not a crash
            passed, evidence = False, f"{type(exc).__name__}: {exc}"
            route, status, moved = "-", "error", 0.0

        results.append({
            "tc": tc, "type": kind, "scenario": scenario, "expected": expected,
            "covers": covers, "passed": passed, "route": route, "status": status,
            "shift_pp": round(moved, 1), "evidence": evidence,
        })
    return results
