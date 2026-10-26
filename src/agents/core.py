"""
The case record and the tool belt.

Two things live here, and they are the two things the Agentic Readiness Test
asks about directly.

**Memory.** The proposal claims three kinds, and all three are real:

1.  A *case record* per weekly cycle - this module's `CaseRecord`. It carries
    the week from trigger to outcome: what was pulled, what was flagged, what
    was proposed, what the guardrail changed, who approved it, what was
    executed and what came back. It is the state object the graph passes
    between stages, and it is what gets written to the audit log.
2.  The allocator's *learned posteriors*, carried across weeks inside the
    allocator object (see `src/allocator.py`).
3.  A *history of manager overrides*, held by the process runner and fed back
    as a constraint on later proposals (exception path E4).

**Tools.** Eight of them, as listed in the proposal. Every call is recorded in
the case record, so the decision trace shown in the viva is the real log the
agents wrote, not a narration added afterwards.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, TypedDict

import pandas as pd

from ..policy import Rulebook
from ..world import CHANNELS

# A fictional clock, so timestamps in the audit log are stable and the 48-hour
# approval SLA means something in a simulated run.
EPOCH = datetime(2026, 1, 5, 9, 0, 0)   # Monday 09:00, the weekly trigger


def week_timestamp(week: int, hours: float = 0.0) -> str:
    return (EPOCH + timedelta(weeks=week - 1, hours=hours)).isoformat(timespec="seconds")


class CaseRecord(TypedDict, total=False):
    """Everything known about one weekly cycle."""

    # identity
    case_id: str
    week: int
    opened_at: str
    status: str                  # running | executed | held | escalated

    # S1 - trigger and data pull
    metrics: dict[str, dict]
    reported_week: int
    missing_feeds: list[str]
    used_fallback_data: bool

    # S2 - validation and anomaly detection
    anomalies: list[dict]
    degraded: bool
    frozen_channels: list[str]
    delivery_caps: dict[str, float]

    # The total budget for this cycle. Carried on the case rather than read
    # from a constant, because finance can change it mid-season (TC15) and
    # the agent has to re-plan against the new total without any code change.
    weekly_budget: float

    # Routing decisions. These are computed and logged inside the nodes, and
    # the conditional-edge functions only read them. LangGraph keeps what a
    # node returns and discards anything an edge function writes, so a
    # decision made inside an edge would never reach the case record or the
    # audit log - and a counter incremented there would never increment.
    d1: str
    d2: str
    d3: str
    manager_route: str

    # S3 - proposal
    target: dict[str, float]
    sampled_per_lakh: dict[str, float]
    anchor: dict[str, float]

    # S4 - guardrail
    proposal: dict[str, float]
    violations: list[str]
    trim_notes: list[str]
    was_trimmed: bool
    feasible: bool

    # S5 - approval routing
    route: str                   # auto_execute | approve | escalate
    approval_reasons: list[str]
    shift_pp: float
    brief: str
    manager_decision: dict

    # S6 - execution
    executed: dict[str, float]
    executed_at: str

    # S7 - feedback and reporting
    report: str

    # cross-cutting
    previous_shares: dict[str, float]
    festive: bool
    replans: int
    escalations: list[str]
    audit: list[dict]


def new_case(week: int, previous_shares: dict[str, float], festive: bool,
             weekly_budget: float) -> CaseRecord:
    return CaseRecord(
        case_id=f"AD-{week:03d}",
        weekly_budget=float(weekly_budget),
        delivery_caps={},
        week=week,
        opened_at=week_timestamp(week),
        status="running",
        metrics={},
        missing_feeds=[],
        used_fallback_data=False,
        anomalies=[],
        degraded=False,
        frozen_channels=[],
        violations=[],
        trim_notes=[],
        was_trimmed=False,
        feasible=True,
        approval_reasons=[],
        manager_decision={},
        previous_shares=dict(previous_shares),
        festive=festive,
        replans=0,
        escalations=[],
        audit=[],
    )


# ---------------------------------------------------------------------------
# Tool belt
# ---------------------------------------------------------------------------

class ToolBelt:
    """
    The eight tools the agents call. Each method logs its own call into the
    case record before returning, which is what makes the audit trail a
    by-product of doing the work rather than something written separately and
    therefore capable of disagreeing with it.
    """

    def __init__(self, rulebook: Rulebook, allocator, latest_valid: dict | None = None):
        self.rulebook = rulebook
        self.allocator = allocator
        self.latest_valid: dict[str, dict] = latest_valid or {}
        self.notifications: list[dict] = []
        self.executions: list[dict] = []

    # -- logging ---------------------------------------------------------

    @staticmethod
    def _log(case: CaseRecord, tool: str, stage: str, summary: str,
             detail: Any = None) -> None:
        case["audit"].append({
            "case_id": case["case_id"],
            "week": case["week"],
            "at": week_timestamp(case["week"], len(case["audit"]) * 0.05),
            "stage": stage,
            "tool": tool,
            "summary": summary,
            "detail": detail if detail is not None else "",
        })

    def note(self, case: CaseRecord, stage: str, summary: str, detail: Any = None) -> None:
        """A decision by an agent, with no tool call behind it."""
        self._log(case, "decision", stage, summary, detail)

    # -- 1. performance database ------------------------------------------

    def read_performance(self, case: CaseRecord, history: pd.DataFrame,
                         week: int) -> tuple[dict, list[str]]:
        """Pull last week reported metrics. Falls back to the last valid week
        for any channel whose feed did not arrive (exception path E2)."""
        reported_week = week - 1
        metrics: dict[str, dict] = {}
        missing: list[str] = []

        # Week 1 has nothing behind it, so `history` arrives as a bare empty
        # frame with no columns at all - not just no rows.
        has_data = (not history.empty) and "channel" in history.columns
        rows = history[history["week"] == reported_week] if has_data else None

        for channel in CHANNELS:
            if rows is None or rows.empty:
                break
            row = rows[rows["channel"] == channel]
            if row.empty:
                continue
            row = row.iloc[0]

            if bool(row.get("feed_missing", False)):
                missing.append(channel)
                fallback = self._last_valid_week(history, channel, reported_week)
                if fallback is not None:
                    metrics[channel] = dict(fallback, stale=True)
                    self.latest_valid[channel] = dict(fallback)
                elif channel in self.latest_valid:
                    metrics[channel] = dict(self.latest_valid[channel], stale=True)
                continue

            clicks = float(row["clicks"]) or 1.0
            metrics[channel] = {
                "week": reported_week,
                "spend": float(row["spend"]),
                "clicks": float(row["clicks"]),
                "conversions": float(row["conversions"]),
                "revenue": float(row["revenue"]),
                "cpc": float(row["spend"]) / clicks,
                "cvr": float(row["conversions"]) / clicks,
                "share": float(row.get("share", 1.0 / len(CHANNELS))),
                "stale": False,
            }

            # Pacing: what the channel was given against what it managed to
            # spend. A platform that cannot absorb its budget silently hands
            # it back, and an allocator that never looks will keep sending
            # money to a channel that cannot use it.
            budget = float(case.get("weekly_budget") or 0.0)
            allocated_share = metrics[channel]["share"]
            if budget > 0 and allocated_share > 0:
                delivered_share = metrics[channel]["spend"] / budget
                metrics[channel]["delivered_share"] = delivered_share
                metrics[channel]["delivery_ratio"] = delivered_share / allocated_share
            self.latest_valid[channel] = dict(metrics[channel])

        self._log(case, "performance_db", "S1",
                  f"pulled week {reported_week} for {len(metrics)} channels"
                  + (f"; no feed from {', '.join(missing)}" if missing else ""),
                  {"reported_week": reported_week, "missing": missing})
        return metrics, missing

    @staticmethod
    def _last_valid_week(history: pd.DataFrame, channel: str,
                         before_week: int) -> dict | None:
        """
        Query the performance database for the most recent week this channel
        actually reported.

        Asking the database beats remembering, which is why this does not
        simply read back whatever the agent happened to see last: on the
        first cycle after a restart there is nothing in memory, and the
        fallback would silently not happen.
        """
        if history.empty or "channel" not in history.columns:
            return None

        rows = history[(history["channel"] == channel)
                       & (history["week"] < before_week)
                       & (~history["feed_missing"].astype(bool))
                       & (history["clicks"] > 0)]
        if rows.empty:
            return None

        row = rows.sort_values("week").iloc[-1]
        clicks = float(row["clicks"]) or 1.0
        return {
            "week": int(row["week"]),
            "spend": float(row["spend"]),
            "clicks": float(row["clicks"]),
            "conversions": float(row["conversions"]),
            "revenue": float(row["revenue"]),
            "cpc": float(row["spend"]) / clicks,
            "cvr": float(row["conversions"]) / clicks,
            "share": float(row.get("share", 1.0 / len(CHANNELS))),
            "stale": True,
        }

    # -- 2. policy knowledge base ------------------------------------------

    def read_policy(self, case: CaseRecord, stage: str = "S2") -> dict:
        limits = self.rulebook.data["hard_limits"]
        active_rules = [r["id"] for r in self.rulebook.data.get("brand_rules", [])
                        if r.get("active")]
        self._log(case, "policy_kb", stage,
                  f"retrieved rulebook v{self.rulebook.data['version']}: "
                  f"{limits['min_channel_share']:.0%}-{limits['max_channel_share']:.0%} "
                  f"per channel, {limits['max_weekly_shift_pp']:.0f} pp weekly cap, "
                  f"brand rules active: {', '.join(active_rules) or 'none'}",
                  {"active_brand_rules": active_rules})
        return self.rulebook.data

    # -- 3. anomaly checker -------------------------------------------------

    def scan_anomalies(self, case: CaseRecord, history: pd.DataFrame,
                       metrics: dict, missing: list[str]) -> list[dict]:
        thresholds = self.rulebook.data["anomaly_thresholds"]
        lookback = int(thresholds["lookback_weeks"])
        reported_week = case["week"] - 1
        anomalies: list[dict] = []

        for channel in missing:
            anomalies.append({
                "channel": channel, "type": "missing_feed", "severity": "high",
                "detail": f"No data feed from {channel} for week {reported_week}.",
                "exception": "E2",
            })

        # Pacing check, before the statistical ones: a channel that under-
        # delivered is a fact about last week, not a deviation to be tested.
        under_pct = float(thresholds.get("under_delivery_pct", 0.0))
        headroom = float(thresholds.get("pacing_headroom_pct", 0.0)) / 100.0
        if under_pct > 0:
            for channel in CHANNELS:
                metric = metrics.get(channel) or {}
                ratio = metric.get("delivery_ratio")
                if ratio is None or metric.get("stale") or ratio >= 1.0 - under_pct / 100.0:
                    continue
                anomalies.append({
                    "channel": channel, "type": "under_delivery", "severity": "medium",
                    "detail": (f"{channel} spent only {ratio:.0%} of the budget it was "
                               f"given in week {reported_week} "
                               f"({metric['delivered_share']:.1%} of the total against "
                               f"{metric['share']:.1%} allocated)."),
                    "exception": "E1",
                    "cap_share": round(metric["delivered_share"] * (1.0 + headroom), 6),
                })

        z_multiple = float(thresholds.get("volatility_z", 0.0))
        min_obs = int(thresholds.get("min_observations", 3))
        alpha_hat = float(thresholds.get("assumed_saturation_alpha", 1.0))
        share_ref = 1.0 / len(CHANNELS)

        if (not history.empty) and "channel" in history.columns:
            window = history[(history["week"] < reported_week)
                             & (history["week"] >= reported_week - lookback)]
            for channel in CHANNELS:
                if channel in missing or channel not in metrics:
                    continue
                base = window[(window["channel"] == channel)
                              & (~window["feed_missing"].astype(bool))
                              & (window["clicks"] > 0)]
                if base.empty:
                    continue

                # Per-week series, so the channel own volatility can be
                # measured - a pooled average hides it completely.
                base = base.sort_values("week")
                weekly_cpc = (base["spend"] / base["clicks"]).astype(float)

                # Observed conversion rate is NOT comparable across weeks as
                # it stands. Diminishing returns mean a channel run at a 45%
                # share converts at a different observed rate from the same
                # channel run at 20%, so the agent own budget moves show up
                # as channel anomalies. Normalising each week back to an equal
                # -split reference spend removes that self-inflicted signal.
                # Observed CPC needs no such correction: clicks = spend / cpc
                # exactly, so spend cancels out of it.
                weekly_cvr = (base["conversions"] / base["clicks"]).astype(float)
                weekly_cvr = weekly_cvr * (
                    base["share"].astype(float) / share_ref) ** (1.0 - alpha_hat)

                cur = metrics[channel]
                cur_cvr_norm = cur["cvr"] * (
                    cur.get("share", share_ref) / share_ref) ** (1.0 - alpha_hat)

                for metric, value, series, direction, kind, label in (
                    ("cpc", cur["cpc"], weekly_cpc, 1.0, "cpc_spike", "CPC up"),
                    ("cvr", cur_cvr_norm, weekly_cvr, -1.0, "cvr_collapse",
                     "conversion rate down"),
                ):
                    if series.empty or float(series.iloc[-1]) <= 0:
                        continue

                    # Compare the WEEK-ON-WEEK CHANGE, not the level. These
                    # channels trend by design - advantage rotates over a
                    # season - so a level sitting well away from its own
                    # trailing median is normal, and a detector built on that
                    # fires on the trend instead of on the event. A genuine
                    # shock shows up as a sudden step: a 60% jump between two
                    # consecutive weeks, where the rotation moves perhaps 20%.
                    change = (value / float(series.iloc[-1]) - 1.0) * 100.0
                    prior_changes = series.pct_change().dropna() * 100.0

                    fixed = float(thresholds["cpc_spike_pct" if metric == "cpc"
                                             else "cvr_drop_pct"])

                    # The bar scales with how jumpy this channel normally is.
                    # Dispersion uses the median absolute deviation, not the
                    # standard deviation: a two-week shock puts its own first
                    # week into the comparison window, and a standard
                    # deviation inflated by that week raises the bar high
                    # enough to hide the second week of the same shock.
                    if len(prior_changes) >= min_obs and z_multiple > 0:
                        mad = float((prior_changes - prior_changes.median()).abs().median())
                        typical_move = 1.4826 * mad
                        bar = max(fixed, z_multiple * typical_move)
                    else:
                        typical_move, bar = 0.0, fixed

                    if direction * change < bar:
                        continue

                    anomalies.append({
                        "channel": channel, "type": kind, "severity": "high",
                        "detail": (f"{channel} {label} {abs(change):.0f}% in one week, "
                                   f"past the {bar:.0f}% bar for this channel "
                                   f"(floor {fixed:.0f}%, {z_multiple:g}x its usual "
                                   f"{typical_move:.0f}% weekly move)."),
                        "exception": "E1", "change_pct": round(change, 1),
                        "bar_pct": round(bar, 1),
                    })

        self._log(case, "anomaly_checker", "S2",
                  f"{len(anomalies)} anomaly flag(s)" if anomalies else "no anomalies",
                  anomalies)
        return anomalies

    # -- 4. allocation engine ----------------------------------------------

    def propose(self, case: CaseRecord, history: pd.DataFrame, *,
                anchor: dict | None = None, anchor_weight: float = 0.0) -> dict:
        target, sampled, _missing = self.allocator.propose_target(
            case["week"], history, anchor=anchor, anchor_weight=anchor_weight)

        self._log(case, "allocator", "S3",
                  "Thompson Sampling proposed "
                  + ", ".join(f"{c} {target[c]:.0%}" for c in CHANNELS)
                  + (f" (anchored {anchor_weight:.0%} to a manager override)"
                     if anchor and anchor_weight else ""),
                  {"sampled_per_lakh": {c: round(sampled[c], 1) for c in CHANNELS},
                   "posterior_mean": {c: round(self.allocator.posteriors[c].mean, 1)
                                      for c in CHANNELS}})
        case["sampled_per_lakh"] = {c: round(sampled[c], 2) for c in CHANNELS}
        return target

    # -- 5. constraint calculator ------------------------------------------

    def enforce(self, case: CaseRecord, target: dict) -> tuple[dict, bool, list[str], list[str]]:
        pinned = {c: case["previous_shares"][c] for c in case["frozen_channels"]} or None
        caps = dict(case.get("delivery_caps") or {}) or None

        violations = [str(v) for v in self.rulebook.check(
            target, case["previous_shares"],
            festive=case["festive"], degraded=case["degraded"])]

        trimmed, feasible, notes = self.rulebook.trim(
            target, previous_shares=case["previous_shares"],
            festive=case["festive"], degraded=case["degraded"],
            pinned=pinned, caps=caps)

        self._log(case, "constraint_calculator", "S4",
                  (f"{len(violations)} violation(s), trimmed to "
                   + ", ".join(f"{c} {trimmed[c]:.0%}" for c in CHANNELS))
                  if violations else "proposal already compliant",
                  {"violations": violations, "notes": notes, "feasible": feasible,
                   "pinned": pinned or {}, "caps": caps or {}})
        return trimmed, feasible, notes, violations

    # -- 6. notifier --------------------------------------------------------

    def notify(self, case: CaseRecord, to: str, subject: str, body: str,
               stage: str = "S5") -> None:
        message = {"week": case["week"], "case_id": case["case_id"],
                   "to": to, "subject": subject, "body": body,
                   "at": week_timestamp(case["week"], 1.0)}
        self.notifications.append(message)
        self._log(case, "notifier", stage, f"notified {to}: {subject}", {"body": body})

    # -- 7. budget execution API (mock) -------------------------------------

    def execute_budget(self, case: CaseRecord, shares: dict,
                       weekly_budget: float) -> dict:
        confirmation = {
            "case_id": case["case_id"], "week": case["week"],
            "at": week_timestamp(case["week"], 2.0),
            "allocation": {c: round(shares[c], 6) for c in CHANNELS},
            "amounts_inr": {c: round(shares[c] * weekly_budget, 2) for c in CHANNELS},
            "reference": f"BUDGET-{case['week']:03d}",
        }
        self.executions.append(confirmation)
        self._log(case, "budget_api", "S6",
                  "executed " + ", ".join(f"{c} {shares[c]:.0%}" for c in CHANNELS),
                  confirmation)
        return confirmation

    # -- 8. audit logger ----------------------------------------------------

    def close_case(self, case: CaseRecord, summary: str) -> None:
        self._log(case, "audit_logger", "S7", summary,
                  {"status": case["status"], "replans": case["replans"]})
