"""
The agents, one function per stage.

Five roles, as in Section 9.1 of the proposal:

    Orchestrator        owns the goal and the constraints, routes between
                        stages from the case state, re-plans after an
                        exception, decides when to escalate. In LangGraph it
                        is the conditional edges - the `route_*` functions -
                        plus the exception handler.
    Data Monitor        S1, S2
    Allocation (RL)     S3
    Guardrail           S4
    Reporting and Audit S5, S6, S7

Nothing here is a fixed sequence. Which node runs next depends on what the
case record says: whether a feed arrived, whether a channel is behaving
abnormally, whether the proposal complies, how big the move is, and what the
manager said. The same week can take six different paths through the graph.
"""

from __future__ import annotations

import pandas as pd

from ..world import CHANNELS, WEEKLY_BUDGET
from .core import CaseRecord, ToolBelt, week_timestamp

# How far a later week is pulled back towards a manager override, and for how
# many weeks. The proposal calls for override history to be fed back as a
# constraint; this is that, deliberately weak - the manager corrected one
# cycle, which is evidence about their preferences, not a new policy.
OVERRIDE_ANCHOR_WEIGHT = 0.15
OVERRIDE_MEMORY_WEEKS = 2

# How far a re-plan is pulled back towards the status quo after a rejection.
REJECTION_ANCHOR_WEIGHT = 0.50


class ProcessAgents:
    """The stage functions, bound to one tool belt and one allocator."""

    def __init__(self, tools: ToolBelt, *, manager_policy=None,
                 human_in_the_loop: bool = False,
                 weekly_budget: float = WEEKLY_BUDGET):
        self.tools = tools
        self.manager_policy = manager_policy
        self.human_in_the_loop = human_in_the_loop
        self.weekly_budget = weekly_budget

        # Cross-week memory: every manager override, kept for later cycles.
        self.overrides: list[dict] = []
        self.history: pd.DataFrame = pd.DataFrame()

    @property
    def max_replans(self) -> int:
        return int(self.tools.rulebook.data["approval"].get("max_replans", 2))

    # ==================================================================
    # S1 - trigger and data pull                      (Data Monitor agent)
    # ==================================================================

    def s1_trigger_and_pull(self, case: CaseRecord) -> CaseRecord:
        self.tools.note(case, "S1",
                        f"Weekly cycle opened for week {case['week']} "
                        f"(scheduled trigger, Monday 09:00).")

        metrics, missing = self.tools.read_performance(case, self.history, case["week"])

        case["metrics"] = metrics
        case["missing_feeds"] = missing
        case["reported_week"] = case["week"] - 1
        case["used_fallback_data"] = any(m.get("stale") for m in metrics.values())
        return case

    # ==================================================================
    # S2 - validate and detect anomalies              (Data Monitor agent)
    # ==================================================================

    def s2_validate_and_detect(self, case: CaseRecord) -> CaseRecord:
        self.tools.read_policy(case)

        anomalies = self.tools.scan_anomalies(
            case, self.history, case["metrics"], case["missing_feeds"])

        case["anomalies"] = anomalies
        case["degraded"] = bool(case["missing_feeds"])

        # D1, decided and logged here so it survives into the case record.
        if anomalies:
            case["d1"] = "exception"
            self.tools.note(case, "D1",
                            f"D1 = exception. {len(anomalies)} flag(s): "
                            + "; ".join(a["detail"] for a in anomalies))
        else:
            case["d1"] = "propose"
            self.tools.note(case, "D1", "D1 = clean week, proceeding to propose.")
        return case

    # -- D1 ------------------------------------------------------------

    def route_after_validation(self, case: CaseRecord) -> str:
        """D1: anomaly or data issue? Decided in S2, read here."""
        return case["d1"]

    # ==================================================================
    # Exception handling (E1, E2)                       (Orchestrator)
    # ==================================================================

    def handle_exception(self, case: CaseRecord) -> CaseRecord:
        frozen: list[str] = []
        caps: dict[str, float] = {}
        lines: list[str] = []

        for anomaly in case["anomalies"]:
            channel = anomaly["channel"]

            if anomaly["type"] == "under_delivery":
                # Pacing is handled with a ceiling, not a freeze: the channel
                # is not misbehaving, it simply cannot absorb what it was
                # given, so the budget is re-planned around what it can take.
                caps[channel] = float(anomaly["cap_share"])
                lines.append(
                    f"Pacing: {anomaly['detail']} Capping {channel} at "
                    f"{caps[channel]:.1%} this week and re-planning the rest."
                )

            elif anomaly["exception"] == "E1":
                # Freeze the affected channel and re-plan across the rest.
                if channel not in frozen:
                    frozen.append(channel)
                lines.append(
                    f"E1: {anomaly['detail']} Holding {channel} at "
                    f"{case['previous_shares'][channel]:.0%} and re-planning the "
                    f"remaining budget across the other channels."
                )

            elif anomaly["exception"] == "E2":
                cap = float(self.tools.rulebook.data["hard_limits"]["max_weekly_shift_pp"])
                scale = float(self.tools.rulebook.data["degraded_mode"]["shift_cap_multiplier"])
                have_fallback = channel in self.tools.latest_valid
                lines.append(
                    f"E2: {anomaly['detail']} "
                    + (f"Using the last valid week for {channel}. "
                       if have_fallback else "No earlier valid week to fall back on. ")
                    + f"Case marked degraded, weekly shift cap reduced from "
                      f"{cap:.0f} pp to {cap * scale:.0f} pp. Feed retry queued for "
                      f"{week_timestamp(case['week'], 24.0)}."
                )

        case["frozen_channels"] = frozen
        case["delivery_caps"] = caps
        case["escalations"] = case.get("escalations", []) + lines

        self.tools.note(case, "EXC", "Exception path taken.", lines)
        self.tools.notify(
            case, to="marketing.manager@advantage.example",
            subject=f"[{case['case_id']}] Week {case['week']}: anomaly detected",
            body="\n".join(lines), stage="EXC")
        return case

    # ==================================================================
    # S3 - propose allocation                      (Allocation agent, RL)
    # ==================================================================

    def s3_propose(self, case: CaseRecord) -> CaseRecord:
        anchor, weight = self._anchor_for(case)

        target = self.tools.propose(case, self.history,
                                    anchor=anchor, anchor_weight=weight)

        if case["frozen_channels"]:
            target = self._respect_frozen(case, target)

        case["target"] = target
        if anchor:
            case["anchor"] = anchor
        return case

    def _anchor_for(self, case: CaseRecord) -> tuple[dict | None, float]:
        """Which constraint, if any, the re-plan or override memory imposes."""
        decision = case.get("manager_decision") or {}

        if decision.get("action") == "modify" and decision.get("shares"):
            # The manager said exactly what they want. Re-plan onto it - and
            # note it still has to clear the guardrail below.
            return dict(decision["shares"]), 1.0

        if decision.get("action") == "reject":
            # No alternative given, so the constraint is "less than that":
            # pull the re-plan halfway back towards the status quo.
            return dict(case["previous_shares"]), REJECTION_ANCHOR_WEIGHT

        recent = [o for o in self.overrides
                  if 0 < case["week"] - o["week"] <= OVERRIDE_MEMORY_WEEKS]
        if recent:
            return dict(recent[-1]["shares"]), OVERRIDE_ANCHOR_WEIGHT

        return None, 0.0

    def _respect_frozen(self, case: CaseRecord, target: dict) -> dict:
        """Hold frozen channels at their current share, re-spread the rest."""
        held = {c: case["previous_shares"][c] for c in case["frozen_channels"]}
        remaining = 1.0 - sum(held.values())
        movable = [c for c in CHANNELS if c not in held]

        free_weight = sum(target[c] for c in movable)
        if free_weight <= 0 or remaining <= 0:
            return dict(case["previous_shares"])

        out = dict(held)
        for c in movable:
            out[c] = remaining * target[c] / free_weight
        return out

    # ==================================================================
    # S4 - guardrail check                             (Guardrail agent)
    # ==================================================================

    def s4_guardrail(self, case: CaseRecord) -> CaseRecord:
        trimmed, feasible, notes, violations = self.tools.enforce(case, case["target"])

        case["proposal"] = trimmed
        case["feasible"] = feasible
        case["trim_notes"] = notes
        case["violations"] = violations
        case["was_trimmed"] = self.tools.rulebook.is_material_trim(case["target"], trimmed)

        # D2, decided and logged here for the same reason as D1.
        if not feasible:
            case["d2"] = "escalate"
            self.tools.note(case, "D2",
                            "D2 = infeasible. E3: no split satisfies the rulebook this "
                            "week, so the agent will not guess - escalating.")
        else:
            case["d2"] = "route"
            if case["was_trimmed"]:
                self.tools.note(case, "D2",
                                "D2 = trimmed and re-checked, now compliant (E3). "
                                + " ".join(notes))
            else:
                self.tools.note(case, "D2", "D2 = compliant, no trim needed.")
        return case

    # -- D2 ------------------------------------------------------------

    def route_after_guardrail(self, case: CaseRecord) -> str:
        """D2: does a compliant allocation exist at all? Decided in S4."""
        return case["d2"]

    # ==================================================================
    # S5 - approval routing                   (Reporting and Audit agent)
    # ==================================================================

    def s5_route_approval(self, case: CaseRecord) -> CaseRecord:
        decision = self.tools.rulebook.approval_decision(
            case["proposal"], case["previous_shares"],
            anomaly_flags=[a["channel"] for a in case["anomalies"]] or None,
            degraded=case["degraded"], was_trimmed=case["was_trimmed"],
            after_override=case["replans"] > 0,
            feasible=case["feasible"])

        case["route"] = decision["route"]
        case["approval_reasons"] = decision["reasons"]
        case["shift_pp"] = decision["shift_pp"]
        case["brief"] = self.build_brief(case, decision)

        if decision["route"] == "approve":
            self.tools.notify(
                case, to="marketing.manager@advantage.example",
                subject=f"[{case['case_id']}] Week {case['week']}: approval needed "
                        f"({decision['shift_pp']:.1f} pp shift)",
                body=case["brief"])

        case["d3"] = {"escalate": "escalate",
                      "approve": "human"}.get(decision["route"], "execute")

        self.tools.note(case, "S5",
                        f"D3 = {case['d3']} (route {decision['route']})"
                        + (f" because {'; '.join(decision['reasons'])}"
                           if decision["reasons"] else " - within all thresholds"))
        return case

    # -- D3 ------------------------------------------------------------

    def route_after_approval(self, case: CaseRecord) -> str:
        """D3: can the agent act, or does a human decide? Decided in S5."""
        return case["d3"]

    # ==================================================================
    # Human in the loop
    # ==================================================================

    def s5_await_manager(self, case: CaseRecord) -> CaseRecord:
        if self.human_in_the_loop:
            from langgraph.types import interrupt
            # The payload uses the same key names as the case record, so a
            # ManagerPolicy can answer it unchanged - which is what lets the
            # dashboard fast-forward weeks through the identical code path a
            # human would take.
            decision = interrupt({
                "case_id": case["case_id"],
                "week": case["week"],
                "brief": case["brief"],
                "previous_shares": dict(case["previous_shares"]),
                "proposal": dict(case["proposal"]),
                "target": dict(case.get("target") or case["proposal"]),
                "reasons": list(case["approval_reasons"]),
                "anomalies": list(case["anomalies"]),
                "frozen_channels": list(case["frozen_channels"]),
                "degraded": case["degraded"],
                "was_trimmed": case["was_trimmed"],
                "trim_notes": list(case["trim_notes"]),
                "shift_pp": case.get("shift_pp", 0.0),
                "replans": case["replans"],
                "options": ["approve", "modify", "reject"],
            })
        else:
            decision = self.manager_policy(case) if self.manager_policy \
                else {"action": "approve", "note": "auto-approved (no manager policy)"}

        case["manager_decision"] = dict(decision)

        action = decision.get("action", "approve")
        if action in ("modify", "reject"):
            self.overrides.append({
                "week": case["week"], "action": action,
                "shares": dict(decision.get("shares") or case["proposal"]),
                "note": decision.get("note", ""),
            })

        self.tools.note(case, "S5",
                        f"Manager decision: {action}."
                        + (f" {decision.get('note')}" if decision.get("note") else ""),
                        decision)

        # The re-plan counter is incremented HERE, inside the node. Doing it
        # in the edge function below left it permanently at zero, so the
        # manager was asked the same question forever and the graph looped
        # until it ran out of recursion.
        if action == "timeout":
            case["manager_route"] = "timeout"

        elif action in ("modify", "reject"):
            if case["replans"] >= self.max_replans:
                case["manager_route"] = "hold"
                self.tools.note(case, "E4",
                                f"E4: {case['replans']} re-plans already this cycle - "
                                "holding the previous allocation rather than looping.")
            else:
                case["replans"] = case["replans"] + 1
                case["manager_route"] = "replan"
                self.tools.note(case, "E4",
                                f"E4: manager {action}. Override logged and fed back as "
                                f"a constraint; re-planning S3-S5 "
                                f"(attempt {case['replans']} of {self.max_replans}).")
        else:
            case["manager_route"] = "execute"

        return case

    # -- manager decision routing (E4, E5) ------------------------------

    def route_after_manager(self, case: CaseRecord) -> str:
        """E4 / E5 routing. Decided in the node above, read here."""
        return case["manager_route"]

    # ==================================================================
    # E5 - no response within the SLA
    # ==================================================================

    def handle_timeout(self, case: CaseRecord) -> CaseRecord:
        approval = self.tools.rulebook.data["approval"]
        sla = int(approval["approval_sla_hours"])
        senior = approval["escalation_after_sla_to"]

        self.tools.notify(
            case, to="marketing.manager@advantage.example",
            subject=f"[{case['case_id']}] Reminder: approval outstanding",
            body=f"The week {case['week']} allocation has been waiting {sla} hours.",
            stage="E5")
        self.tools.notify(
            case, to=senior,
            subject=f"[{case['case_id']}] Escalated after {sla}h with no response",
            body=case["brief"], stage="E5")

        case["escalations"] = case.get("escalations", []) + [
            f"E5: no manager response within {sla}h. Reminder sent, escalated to "
            f"{senior}, previous allocation held."
        ]
        self.tools.note(case, "E5",
                        f"E5: SLA of {sla}h breached. Reminder then escalation to "
                        f"{senior}; holding the previous allocation.")
        return case

    # ==================================================================
    # E3 - escalation when nothing complies
    # ==================================================================

    def handle_escalation(self, case: CaseRecord) -> CaseRecord:
        senior = self.tools.rulebook.data["approval"]["escalation_after_sla_to"]
        reason = ("no compliant allocation exists for this week"
                  if not case["feasible"] else "; ".join(case["approval_reasons"]))

        self.tools.notify(
            case, to=senior,
            subject=f"[{case['case_id']}] Week {case['week']} escalated",
            body=f"{reason}\n\n{case.get('brief', '')}", stage="E3")

        case["status"] = "escalated"
        case["escalations"] = case.get("escalations", []) + [f"E3: {reason}"]
        return case

    # ==================================================================
    # S6 - execute
    # ==================================================================

    def s6_execute(self, case: CaseRecord) -> CaseRecord:
        decision = case.get("manager_decision") or {}
        shares = dict(case["proposal"])

        case["executed"] = shares
        case["executed_at"] = week_timestamp(case["week"], 2.0)
        case["status"] = "executed"

        self.tools.execute_budget(
            case, shares, float(case.get("weekly_budget") or self.weekly_budget))

        if decision.get("action") == "approve":
            self.tools.note(case, "S6", "Executed after manager approval.")
        elif not decision:
            self.tools.note(case, "S6",
                            "Executed under act-then-report autonomy "
                            "(within the auto-execute threshold).")
        return case

    # ==================================================================
    # Hold - the safe fallback
    # ==================================================================

    def hold_previous(self, case: CaseRecord) -> CaseRecord:
        case["executed"] = dict(case["previous_shares"])
        case["executed_at"] = week_timestamp(case["week"], 2.0)
        if case["status"] != "escalated":
            case["status"] = "held"

        self.tools.note(case, "HOLD",
                        "Holding last week allocation unchanged. No budget moved.")
        return case

    # ==================================================================
    # S7 - feedback and report
    # ==================================================================

    def s7_feedback_and_report(self, case: CaseRecord) -> CaseRecord:
        moves = ", ".join(
            f"{c} {case['previous_shares'][c]:.0%} -> {case['executed'][c]:.0%}"
            for c in CHANNELS)

        report = [
            f"Week {case['week']} ({case['case_id']}) - {case['status']}",
            f"Allocation: {moves}",
        ]
        if case["anomalies"]:
            report.append("Anomalies: " + "; ".join(a["detail"] for a in case["anomalies"]))
        if case["was_trimmed"]:
            report.append("Guardrail: " + " ".join(case["trim_notes"]))
        if case["approval_reasons"]:
            report.append("Approval: " + "; ".join(case["approval_reasons"]))
        if case["escalations"]:
            report.append("Escalations: " + " | ".join(case["escalations"]))

        case["report"] = "\n".join(report)
        self.tools.close_case(case, f"Case closed as {case['status']}.")
        return case

    # ==================================================================
    # The manager brief
    # ==================================================================

    def build_brief(self, case: CaseRecord, decision: dict) -> str:
        """
        Plain-language rationale for the manager. Deliberately templated and
        deterministic: the numbers in a brief that authorises spending should
        come from the case record, not from a language model. An LLM can
        rewrite this text more fluently (see `narrate` in graph.py), but it
        never supplies a figure and never makes the decision.
        """
        posteriors = self.tools.allocator.posteriors
        budget = float(case.get("weekly_budget") or self.weekly_budget)
        lines = [
            f"Week {case['week']} budget recommendation ({case['case_id']})",
            "",
            f"Total for the week: Rs {budget:,.0f}"
            + (f"  ({(budget / self.weekly_budget - 1) * 100:+.0f}% against the "
               f"standard weekly budget)" if abs(budget - self.weekly_budget) > 1 else ""),
            "",
            "Proposed split:",
        ]
        for c in CHANNELS:
            prev, prop = case["previous_shares"][c], case["proposal"][c]
            lines.append(
                f"  {c:<8} {prev:>5.1%} -> {prop:>5.1%} ({(prop - prev) * 100:+.1f} pp)"
                f"   recent performance {posteriors[c].mean:.0f} conversions per lakh"
            )

        lines += ["", f"Largest move: {decision['shift_pp']:.1f} pp "
                      f"(auto-execute threshold is {decision['threshold_pp']:.0f} pp)."]

        if case["anomalies"]:
            lines += ["", "Flagged this week:"]
            lines += [f"  - {a['detail']}" for a in case["anomalies"]]
        if case["frozen_channels"]:
            lines.append(f"  - Held frozen: {', '.join(case['frozen_channels'])}.")
        if case["was_trimmed"]:
            lines += ["", "Guardrail adjustments:"]
            lines += [f"  - {n}" for n in case["trim_notes"]]
        if decision["reasons"]:
            lines += ["", "Why this needs your approval:"]
            lines += [f"  - {r}" for r in decision["reasons"]]

        lines += ["", f"Reply approve / modify / reject within "
                      f"{decision['sla_hours']}h, after which this escalates to "
                      f"{decision['escalate_to']} and the current split is held."]
        return "\n".join(lines)
