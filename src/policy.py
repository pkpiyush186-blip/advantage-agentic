"""
Policy and guardrail engine - the knowledge source the Guardrail agent reads.

Two design decisions here are worth defending in the viva:

1.  The rules live in `config/policy_rulebook.json`, not in this code. The
    agent retrieves them at run time. That is what makes TC16 ("a new brand
    rule caps Meta at 40%") a data change rather than a code change, and it
    is the "knowledge retrieval" capability claimed in the proposal.

2.  This module is deterministic Python, with no LLM anywhere in it. The
    agent that proposes an allocation is never the agent that approves it,
    and the approving side must give the same answer every single time.
    A guardrail that occasionally hallucinates is not a guardrail.

The core operation is a projection: given a desired allocation, find the
nearest allocation that satisfies every bound and still sums to 100% of the
budget. If no such allocation exists, the proposal is infeasible and the
process escalates to a human (exception path E3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "policy_rulebook.json"

TOL = 1e-9

# Tolerance on "the shares must sum to 1". Deliberately looser than float
# precision: shares are rounded to six decimals when they are recorded, so a
# tolerance of 1e-6 would make a perfectly compliant allocation fail its own
# check. 1e-5 of a weekly budget of Rs 11,50,000 is Rs 11.50, which is below
# any meaningful allocation precision while still well above the rounding.
SUM_TOL = 1e-5


@dataclass(frozen=True)
class Violation:
    """One broken rule, in a form the manager brief can print verbatim."""

    rule: str
    channel: str
    observed: float
    limit: float
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass
class Bounds:
    """Effective per-channel floor and ceiling after every rule is applied."""

    lo: dict[str, float]
    hi: dict[str, float]
    sources: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return sum(self.lo.values()) <= 1.0 + TOL and sum(self.hi.values()) >= 1.0 - TOL


class Rulebook:
    """Loaded policy rulebook plus the checks that run against it."""

    def __init__(self, data: dict):
        self.data = data
        self.channels: list[str] = list(data["channels"])
        self.weekly_budget: float = float(data["budget"]["total_weekly_budget_inr"])

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> "Rulebook":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def effective_bounds(self, previous_shares: dict[str, float] | None = None, *,
                         festive: bool = False, degraded: bool = False,
                         pinned: dict[str, float] | None = None,
                         caps: dict[str, float] | None = None) -> Bounds:
        """
        Combine hard limits, active brand rules and the weekly shift cap into
        one floor and ceiling per channel.

        `pinned` freezes a channel at an exact share by collapsing its floor
        and ceiling onto that value. Exception path E1 uses it: when a channel
        shows a CPC spike or a conversion collapse, its share is held where it
        is and the allocator re-plans across the remaining channels only.
        """
        limits = self.data["hard_limits"]
        lo = {c: float(limits["min_channel_share"]) for c in self.channels}
        hi = {c: float(limits["max_channel_share"]) for c in self.channels}
        sources = ["hard_limits"]

        for rule in self.data.get("brand_rules", []):
            if not rule.get("active", False):
                continue
            if rule.get("applies_when") == "festive" and not festive:
                continue
            channel = rule["channel"]
            if channel not in lo:
                continue
            if "max_share" in rule:
                hi[channel] = min(hi[channel], float(rule["max_share"]))
            if "min_share" in rule:
                lo[channel] = max(lo[channel], float(rule["min_share"]))
            sources.append(rule["id"])

        cap = float(limits["max_weekly_shift_pp"]) / 100.0
        if degraded:
            cap *= float(self.data["degraded_mode"]["shift_cap_multiplier"])
            sources.append("degraded_mode")

        if previous_shares:
            for c in self.channels:
                prev = float(previous_shares[c])
                lo[c] = max(lo[c], prev - cap)
                hi[c] = min(hi[c], prev + cap)
            sources.append("weekly_shift_cap")

        if caps:
            # A pacing cap: a channel that could not spend what it was given
            # last week should not be given more this week.
            for c, value in caps.items():
                if c in hi:
                    hi[c] = min(hi[c], float(value))
            sources.append("pacing_caps")

        if pinned:
            for c, value in pinned.items():
                if c in lo:
                    lo[c] = hi[c] = float(value)
            sources.append("frozen_channels")

        # A floor can end up above a ceiling if a brand rule and the shift cap
        # pull in opposite directions. Keep the box non-empty per channel and
        # let the feasibility check decide whether the week can proceed.
        for c in self.channels:
            if lo[c] > hi[c]:
                lo[c] = hi[c]

        return Bounds(lo=lo, hi=hi, sources=sources)

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------

    def check(self, proposal: dict[str, float],
              previous_shares: dict[str, float] | None = None, *,
              festive: bool = False, degraded: bool = False) -> list[Violation]:
        """Return every rule the proposal breaks. Empty list means compliant."""
        violations: list[Violation] = []
        limits = self.data["hard_limits"]

        total = sum(proposal.values())
        if abs(total - 1.0) > SUM_TOL:
            violations.append(Violation(
                rule="budget_total", channel="-", observed=total, limit=1.0,
                detail=f"Shares sum to {total:.8f}, not 1.0 - the total budget must be fully allocated.",
            ))

        bounds = self.effective_bounds(previous_shares, festive=festive, degraded=degraded)

        for c in self.channels:
            share = float(proposal[c])

            if share < float(limits["min_channel_share"]) - TOL:
                violations.append(Violation(
                    rule="min_channel_share", channel=c, observed=share,
                    limit=float(limits["min_channel_share"]),
                    detail=f"{c} at {share:.1%} is below the {limits['min_channel_share']:.0%} floor.",
                ))

            if share > float(limits["max_channel_share"]) + TOL:
                violations.append(Violation(
                    rule="max_channel_share", channel=c, observed=share,
                    limit=float(limits["max_channel_share"]),
                    detail=f"{c} at {share:.1%} exceeds the {limits['max_channel_share']:.0%} ceiling.",
                ))

            for rule in self.data.get("brand_rules", []):
                if not rule.get("active", False) or rule["channel"] != c:
                    continue
                if rule.get("applies_when") == "festive" and not festive:
                    continue
                if "max_share" in rule and share > float(rule["max_share"]) + TOL:
                    violations.append(Violation(
                        rule=rule["id"], channel=c, observed=share,
                        limit=float(rule["max_share"]),
                        detail=f"{rule['id']}: {c} at {share:.1%} exceeds the {rule['max_share']:.0%} brand cap.",
                    ))
                if "min_share" in rule and share < float(rule["min_share"]) - TOL:
                    violations.append(Violation(
                        rule=rule["id"], channel=c, observed=share,
                        limit=float(rule["min_share"]),
                        detail=f"{rule['id']}: {c} at {share:.1%} is below the {rule['min_share']:.0%} brand floor.",
                    ))

            if previous_shares:
                shift = abs(share - float(previous_shares[c])) * 100.0
                cap = float(limits["max_weekly_shift_pp"])
                if degraded:
                    cap *= float(self.data["degraded_mode"]["shift_cap_multiplier"])
                if shift > cap + 1e-6:
                    violations.append(Violation(
                        rule="max_weekly_shift_pp", channel=c, observed=shift, limit=cap,
                        detail=f"{c} moves {shift:.1f} pp this week, above the {cap:.1f} pp cap.",
                    ))

        return violations

    # ------------------------------------------------------------------
    # Trimming (projection onto the feasible set)
    # ------------------------------------------------------------------

    def trim(self, proposal: dict[str, float],
             previous_shares: dict[str, float] | None = None, *,
             festive: bool = False, degraded: bool = False,
             pinned: dict[str, float] | None = None,
             caps: dict[str, float] | None = None) -> tuple[dict[str, float], bool, list[str]]:
        """
        Find the nearest compliant allocation to `proposal`.

        Returns (trimmed, feasible, notes). When `feasible` is False no
        allocation satisfies the rules and the caller must escalate (E3).
        """
        bounds = self.effective_bounds(previous_shares, festive=festive,
                                       degraded=degraded, pinned=pinned, caps=caps)
        notes: list[str] = []

        if not bounds.feasible:
            notes.append(
                f"No compliant split exists: floors sum to {sum(bounds.lo.values()):.1%} "
                f"and ceilings to {sum(bounds.hi.values()):.1%}."
            )
            return dict(proposal), False, notes

        x = {c: min(max(float(proposal[c]), bounds.lo[c]), bounds.hi[c]) for c in self.channels}

        # Redistribute whatever the clipping left over, proportionally to the
        # headroom each channel still has, until the shares sum to 1.
        for _ in range(500):
            residual = 1.0 - sum(x.values())
            if abs(residual) < 1e-12:
                break

            if residual > 0:
                headroom = {c: bounds.hi[c] - x[c] for c in self.channels}
            else:
                headroom = {c: x[c] - bounds.lo[c] for c in self.channels}

            available = sum(v for v in headroom.values() if v > 0)
            if available <= 1e-15:
                notes.append("Projection stalled with no headroom left.")
                return x, False, notes

            for c in self.channels:
                if headroom[c] > 0:
                    x[c] += residual * headroom[c] / available
            x = {c: min(max(x[c], bounds.lo[c]), bounds.hi[c]) for c in self.channels}

        for c in self.channels:
            delta = (x[c] - float(proposal[c])) * 100.0
            if abs(delta) > 0.05:
                notes.append(f"{c} trimmed by {delta:+.1f} pp to satisfy {', '.join(bounds.sources)}.")

        feasible = abs(sum(x.values()) - 1.0) < 1e-7
        return {c: round(x[c], 6) for c in self.channels}, feasible, notes

    # ------------------------------------------------------------------
    # Approval routing
    # ------------------------------------------------------------------

    def shift_pp(self, proposal: dict[str, float],
                 previous_shares: dict[str, float] | None) -> float:
        """Largest single-channel move, in percentage points."""
        if not previous_shares:
            return 0.0
        return max(abs(float(proposal[c]) - float(previous_shares[c])) * 100.0
                   for c in self.channels)

    @property
    def material_trim_pp(self) -> float:
        """How far a trim must move a channel before it counts as a trim."""
        return float(self.data["approval"].get("material_trim_pp", 0.0))

    def is_material_trim(self, target: dict[str, float],
                         trimmed: dict[str, float]) -> bool:
        """
        True when the guardrail changed the proposal enough to matter.

        Without a materiality threshold a 0.01 pp rounding correction would
        count as "the guardrail trimmed this proposal" and force a manager
        approval, which would hold straight-through processing at zero for
        reasons that have nothing to do with risk.
        """
        moved = max(abs(float(trimmed[c]) - float(target[c])) * 100.0
                    for c in self.channels)
        return moved >= self.material_trim_pp

    def approval_decision(self, proposal: dict[str, float],
                          previous_shares: dict[str, float] | None = None, *,
                          anomaly_flags: list[str] | None = None,
                          degraded: bool = False,
                          was_trimmed: bool = False,
                          after_override: bool = False,
                          feasible: bool = True) -> dict:
        """
        Decide the autonomy level for stage S5: act and report, ask the
        manager to approve, or escalate because no compliant split exists.
        """
        approval = self.data["approval"]
        reasons: list[str] = []

        shift = self.shift_pp(proposal, previous_shares)
        threshold = float(approval["auto_execute_max_shift_pp"])

        if shift > threshold + 1e-6:
            reasons.append(f"largest shift {shift:.1f} pp is above the {threshold:.0f} pp auto-execute threshold")

        always = approval.get("always_require_approval_on", [])
        if anomaly_flags and "anomaly_flag" in always:
            reasons.append(f"anomaly flagged on {', '.join(anomaly_flags)}")
        if was_trimmed and "trimmed_proposal" in always:
            reasons.append("proposal was trimmed by the guardrail agent")
        if degraded and "degraded_data" in always:
            reasons.append("the week is running on stale or incomplete data")
        if after_override and "after_override" in always:
            # Without this rule the agent can answer a rejection by proposing
            # something small enough to slip under the auto-execute threshold
            # and then move the budget without asking again. A manager who
            # said no must see what the agent does next.
            reasons.append("the manager has already overridden this cycle, so the "
                           "revised proposal goes back to them rather than "
                           "auto-executing")

        if not feasible:
            route = "escalate"
            reasons.append("no compliant allocation exists for this week")
        elif reasons:
            route = "approve"
        else:
            route = "auto_execute"

        return {
            "route": route,
            "reasons": reasons,
            "shift_pp": round(shift, 2),
            "threshold_pp": threshold,
            "sla_hours": int(approval["approval_sla_hours"]),
            "escalate_to": approval["escalation_after_sla_to"],
        }


def equal_split(channels) -> dict[str, float]:
    """Convenience: the neutral starting allocation."""
    channels = list(channels)
    return {c: 1.0 / len(channels) for c in channels}
