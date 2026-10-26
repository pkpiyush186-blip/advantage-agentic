"""
The reinforcement-learning allocation engine: discounted Thompson Sampling
over a Gamma-Poisson model of conversions per rupee.

Why this model
--------------
Conversions arrive as counts from spend, so the natural likelihood is Poisson:
observing k conversions from a spend of S rupees on channel c is
Poisson(lambda_c * S), where lambda_c is that channel's conversions per rupee.
The conjugate prior for a Poisson rate is the Gamma distribution, so the
posterior after observing (k, S) is simply

    lambda_c ~ Gamma(shape = a0 + sum k, rate = b0 + sum S)

There is no gradient step, no neural network and nothing to train. That is the
point: the update is one line of arithmetic, it is exact rather than
approximate, and a student can derive it on a whiteboard in a viva. Spend is
carried in units of one lakh so lambda reads as "conversions per lakh spent"
(around 145 in this world) instead of a number with four leading zeros.

Why discounting
---------------
The world is non-stationary - channel advantage rotates. A posterior that
accumulates all 26 weeks of history converges and then stops adapting, which
is exactly the failure mode of the As-Is monthly manager, just slower. So
every week the accumulated evidence is decayed towards the prior by a factor
`discount` before the new week is added:

    a <- a0 + discount * (a - a0) + k
    b <- b0 + discount * (b - b0) + S

discount = 0.75 gives an effective memory of about four weeks. The decay
applies to *every* channel each week, including any channel whose data feed
did not arrive. That is deliberate: a channel we heard nothing about should
become less certain, not stay frozen, and the widening posterior makes the
allocator more willing to probe it again.

From posterior to a budget split
--------------------------------
Thompson Sampling draws one sample per channel from its posterior and acts as
if that draw were the truth. Acting here means choosing shares, not picking a
single arm, so the sampled values are turned into a split by

    share_c  proportional to  (sampled lambda_c) ** concentration

`concentration` controls how hard the allocator leans into the channel it
currently believes is best. If the agent knew the diminishing-returns exponent
alpha of the response curve, the exact optimum would be
concentration = 1 / (1 - alpha). It does not know alpha - alpha is a property
of the world, not something the agent observes - so concentration is treated
as a hyperparameter and tuned on separate training worlds, never on the world
used for the final result. See scripts/03_run_allocator.py.

The result is then projected onto the policy set by the guardrail (floors,
ceilings, brand rules, weekly shift cap). Two consequences worth noting in the
report:

*   The 15% per-channel floor doubles as the exploration guarantee. No channel
    can ever be starved of spend, so no channel can ever go dark and stop
    producing the data needed to re-evaluate it. The policy and the learning
    requirement happen to agree.
*   The allocator proposes; it never executes. Every proposal goes through the
    guardrail and the approval rules before any budget moves.

Known approximation
-------------------
lambda_c is estimated from the spend the channel actually ran, so it already
embeds whatever saturation applied at that share. Treating it as a constant
per-rupee rate when re-allocating is therefore an approximation: moving a
channel from a 20% to a 45% share will not deliver the same conversions per
rupee it delivered at 20%. The diminishing-returns effect is mild over the
range the shift cap allows, and the `concentration` tuning absorbs part of it,
but it is a real limitation and belongs in the report rather than in a
footnote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .policy import Rulebook
from .world import CHANNELS, FESTIVE_WEEKS

LAKH = 100_000.0

# Defaults. `concentration` and `discount` are overwritten by the tuned values
# in scripts/03_run_allocator.py; these are the starting points for the sweep.
DEFAULT_CONCENTRATION = 3.0
DEFAULT_DISCOUNT = 0.75

# Weakly informative prior: 150 conversions per lakh, carrying the weight of
# half a lakh of spend - a small fraction of one channel-week, so the data
# takes over almost immediately.
PRIOR_MEAN_PER_LAKH = 150.0
PRIOR_WEIGHT_LAKH = 0.5


@dataclass
class Posterior:
    """Gamma posterior over one channel conversions-per-lakh rate."""

    shape: float   # a: prior pseudo-conversions + discounted observed conversions
    rate: float    # b: prior pseudo-spend + discounted observed spend, in lakh

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.shape)) / self.rate

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.gamma(shape=self.shape, scale=1.0 / self.rate))

    def as_dict(self) -> dict:
        return {"shape": round(self.shape, 4), "rate": round(self.rate, 6),
                "mean": round(self.mean, 2), "sd": round(self.sd, 3)}


class ThompsonSamplingAllocator:
    """
    Discounted Thompson Sampling over conversions per rupee, projected onto
    the policy set by the guardrail.

    Signature matches every other strategy - (week, history) -> shares - so
    world.run_allocation replays it against the identical world as the
    baselines, with no separate measurement code.
    """

    name = "thompson_sampling"
    description = "Discounted Gamma-Poisson Thompson Sampling, projected onto the policy set"

    def __init__(self, rulebook: Rulebook, *,
                 concentration: float = DEFAULT_CONCENTRATION,
                 discount: float = DEFAULT_DISCOUNT,
                 prior_mean_per_lakh: float = PRIOR_MEAN_PER_LAKH,
                 prior_weight_lakh: float = PRIOR_WEIGHT_LAKH,
                 seed: int = 7,
                 start_shares: dict[str, float] | None = None,
                 festive_weeks=FESTIVE_WEEKS):
        self.rulebook = rulebook
        self.concentration = float(concentration)
        self.discount = float(discount)
        self.rng = np.random.default_rng(seed)
        self.festive_weeks = set(festive_weeks)

        self.a0 = float(prior_mean_per_lakh) * float(prior_weight_lakh)
        self.b0 = float(prior_weight_lakh)
        self.posteriors = {c: Posterior(self.a0, self.b0) for c in CHANNELS}

        # The agent starts from whatever split is live today, so week 1 is a
        # move from the real world rather than from a blank slate.
        self.current = dict(start_shares or {c: 1.0 / len(CHANNELS) for c in CHANNELS})
        total = sum(self.current.values())
        self.current = {c: v / total for c, v in self.current.items()}

        self._weeks_consumed: set[int] = set()
        self.decisions: list[dict] = []

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _decay(self) -> None:
        """A week passes: decay all evidence towards the prior."""
        for p in self.posteriors.values():
            p.shape = self.a0 + self.discount * (p.shape - self.a0)
            p.rate = self.b0 + self.discount * (p.rate - self.b0)

    def _absorb(self, rows: pd.DataFrame) -> list[str]:
        """
        Add one week of reported results. Returns the channels whose feed did
        not arrive, which is what puts the week into degraded mode (E2).
        """
        missing: list[str] = []

        for _, row in rows.iterrows():
            channel = row["channel"]
            if bool(row.get("feed_missing", False)):
                missing.append(channel)
                continue
            p = self.posteriors[channel]
            p.shape += float(row["conversions"])
            p.rate += float(row["spend"]) / LAKH

        return missing

    def _update_from(self, history: pd.DataFrame, week: int) -> list[str]:
        """Consume every reported week not yet seen, oldest first."""
        missing: list[str] = []
        if history.empty:
            return missing

        pending = sorted(w for w in history["week"].unique()
                         if w < week and w not in self._weeks_consumed)

        for w in pending:
            self._decay()
            missing = self._absorb(history[history["week"] == w])
            self._weeks_consumed.add(int(w))

        return missing

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def propose_target(self, week: int, history: pd.DataFrame, *,
                       anchor: dict[str, float] | None = None,
                       anchor_weight: float = 0.0) -> tuple[dict, dict, list[str]]:
        """
        Learn from everything reported so far and propose a target split.

        This is the Allocation agent acting alone: it proposes and nothing
        more. Checking and trimming belong to the Guardrail agent, which is
        the separation of duties the proposal argues for - the agent that
        proposes an allocation is never the one that approves it.

        `anchor` pulls the proposal towards a split the manager indicated on
        a previous cycle (exception path E4), with `anchor_weight` in [0, 1]
        controlling how far. Returns (target, sampled_values, missing_feeds).
        """
        missing_feeds = self._update_from(history, week)

        sampled = {c: self.posteriors[c].sample(self.rng) for c in CHANNELS}

        weights = {c: max(sampled[c], 1e-9) ** self.concentration for c in CHANNELS}
        total_weight = sum(weights.values())
        target = {c: weights[c] / total_weight for c in CHANNELS}

        if anchor and anchor_weight > 0.0:
            w = min(max(anchor_weight, 0.0), 1.0)
            blended = {c: (1.0 - w) * target[c] + w * float(anchor[c]) for c in CHANNELS}
            total = sum(blended.values())
            target = {c: v / total for c, v in blended.items()}

        return target, sampled, missing_feeds

    def __call__(self, week: int, history: pd.DataFrame) -> dict[str, float]:
        target, sampled, missing_feeds = self.propose_target(week, history)
        degraded = bool(missing_feeds)

        festive = week in self.festive_weeks
        trimmed, feasible, notes = self.rulebook.trim(
            target, previous_shares=self.current, festive=festive, degraded=degraded)

        # Materiality comes from the rulebook, not from this code: a
        # rounding-level correction is not a trim the manager needs to see.
        was_trimmed = self.rulebook.is_material_trim(target, trimmed)

        decision = self.rulebook.approval_decision(
            trimmed, self.current, anomaly_flags=missing_feeds or None,
            degraded=degraded, was_trimmed=was_trimmed, feasible=feasible)

        if not feasible:
            # Nothing compliant exists: hold the previous allocation and let a
            # human decide. Never guess when the rules cannot be satisfied.
            trimmed = dict(self.current)

        self.decisions.append({
            "week": week,
            "posteriors": {c: self.posteriors[c].as_dict() for c in CHANNELS},
            "sampled_per_lakh": {c: round(sampled[c], 2) for c in CHANNELS},
            "target": {c: round(target[c], 4) for c in CHANNELS},
            "proposed": {c: round(trimmed[c], 4) for c in CHANNELS},
            "previous": {c: round(self.current[c], 4) for c in CHANNELS},
            "shift_pp": decision["shift_pp"],
            "route": decision["route"],
            "reasons": decision["reasons"],
            "was_trimmed": was_trimmed,
            "degraded": degraded,
            "missing_feeds": missing_feeds,
            "festive": festive,
            "feasible": feasible,
            "trim_notes": notes,
        })

        self.current = dict(trimmed)
        return dict(trimmed)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def decisions_frame(self) -> pd.DataFrame:
        """Flat per-week table for the audit log and the dashboard."""
        rows = []
        for d in self.decisions:
            row = {
                "week": d["week"], "route": d["route"], "shift_pp": d["shift_pp"],
                "degraded": d["degraded"], "was_trimmed": d["was_trimmed"],
                "festive": d["festive"],
                "reasons": "; ".join(d["reasons"]),
            }
            for c in CHANNELS:
                row[f"posterior_mean_{c}"] = d["posteriors"][c]["mean"]
                row[f"posterior_sd_{c}"] = d["posteriors"][c]["sd"]
                row[f"sampled_{c}"] = d["sampled_per_lakh"][c]
                row[f"target_{c}"] = d["target"][c]
                row[f"proposed_{c}"] = d["proposed"][c]
            rows.append(row)
        return pd.DataFrame(rows)

    def process_counts(self) -> dict:
        """Counts the To-Be process KPIs are derived from."""
        n = len(self.decisions)
        auto = sum(1 for d in self.decisions if d["route"] == "auto_execute")
        approve = sum(1 for d in self.decisions if d["route"] == "approve")
        escalate = sum(1 for d in self.decisions if d["route"] == "escalate")
        return {
            "cycles": n,
            "auto_executed": auto,
            "sent_for_approval": approve,
            "escalated": escalate,
            "stp_rate": round(auto / n, 4) if n else 0.0,
            "degraded_weeks": sum(1 for d in self.decisions if d["degraded"]),
            "trimmed_weeks": sum(1 for d in self.decisions if d["was_trimmed"]),
            "infeasible_weeks": sum(1 for d in self.decisions if not d["feasible"]),
        }


class GreedyRecencyAllocator(ThompsonSamplingAllocator):
    """
    Ablation: identical machinery, but acts on the posterior *mean* instead of
    a sample - so no exploration at all.

    This exists to answer a question the viva will almost certainly ask: how
    much of the gain comes from Thompson Sampling specifically, and how much
    comes simply from recency-weighted estimation? Reporting the ablation is
    more honest than claiming the whole gain for the bandit.
    """

    name = "greedy_recency"
    description = "Same model, acts on the posterior mean (no sampling, no exploration)"

    def __call__(self, week: int, history: pd.DataFrame) -> dict[str, float]:
        saved = self.rng
        try:
            # Replace sampling with the posterior mean.
            self.rng = _MeanStub(self.posteriors)
            return super().__call__(week, history)
        finally:
            self.rng = saved


class _MeanStub:
    """Makes Posterior.sample return the posterior mean."""

    def __init__(self, posteriors):
        self._posteriors = posteriors

    def gamma(self, shape: float, scale: float) -> float:
        return shape * scale
