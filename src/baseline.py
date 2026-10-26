"""
As-Is baselines, plus an oracle upper bound.

Three strategies, all with the same signature `(week, history) -> shares`, so
`world.run_allocation` can replay any of them against the identical world:

*   `StaticSplitAllocator` - the split is set once on judgment and never
    changes. This is the literal "static split" named in the proposal.

*   `MonthlyManualAllocator` - the honest As-Is from Section 6 of the
    proposal: the manager does rebalance, but only once every four weeks,
    only on the previous month of reported data, and only part of the way
    towards what the data implies, because managers are conservative and
    anchor on last month. This is the *generous* baseline, and it is the one
    the agentic run should be judged against. Beating a frozen split would
    prove very little.

*   `OracleAllocator` - cheats. It reads the latent truth for the week and
    picks the expected-conversion-maximising split inside the hard limits.
    Nothing can beat it, so it says how much headroom actually exists in this
    world. If the oracle only gains 6% over the manual baseline, then the
    proposal target of 8-12% is not reachable and we should say so in the
    report rather than tune the simulation until the target appears.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .policy import Rulebook
from .world import CHANNELS, WEEKLY_BUDGET

# The split the manager starts from, over-weighting paid search. Stated as a
# judgment call in the As-Is process, not derived from data.
INITIAL_JUDGMENT_SPLIT = {"google": 0.40, "meta": 0.35, "amazon": 0.25}

# How far the manager moves towards what the data implies. 0.0 = never moves,
# 1.0 = follows the data exactly. 0.5 reflects the anchoring described in the
# proposal pain points.
MANAGER_INERTIA = 0.50

REVIEW_EVERY_N_WEEKS = 4


class StaticSplitAllocator:
    """Set once on judgment, never revisited."""

    name = "static_split"
    description = "Fixed split held for the whole horizon (set once on judgment)"

    def __init__(self, shares: dict[str, float] | None = None):
        self.shares = dict(shares or INITIAL_JUDGMENT_SPLIT)
        total = sum(self.shares.values())
        self.shares = {c: v / total for c, v in self.shares.items()}
        self.decisions: list[dict] = []

    def __call__(self, week: int, history: pd.DataFrame) -> dict[str, float]:
        if week == 1:
            self.decisions.append({"week": week, "action": "set_initial_split",
                                   "shares": dict(self.shares)})
        return dict(self.shares)


class MonthlyManualAllocator:
    """
    The As-Is manager: reviews every four weeks, on the previous month only,
    and moves only part of the way towards what that month implies.
    """

    name = "monthly_manual"
    description = "Manual monthly reallocation on last month of reported data"

    def __init__(self, rulebook: Rulebook,
                 initial: dict[str, float] | None = None,
                 inertia: float = MANAGER_INERTIA,
                 review_every: int = REVIEW_EVERY_N_WEEKS):
        self.rulebook = rulebook
        self.current = dict(initial or INITIAL_JUDGMENT_SPLIT)
        total = sum(self.current.values())
        self.current = {c: v / total for c, v in self.current.items()}
        self.inertia = inertia
        self.review_every = review_every
        self.decisions: list[dict] = []

    def __call__(self, week: int, history: pd.DataFrame) -> dict[str, float]:
        # Reviews happen at the start of a month, and never in week 1 (there
        # is no reported month yet).
        is_review_week = week > 1 and (week - 1) % self.review_every == 0
        if not is_review_week or history.empty:
            return dict(self.current)

        lookback = history[history["week"] >= week - self.review_every]
        by_channel = lookback.groupby("channel").agg(
            conversions=("conversions", "sum"), spend=("spend", "sum")
        )
        by_channel["per_rupee"] = by_channel["conversions"] / by_channel["spend"]

        # A missing feed simply leaves that channel short of data. The manual
        # process has no degraded mode - it works with whatever came through,
        # which is exactly the weakness the agentic design addresses.
        signal = by_channel["per_rupee"].fillna(0.0)
        if signal.sum() <= 0:
            return dict(self.current)

        implied = (signal / signal.sum()).to_dict()
        moved = {c: self.current[c] + self.inertia * (implied.get(c, 0.0) - self.current[c])
                 for c in CHANNELS}

        # The manager still has to respect the published hard limits. No
        # weekly shift cap applies, because this is a monthly decision.
        trimmed, feasible, notes = self.rulebook.trim(moved, previous_shares=None)
        if feasible:
            self.decisions.append({
                "week": week, "action": "monthly_review",
                "lookback_weeks": self.review_every,
                "implied": {c: round(implied.get(c, 0.0), 4) for c in CHANNELS},
                "from": dict(self.current), "to": trimmed, "notes": notes,
            })
            self.current = trimmed

        return dict(self.current)


class OracleAllocator:
    """
    Upper bound. Reads the latent truth and maximises expected conversions
    for the week inside the hard limits. Not a strategy anyone could run -
    it exists only to measure how much headroom this world contains.
    """

    name = "oracle"
    description = "Per-week expected-conversion optimum within hard limits (upper bound)"

    def __init__(self, latent: pd.DataFrame, rulebook: Rulebook, step: float = 0.005):
        self.latent = latent
        self.rulebook = rulebook
        self.grid = self._build_grid(step)
        self.decisions: list[dict] = []

    @staticmethod
    def _build_grid(step: float) -> np.ndarray:
        """All share triples on a `step` grid that sum to 1."""
        n = int(round(1.0 / step))
        triples = [
            (g / n, m / n, (n - g - m) / n)
            for g, m in itertools.product(range(n + 1), repeat=2)
            if g + m <= n
        ]
        return np.array(triples)

    def __call__(self, week: int, history: pd.DataFrame) -> dict[str, float]:
        week_latent = self.latent[self.latent["week"] == week].set_index("channel")
        festive = bool(week_latent["festive"].iloc[0])
        bounds = self.rulebook.effective_bounds(previous_shares=None, festive=festive)

        candidates = self.grid
        for i, channel in enumerate(CHANNELS):
            keep = (candidates[:, i] >= bounds.lo[channel] - 1e-9) & \
                   (candidates[:, i] <= bounds.hi[channel] + 1e-9)
            candidates = candidates[keep]

        if len(candidates) == 0:
            return {c: 1.0 / len(CHANNELS) for c in CHANNELS}

        # Vectorised form of world.expected_conversions over the whole grid -
        # the same formula, evaluated for ~20,000 candidate splits at once.
        totals = np.zeros(len(candidates))
        s_ref = WEEKLY_BUDGET / len(CHANNELS)
        for i, channel in enumerate(CHANNELS):
            row = week_latent.loc[channel]
            cpc, cvr, alpha = float(row["cpc"]), float(row["cvr"]), float(row["alpha"])
            clicks = candidates[:, i] * WEEKLY_BUDGET / cpc
            clicks_ref = s_ref / cpc
            totals += cvr * (clicks_ref ** (1.0 - alpha)) * np.power(clicks, alpha)

        best = candidates[int(np.argmax(totals))]
        shares = {c: float(best[i]) for i, c in enumerate(CHANNELS)}

        # Guard against float drift in the grid.
        total = sum(shares.values())
        shares = {c: v / total for c, v in shares.items()}

        self.decisions.append({"week": week, "action": "oracle_optimum",
                               "shares": {c: round(v, 4) for c, v in shares.items()},
                               "expected_conversions": round(float(totals.max()), 2)})
        return shares
