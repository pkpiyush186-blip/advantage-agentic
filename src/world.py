"""
AdVantage Retail - simulated digital advertising world (ground truth).

The world is a 26-week sequence of latent weekly parameters (CPC, CTR,
conversion rate, AOV) for three paid channels. The latent parameters are
generated once from a fixed seed and then frozen, so any budget allocation -
the manual As-Is baseline or the agentic To-Be - can be replayed against the
identical world. That is what makes the before/after comparison a fair test
rather than two runs of two different simulations.

Response model
--------------
For a weekly spend S on a channel in week w:

    clicks      = S / cpc
    clicks_ref  = s_ref / cpc          (s_ref = equal split of the budget)
    conversions = cvr * clicks_ref**(1 - alpha) * clicks**alpha

alpha < 1 produces diminishing returns: doubling spend less than doubles
conversions. This matters. Without it the best allocation would always be
"put the whole budget in the strongest channel of the week", there would be no
interior optimum, and nothing for an allocator to learn. At S = s_ref the
model collapses to conversions = cvr * clicks, i.e. the quoted benchmark rate.

Observed conversions are drawn from Poisson(mean), so the agent sees noisy
feedback and never the latent truth - which is the whole reason a
Thompson Sampling allocator has something to do.

Rotating channel advantage
--------------------------
On top of the random walk, each channel carries a slow cycle in its
conversion rate with a different phase, so which channel is strongest keeps
changing through the horizon. This is a deliberate property of the world, not
an emergent one, and it has to be declared: in practice channel effectiveness
does rotate with seasonality, creative fatigue and competitor bidding, but we
have built a world in which that rotation is present and material. A result
measured here shows that weekly reallocation captures rotating advantage when
rotating advantage exists. It is not evidence about the size of the effect in
a real ad account.

All rates and shock dates are illustrative. See README.md for sources,
assumptions and limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Scenario constants
# --------------------------------------------------------------------------

CHANNELS = ("google", "meta", "amazon")
N_WEEKS = 26

# Proposal: Rs 50 lakh per month. 50,00,000 * 12 / 52 = 11,53,846 per week,
# rounded to a clean weekly cycle budget.
WEEKLY_BUDGET = 1_150_000.0

# Base parameters, calibrated so conversions per rupee at the reference spend
# are close to tied across the three channels (0.00144 - 0.00150). The
# near-tie is deliberate: it is what makes the explore/exploit trade-off real.
# If one channel were obviously best, a static split would be fine and the
# project would not need an allocator at all.
# cvr is set so that cvr / cpc = 0.0015 for all three: an exact tie at the
# reference spend.
BASE = {
    "google": {"cpc": 18.0, "ctr": 0.028, "cvr": 0.027, "aov": 2400.0, "alpha": 0.88},
    "meta":   {"cpc": 12.0, "ctr": 0.016, "cvr": 0.018, "aov": 2100.0, "alpha": 0.82},
    "amazon": {"cpc": 22.0, "ctr": 0.009, "cvr": 0.033, "aov": 1800.0, "alpha": 0.90},
}

# Weekly drift: mean-reverting multiplicative random walk. Reversion is fast
# enough that the walk cannot wander off and permanently crown one channel -
# otherwise a static split would quietly be the right answer.
CPC_SIGMA, CVR_SIGMA = 0.050, 0.070
MEAN_REVERSION = 0.45

# Slow rotation in channel effectiveness, one phase per channel. See the
# module docstring: this is declared, not discovered.
#
# The amplitude is the single most consequential assumption in the project,
# because the gain any allocator can earn is bounded by how much the channels
# actually differ week to week. It is set from realism, NOT from the target in
# the proposal: 0.45 means a channel conversion rate swings roughly +/-45%
# around its base over the cycle, so in a given week the strongest channel can
# be around twice as efficient per rupee as the weakest. Weekly swings of that
# size are ordinary in a live D2C account once CPC and conversion rate both
# move. Whatever headroom this produces is then measured and reported, not
# adjusted to match an expectation - scripts/02_run_baseline.py prints the
# oracle ceiling precisely so the target can be checked against it.
ROTATION_PERIOD_WEEKS = 13.0
ROTATION_AMPLITUDE = 0.45
CHANNEL_PHASE = {"google": 0.0, "meta": 1.0 / 3.0, "amazon": 2.0 / 3.0}

# Festive season (a Diwali-style window): demand and competition both rise.
FESTIVE_WEEKS = range(19, 24)
FESTIVE_CVR_UPLIFT, FESTIVE_CPC_UPLIFT = 1.35, 1.25

# Injected shocks. Each one exists to drive a specific exception test case.
# (week, channel, field, multiplier, weeks_active, label, test case)
SHOCKS = [
    (8,  "google", "cpc", 1.60, 2, "google_cpc_spike",    "TC09"),
    (14, "meta",   "cvr", 0.40, 2, "meta_cvr_collapse",   "TC10"),
    (17, "amazon", None,  None, 1, "amazon_feed_missing", "TC11"),
]

SEED = 20260115


@dataclass(frozen=True)
class WeekChannel:
    """Latent truth for one channel in one week."""

    week: int
    channel: str
    cpc: float
    ctr: float
    cvr: float
    aov: float
    alpha: float
    festive: bool
    feed_missing: bool
    shock: str  # "" when nothing was injected


# --------------------------------------------------------------------------
# World generation
# --------------------------------------------------------------------------

def generate_latent(seed: int = SEED, n_weeks: int = N_WEEKS) -> pd.DataFrame:
    """Generate the frozen 26-week ground truth. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    rows: list[WeekChannel] = []

    # Drift state, carried week to week as a multiplier on the base value.
    drift = {c: {"cpc": 1.0, "cvr": 1.0} for c in CHANNELS}

    for week in range(1, n_weeks + 1):
        festive = week in FESTIVE_WEEKS

        for channel in CHANNELS:
            base = BASE[channel]

            for field, sigma in (("cpc", CPC_SIGMA), ("cvr", CVR_SIGMA)):
                noise = rng.normal(0.0, sigma)
                pull_back = -MEAN_REVERSION * np.log(drift[channel][field])
                drift[channel][field] *= float(np.exp(pull_back + noise))

            rotation = 1.0 + ROTATION_AMPLITUDE * np.sin(
                2.0 * np.pi * ((week - 1) / ROTATION_PERIOD_WEEKS + CHANNEL_PHASE[channel])
            )

            cpc = base["cpc"] * drift[channel]["cpc"]
            cvr = base["cvr"] * drift[channel]["cvr"] * rotation

            if festive:
                cpc *= FESTIVE_CPC_UPLIFT
                cvr *= FESTIVE_CVR_UPLIFT

            label, feed_missing = "", False
            for s_week, s_channel, field, mult, duration, s_label, _tc in SHOCKS:
                if s_channel != channel or not (s_week <= week < s_week + duration):
                    continue
                label = s_label
                if field is None:
                    feed_missing = True
                elif field == "cpc":
                    cpc *= mult
                elif field == "cvr":
                    cvr *= mult

            rows.append(
                WeekChannel(
                    week=week,
                    channel=channel,
                    cpc=round(cpc, 4),
                    ctr=round(base["ctr"] * float(rng.normal(1.0, 0.05)), 6),
                    cvr=round(cvr, 6),
                    aov=round(base["aov"] * float(rng.normal(1.0, 0.03)), 2),
                    alpha=base["alpha"],
                    festive=festive,
                    feed_missing=feed_missing,
                    shock=label,
                )
            )

    return pd.DataFrame([asdict(r) for r in rows])


# --------------------------------------------------------------------------
# Observation: what a given spend actually produces
# --------------------------------------------------------------------------

def observe(latent_row, spend: float, rng: np.random.Generator,
            weekly_budget: float = WEEKLY_BUDGET) -> dict:
    """
    Run one channel-week at a given spend and return the observed metrics.

    `rng` is passed in so a caller can hold the noise stream fixed across
    competing allocation strategies - the same luck for everybody.
    """
    cpc, cvr, alpha = latent_row["cpc"], latent_row["cvr"], latent_row["alpha"]
    s_ref = weekly_budget / len(CHANNELS)

    if spend <= 0:
        return {"spend": 0.0, "impressions": 0, "clicks": 0,
                "conversions": 0, "revenue": 0.0}

    clicks = spend / cpc
    clicks_ref = s_ref / cpc
    mean_conversions = cvr * (clicks_ref ** (1.0 - alpha)) * (clicks ** alpha)

    conversions = int(rng.poisson(max(mean_conversions, 0.0)))

    return {
        "spend": round(float(spend), 2),
        "impressions": int(clicks / max(latent_row["ctr"], 1e-9)),
        "clicks": int(clicks),
        "conversions": conversions,
        "revenue": round(float(conversions * latent_row["aov"]), 2),
    }


def expected_conversions(latent_row, spend: float,
                         weekly_budget: float = WEEKLY_BUDGET) -> float:
    """Noise-free expected conversions. Used by the oracle upper bound only."""
    if spend <= 0:
        return 0.0
    cpc, cvr, alpha = latent_row["cpc"], latent_row["cvr"], latent_row["alpha"]
    clicks = spend / cpc
    clicks_ref = (weekly_budget / len(CHANNELS)) / cpc
    return float(cvr * (clicks_ref ** (1.0 - alpha)) * (clicks ** alpha))


def run_allocation(latent: pd.DataFrame, allocator, seed: int = SEED,
                   weekly_budget: float = WEEKLY_BUDGET) -> pd.DataFrame:
    """
    Replay the whole 26 weeks under one allocation strategy.

    `allocator` is any callable (week, history_df) -> {channel: share} with
    shares summing to 1. Every strategy - manual baseline, Thompson Sampling,
    oracle - goes through this one function against the same latent world and
    the same noise stream, so the KPI comparison is apples to apples.
    """
    rng = np.random.default_rng(seed + 1)
    history: list[dict] = []

    for week in sorted(latent["week"].unique()):
        week_latent = latent[latent["week"] == week].set_index("channel")
        shares = allocator(int(week), pd.DataFrame(history))

        total = sum(shares.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"week {week}: shares sum to {total:.6f}, not 1.0")

        # Allocators round their shares for reporting, which can leave the
        # week a rupee or two short of the budget. Renormalise before spending
        # so the books close exactly: "the total budget must be fully
        # allocated" is one of the rules the guardrail enforces.
        shares = {c: shares[c] / total for c in CHANNELS}

        for channel in CHANNELS:
            row = week_latent.loc[channel]
            result = observe(row, shares[channel] * weekly_budget, rng, weekly_budget)
            history.append({
                "week": int(week),
                "channel": channel,
                "share": round(float(shares[channel]), 6),
                **result,
                "festive": bool(row["festive"]),
                "feed_missing": bool(row["feed_missing"]),
                "shock": row["shock"],
            })

    out = pd.DataFrame(history)
    out["cpc_observed"] = (out["spend"] / out["clicks"].replace(0, np.nan)).round(4)
    out["cvr_observed"] = (out["conversions"] / out["clicks"].replace(0, np.nan)).round(6)
    out["conv_per_rupee"] = (out["conversions"] / out["spend"].replace(0, np.nan)).round(8)
    return out
