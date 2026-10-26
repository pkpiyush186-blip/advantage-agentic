"""
KPI measurement - the single place where "did it work?" is answered.

The proposal splits KPIs into two kinds and so does this module:

*   Outcome KPIs are measured from a run. Conversions per rupee, revenue,
    ROAS. These come straight out of the simulation and are directly
    comparable between the manual baseline and the agentic run, because both
    are replayed against the same frozen world.

*   Process KPIs describe how the process itself behaves - cycle time,
    manager hours, human touchpoints, straight-through processing. For the
    As-Is these are the estimates stated in the proposal and are labelled
    `estimated=True`. For the To-Be they will be counted from the agent case
    log, and labelled `estimated=False`. Keeping the label in the data stops
    an estimate from quietly being presented as a measurement in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


# --------------------------------------------------------------------------
# Outcome KPIs (measured from a run)
# --------------------------------------------------------------------------

def outcome_kpis(run: pd.DataFrame) -> dict:
    """Headline outcome KPIs for one 26-week run."""
    spend = float(run["spend"].sum())
    conversions = int(run["conversions"].sum())
    revenue = float(run["revenue"].sum())

    return {
        "weeks": int(run["week"].nunique()),
        "total_spend_inr": round(spend, 2),
        "total_conversions": conversions,
        "total_revenue_inr": round(revenue, 2),
        "conversions_per_rupee": round(conversions / spend, 8) if spend else 0.0,
        "cost_per_conversion_inr": round(spend / conversions, 2) if conversions else None,
        "roas": round(revenue / spend, 4) if spend else 0.0,
        "conversions_per_lakh": round(conversions / (spend / 100000.0), 2) if spend else 0.0,
    }


def weekly_series(run: pd.DataFrame) -> pd.DataFrame:
    """Per-week totals - what the dashboard charts and the report plots."""
    weekly = run.groupby("week").agg(
        spend=("spend", "sum"),
        conversions=("conversions", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    weekly["conv_per_rupee"] = (weekly["conversions"] / weekly["spend"]).round(8)
    weekly["cumulative_conversions"] = weekly["conversions"].cumsum()
    return weekly


def share_series(run: pd.DataFrame) -> pd.DataFrame:
    """Week x channel allocation shares - the allocation chart."""
    return run.pivot(index="week", columns="channel", values="share").round(4)


# --------------------------------------------------------------------------
# Process KPIs (declared for As-Is, counted for To-Be)
# --------------------------------------------------------------------------

@dataclass
class ProcessKpis:
    """How the process runs, independent of how well the ads perform."""

    label: str
    reallocation_frequency_per_month: float
    mean_detection_lag_days: float
    manager_hours_per_cycle: float
    human_touchpoints_per_cycle: float
    stp_rate: float                     # share of cycles completed with no human touch
    escalation_rate: float
    guardrail_breaches_executed: int
    cycles_in_horizon: int
    blended_hourly_rate_inr: float = 500.0
    estimated: bool = True
    note: str = ""

    @property
    def cost_per_cycle_inr(self) -> float:
        return round(self.manager_hours_per_cycle * self.blended_hourly_rate_inr, 2)

    @property
    def total_process_cost_inr(self) -> float:
        return round(self.cost_per_cycle_inr * self.cycles_in_horizon, 2)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["cost_per_cycle_inr"] = self.cost_per_cycle_inr
        data["total_process_cost_inr"] = self.total_process_cost_inr
        return data


# As-Is process figures. Every number here is an estimate for the fictional
# case, exactly as stated in Section 6 of the proposal - not a measurement.
AS_IS_PROCESS = ProcessKpis(
    label="As-Is (manual, monthly reallocation)",
    reallocation_frequency_per_month=1.0,
    mean_detection_lag_days=15.0,
    manager_hours_per_cycle=8.0,
    human_touchpoints_per_cycle=5.0,
    stp_rate=0.0,
    escalation_rate=0.0,            # not tracked in the As-Is process
    guardrail_breaches_executed=-1,  # -1 = not systematically checked
    cycles_in_horizon=6,             # 26 weeks at one reallocation a month
    estimated=True,
    note="Estimates for the fictional case (proposal Section 6). Escalations are "
         "not tracked and compliance is not systematically checked in the As-Is process.",
)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def compare_outcomes(baseline: dict, variant: dict,
                     baseline_label: str = "baseline",
                     variant_label: str = "variant") -> dict:
    """Percentage deltas on the outcome KPIs that the rubric asks about."""
    out = {"baseline_label": baseline_label, "variant_label": variant_label}

    for key in ("conversions_per_rupee", "total_conversions",
                "total_revenue_inr", "roas", "cost_per_conversion_inr"):
        base_val, var_val = baseline.get(key), variant.get(key)
        if not base_val or var_val is None:
            continue
        out[key] = {
            "baseline": base_val,
            "variant": var_val,
            "delta_pct": round((var_val - base_val) / base_val * 100.0, 2),
        }
    return out


def format_inr(amount: float) -> str:
    """Indian digit grouping, as 'Rs 11,50,000'. ASCII only - the Windows
    console here cannot print the rupee sign."""
    amount = round(float(amount))
    sign = "-" if amount < 0 else ""
    digits = str(abs(amount))

    if len(digits) <= 3:
        return f"{sign}Rs {digits}"

    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return f"{sign}Rs {','.join(groups)},{tail}"
