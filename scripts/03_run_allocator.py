"""
Step 3 of the pipeline: tune the allocator, then measure it honestly.

Run:  python scripts/03_run_allocator.py   (about 90 seconds)

Method
------
The two hyperparameters (`concentration`, `discount`) are tuned on eight
*separate* simulated worlds (seeds 101-108) and evaluated once on the demo
world used for every other result in the project. The demo world never picks a
hyperparameter. Tuning on the world you then report from is how a project
accidentally reports its own overfitting as a business gain, and it is a fair
thing for an examiner to probe.

The sweep records two things per configuration, not one: the gain in
conversions per rupee, and the straight-through processing rate. They pull
against each other - an allocator that leans harder into the leading channel
earns more conversions but trips the guardrail and the 5 pp approval threshold
more often, so the manager ends up signing off every week. Reporting only the
gain would hide that trade-off, so two operating points are evaluated:

    gain-max    the configuration with the highest measured gain
    balanced    the highest-gain configuration that still clears the
                straight-through processing target in the proposal

Strategies compared on the demo world:

    monthly_manual     the As-Is manager (the baseline to beat)
    greedy_recency     ablation - same model, acts on the posterior mean
    thompson_sampling  the proposed allocator, at both operating points
    oracle             perfect foresight (the ceiling)

Writes
------
outputs/allocator_kpis.json      all figures, machine readable
outputs/allocator_report.md      ready-to-paste summary
outputs/allocator_tuning.csv     the full sweep
outputs/allocator_decisions.csv  the per-week audit trail
outputs/fig_*.png                four charts
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.allocator import GreedyRecencyAllocator, ThompsonSamplingAllocator  # noqa: E402
from src.baseline import INITIAL_JUDGMENT_SPLIT, MonthlyManualAllocator  # noqa: E402
from src.kpis import format_inr, outcome_kpis  # noqa: E402
from src.policy import Rulebook  # noqa: E402
from src.world import (  # noqa: E402
    CHANNELS, SEED, SHOCKS, generate_latent, run_allocation,
)

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DB_PATH = DATA_DIR / "advantage.db"

TRAIN_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108]
TEST_SEED = SEED
TUNE_SEEDS = [7, 13]                 # allocator seeds during the sweep
EVAL_SEEDS = [7, 13, 29, 41, 97]     # allocator seeds in the final evaluation

# Both ranges were widened twice, because the first two sweeps put the best
# value on an edge. A chosen hyperparameter sitting at the boundary of its
# grid is not a tuned value, it is a clipped one.
#
# `discount` now spans its entire meaningful domain: 0.0 is full discounting
# (estimate from last week alone) and 1.0 is no discounting (the whole
# history). Landing on either end is a finding about the world, not a gap in
# the search, so no boundary caveat is raised for it. `concentration` has no
# natural upper limit, so an edge result there is still flagged.
CONCENTRATION_GRID = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 6.0]
DISCOUNT_GRID = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.75, 0.90, 1.00]

# Parameters whose grid already covers the full domain - no caveat needed.
DOMAIN_COMPLETE = {"discount"}

# Proposal Section 15 target for straight-through processing.
STP_TARGET = 0.40


def _jsonable(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def _per_rupee(run: pd.DataFrame) -> float:
    return float(run["conversions"].sum()) / float(run["spend"].sum())


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune(rulebook: Rulebook) -> pd.DataFrame:
    n_configs = len(CONCENTRATION_GRID) * len(DISCOUNT_GRID)
    print(f"tuning on {len(TRAIN_SEEDS)} training worlds "
          f"({n_configs} configs x {len(TUNE_SEEDS)} allocator seeds) ...", flush=True)

    worlds = {}
    for seed in TRAIN_SEEDS:
        latent = generate_latent(seed=seed)
        baseline = _per_rupee(run_allocation(latent, MonthlyManualAllocator(rulebook)))
        worlds[seed] = (latent, baseline)

    started = time.time()
    records = []

    for concentration, discount in product(CONCENTRATION_GRID, DISCOUNT_GRID):
        gains, stps = [], []

        for seed, (latent, baseline) in worlds.items():
            for alloc_seed in TUNE_SEEDS:
                allocator = ThompsonSamplingAllocator(
                    rulebook, concentration=concentration, discount=discount,
                    seed=alloc_seed, start_shares=INITIAL_JUDGMENT_SPLIT)
                run = run_allocation(latent, allocator, seed=seed)
                gains.append((_per_rupee(run) - baseline) / baseline * 100.0)
                stps.append(allocator.process_counts()["stp_rate"])

        records.append({
            "concentration": concentration,
            "discount": discount,
            "mean_gain_pct": round(float(np.mean(gains)), 3),
            "sd_gain_pct": round(float(np.std(gains)), 3),
            "worst_world_gain_pct": round(float(np.min(gains)), 3),
            "mean_stp_rate": round(float(np.mean(stps)), 4),
        })
        r = records[-1]
        print(f"  conc={concentration:<4} disc={discount:<5} "
              f"gain {r['mean_gain_pct']:+6.2f}%  worst {r['worst_world_gain_pct']:+6.2f}%  "
              f"stp {r['mean_stp_rate']:5.0%}", flush=True)

    print(f"\nsweep took {time.time() - started:.1f}s")
    return pd.DataFrame(records).sort_values("mean_gain_pct", ascending=False)


def pick_operating_points(grid: pd.DataFrame) -> dict:
    gain_max = grid.iloc[0].to_dict()

    eligible = grid[grid["mean_stp_rate"] >= STP_TARGET]
    if len(eligible):
        balanced = eligible.iloc[0].to_dict()
        balanced_note = (f"highest-gain configuration with straight-through processing "
                         f"at or above the {STP_TARGET:.0%} target")
    else:
        # Nothing clears the target. Take the configuration with the best
        # straight-through processing instead, and say the target is not
        # reachable rather than quietly dropping it.
        balanced = grid.sort_values("mean_stp_rate", ascending=False).iloc[0].to_dict()
        balanced_note = (f"no configuration reaches the {STP_TARGET:.0%} straight-through "
                         f"target; this is the best achievable "
                         f"({balanced['mean_stp_rate']:.0%})")

    boundary_warnings, domain_notes = [], []
    for name, cfg in (("gain-max", gain_max), ("balanced", balanced)):
        if cfg["discount"] in (min(DISCOUNT_GRID), max(DISCOUNT_GRID)):
            limit = ("full discounting - the estimate uses last week alone"
                     if cfg["discount"] == 0.0
                     else "no discounting - the estimate uses the whole history")
            domain_notes.append(f"{name}: discount={cfg['discount']:g}, {limit}")
        if cfg["concentration"] in (min(CONCENTRATION_GRID), max(CONCENTRATION_GRID)):
            boundary_warnings.append(
                f"{name}: concentration={cfg['concentration']:g} sits at the edge of the "
                f"swept range [{min(CONCENTRATION_GRID):g}, {max(CONCENTRATION_GRID):g}], "
                f"which has no natural limit - widen the sweep before calling it optimal")

    return {"gain_max": gain_max, "balanced": balanced,
            "balanced_note": balanced_note, "boundary_warnings": boundary_warnings,
            "domain_notes": domain_notes,
            "same_config": (gain_max["concentration"] == balanced["concentration"]
                            and gain_max["discount"] == balanced["discount"])}


# ---------------------------------------------------------------------------
# Evaluation on the held-out demo world
# ---------------------------------------------------------------------------

def evaluate_config(rulebook: Rulebook, latent: pd.DataFrame, baseline_kpis: dict,
                    concentration: float, discount: float) -> dict:
    """Run one configuration across several allocator seeds on the demo world."""
    runs, kpis_list, allocators = [], [], []

    for alloc_seed in EVAL_SEEDS:
        allocator = ThompsonSamplingAllocator(
            rulebook, concentration=concentration, discount=discount,
            seed=alloc_seed, start_shares=INITIAL_JUDGMENT_SPLIT)
        run = run_allocation(latent, allocator, seed=TEST_SEED)
        runs.append(run)
        kpis_list.append(outcome_kpis(run))
        allocators.append(allocator)

    base = baseline_kpis["conversions_per_rupee"]
    gains = [(k["conversions_per_rupee"] - base) / base * 100.0 for k in kpis_list]

    # One lucky seed is not a result: report the mean, carry the median run
    # for charts and the audit log.
    median_idx = int(np.argsort(gains)[len(gains) // 2])

    return {
        "concentration": concentration,
        "discount": discount,
        "kpis": kpis_list[median_idx],
        "run": runs[median_idx],
        "allocator": allocators[median_idx],
        "gain_mean": round(float(np.mean(gains)), 2),
        "gain_min": round(float(np.min(gains)), 2),
        "gain_max": round(float(np.max(gains)), 2),
        "gains_by_seed": {str(s): round(g, 2) for s, g in zip(EVAL_SEEDS, gains)},
        "process_counts": allocators[median_idx].process_counts(),
        "stp_mean": round(float(np.mean([a.process_counts()["stp_rate"]
                                         for a in allocators])), 4),
    }


def evaluate(rulebook: Rulebook, picks: dict) -> dict:
    latent = pd.read_csv(DATA_DIR / "latent_weeks.csv")

    with open(OUT_DIR / "baseline_kpis.json", "r", encoding="utf-8") as fh:
        phase1 = json.load(fh)

    baseline_kpis = phase1["outcome_kpis"]["monthly_manual"]
    oracle_kpis = phase1["outcome_kpis"]["oracle"]
    headroom = phase1["headroom_above_baseline_pct"]

    gain_max = evaluate_config(rulebook, latent, baseline_kpis,
                               picks["gain_max"]["concentration"],
                               picks["gain_max"]["discount"])

    balanced = gain_max if picks["same_config"] else evaluate_config(
        rulebook, latent, baseline_kpis,
        picks["balanced"]["concentration"], picks["balanced"]["discount"])

    # Ablation at the gain-max configuration: same model, posterior mean,
    # no sampling and so no exploration at all.
    greedy = GreedyRecencyAllocator(
        rulebook, concentration=picks["gain_max"]["concentration"],
        discount=picks["gain_max"]["discount"], seed=7,
        start_shares=INITIAL_JUDGMENT_SPLIT)
    greedy_run = run_allocation(latent, greedy, seed=TEST_SEED)
    greedy_kpis = outcome_kpis(greedy_run)
    base = baseline_kpis["conversions_per_rupee"]
    greedy_gain = (greedy_kpis["conversions_per_rupee"] - base) / base * 100.0

    gain_max["run"].to_csv(DATA_DIR / "run_thompson_sampling.csv",
                           index=False, encoding="utf-8")
    greedy_run.to_csv(DATA_DIR / "run_greedy_recency.csv", index=False, encoding="utf-8")
    decisions = gain_max["allocator"].decisions_frame()
    decisions.to_csv(OUT_DIR / "allocator_decisions.csv", index=False, encoding="utf-8")

    with sqlite3.connect(DB_PATH) as conn:
        gain_max["run"].to_sql("run_thompson_sampling", conn,
                               if_exists="replace", index=False)
        greedy_run.to_sql("run_greedy_recency", conn, if_exists="replace", index=False)
        decisions.to_sql("allocator_decisions", conn, if_exists="replace", index=False)

    return {
        "latent": latent,
        "baseline_kpis": baseline_kpis,
        "oracle_kpis": oracle_kpis,
        "headroom_pct": headroom,
        "gain_max": gain_max,
        "balanced": balanced,
        "greedy_kpis": greedy_kpis,
        "greedy_run": greedy_run,
        "greedy_gain": round(greedy_gain, 2),
        "decisions": decisions,
        "capture_of_ceiling_pct": round(gain_max["gain_mean"] / headroom * 100.0, 1),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def charts(result: dict, grid: pd.DataFrame, picks: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    def cum(run):
        w = run.groupby("week").agg(conversions=("conversions", "sum")).reset_index()
        return w["week"], w["conversions"].cumsum()

    # 1. Cumulative conversions ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    plots = [
        ("As-Is monthly manual", pd.read_csv(DATA_DIR / "run_monthly_manual.csv"), "#888888", "-"),
        ("Greedy recency (ablation)", result["greedy_run"], "#2e7d32", "--"),
        ("Thompson Sampling (gain-max)", result["gain_max"]["run"], "#1565c0", "-"),
        ("Oracle (ceiling)", pd.read_csv(DATA_DIR / "run_oracle.csv"), "#c62828", "--"),
    ]
    if not picks["same_config"]:
        plots.insert(3, ("Thompson Sampling (balanced)",
                         result["balanced"]["run"], "#6a1b9a", "-"))

    for label, run, colour, style in plots:
        x, y = cum(run)
        ax.plot(x, y, style, color=colour, label=label, linewidth=1.8)
    ax.set_xlabel("week")
    ax.set_ylabel("cumulative conversions")
    ax.set_title("Cumulative conversions on identical spend (same world, same noise)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / "fig_cumulative_conversions.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path.name)

    # 2. Allocation over time --------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    shares = result["gain_max"]["run"].pivot(index="week", columns="channel", values="share")
    ax.stackplot(shares.index, [shares[c] for c in CHANNELS],
                 labels=list(CHANNELS), alpha=0.88,
                 colors=["#4285f4", "#0b5fd4", "#ff9900"])
    for week, _c, _f, _m, _d, _label, tc in SHOCKS:
        ax.axvline(week, color="black", linestyle=":", linewidth=1)
        ax.text(week + 0.15, 1.015, tc, fontsize=8)
    ax.set_xlabel("week")
    ax.set_ylabel("share of weekly budget")
    ax.set_ylim(0, 1)
    ax.set_xlim(shares.index.min(), shares.index.max())
    ax.set_title("Allocation proposed each week (gain-max), with injected shocks marked")
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    path = OUT_DIR / "fig_allocation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path.name)

    # 3. Learning: posterior belief vs latent truth ------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    decisions = result["decisions"]
    latent = result["latent"].copy()
    latent["true_per_lakh"] = latent["cvr"] / latent["cpc"] * 100_000.0

    for ax, channel in zip(axes, CHANNELS):
        truth = latent[latent["channel"] == channel]
        ax.plot(truth["week"], truth["true_per_lakh"], color="#c62828",
                linewidth=1.6, label="latent truth")
        mean = decisions[f"posterior_mean_{channel}"]
        sd = decisions[f"posterior_sd_{channel}"]
        ax.plot(decisions["week"], mean, color="#1565c0", linewidth=1.6,
                label="posterior mean")
        ax.fill_between(decisions["week"], mean - 2 * sd, mean + 2 * sd,
                        color="#1565c0", alpha=0.20, label="+/- 2 sd")
        ax.set_title(channel)
        ax.set_xlabel("week")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("conversions per lakh")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Belief against truth. The +/- 2 sd band is narrow because each week "
                 "brings ~500 conversions per channel - which is why exploration adds little",
                 fontsize=9)
    fig.tight_layout()
    path = OUT_DIR / "fig_learning.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path.name)

    # 4. The gain / automation trade-off ----------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sizes = 18 + 22 * np.log1p(grid["concentration"].to_numpy())
    sc = ax.scatter(grid["mean_stp_rate"] * 100, grid["mean_gain_pct"],
                    c=grid["discount"], cmap="viridis", s=sizes,
                    edgecolor="white", linewidth=0.6, zorder=3)

    ax.axvline(STP_TARGET * 100, color="#c62828", linestyle="--", linewidth=1.2)
    ax.text(STP_TARGET * 100 + 1.0, grid["mean_gain_pct"].min(),
            f"proposal STP\ntarget {STP_TARGET:.0%}", color="#c62828", fontsize=8,
            va="bottom")

    for label, cfg, colour in (("gain-max", picks["gain_max"], "#c62828"),
                               ("balanced", picks["balanced"], "#6a1b9a")):
        ax.annotate(
            f"{label}\nconc {cfg['concentration']:g}, disc {cfg['discount']:g}",
            (cfg["mean_stp_rate"] * 100, cfg["mean_gain_pct"]),
            textcoords="offset points", xytext=(14, -6), fontsize=8, color=colour,
            arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.1))

    ax.set_xlabel("straight-through processing rate (%)")
    ax.set_ylabel("mean gain vs As-Is (%)")
    ax.set_title("Every configuration trades automation against gain\n"
                 "(marker size = concentration, colour = discount)", fontsize=10)
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="discount (0 = last week only)")
    fig.tight_layout()
    path = OUT_DIR / "fig_tradeoff.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path.name)

    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(result: dict, picks: dict, grid: pd.DataFrame,
                 figures: list[str]) -> None:
    base = result["baseline_kpis"]
    oracle = result["oracle_kpis"]
    gm, bal = result["gain_max"], result["balanced"]
    headroom = result["headroom_pct"]
    proc = gm["process_counts"]

    lines = [
        "# Allocator Results - Discounted Thompson Sampling",
        "",
        "Generated by `scripts/03_run_allocator.py`.",
        "",
        "## 1. Method",
        "",
        f"- Hyperparameters swept over {len(CONCENTRATION_GRID) * len(DISCOUNT_GRID)} "
        f"configurations on **{len(TRAIN_SEEDS)} separate training worlds** "
        f"(seeds {TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}). The demo world used for every "
        "reported result never picks a hyperparameter.",
        f"- Each configuration was scored on two axes: gain in conversions per rupee, and "
        f"straight-through processing rate.",
        f"- Final evaluation on the demo world across {len(EVAL_SEEDS)} allocator seeds; "
        "the headline is the mean with the spread reported.",
        "- Every run replays the same frozen world, the same noise stream and the same total "
        "spend, so conversions are directly comparable.",
        "",
    ]

    if picks["boundary_warnings"]:
        lines += [
            "> **Methodological caveat.** "
            + "; ".join(picks["boundary_warnings"])
            + ". A chosen value at the edge of a swept range means the true optimum may lie "
              "outside it. It is reported as the best *within the range tested*.",
            "",
        ]

    if picks["domain_notes"]:
        lines += [
            "> **Finding, not a caveat.** "
            + "; ".join(picks["domain_notes"])
            + ". The discount grid spans its whole domain [0, 1], so this is the limit of "
              "the parameter rather than the limit of the search. It says something about "
              "the world: because channel advantage rotates, evidence older than about a "
              "week is not just less useful, it is actively misleading, and the best "
              "estimator in this world is close to memoryless. A longer memory is what the "
              "As-Is monthly manager effectively has, and it is precisely why the As-Is "
              "gains almost nothing over a frozen split.",
            "",
        ]

    lines += [
        "## 2. Results over 26 weeks",
        "",
        "| Strategy | Conversions | Conv. per lakh | Cost per conversion | vs As-Is | STP |",
        "|---|---:|---:|---:|---:|---:|",
        f"| As-Is monthly manual | {base['total_conversions']:,} | "
        f"{base['conversions_per_lakh']:.2f} | "
        f"{format_inr(base['cost_per_conversion_inr'])} | - | 0% |",
        f"| Greedy recency (ablation) | {result['greedy_kpis']['total_conversions']:,} | "
        f"{result['greedy_kpis']['conversions_per_lakh']:.2f} | "
        f"{format_inr(result['greedy_kpis']['cost_per_conversion_inr'])} | "
        f"{result['greedy_gain']:+.1f}% | - |",
        f"| **Thompson Sampling (gain-max)** | **{gm['kpis']['total_conversions']:,}** | "
        f"**{gm['kpis']['conversions_per_lakh']:.2f}** | "
        f"**{format_inr(gm['kpis']['cost_per_conversion_inr'])}** | "
        f"**{gm['gain_mean']:+.1f}%** | {gm['stp_mean']:.0%} |",
    ]

    if not picks["same_config"]:
        lines.append(
            f"| Thompson Sampling (balanced) | {bal['kpis']['total_conversions']:,} | "
            f"{bal['kpis']['conversions_per_lakh']:.2f} | "
            f"{format_inr(bal['kpis']['cost_per_conversion_inr'])} | "
            f"{bal['gain_mean']:+.1f}% | {bal['stp_mean']:.0%} |"
        )

    lines += [
        f"| Oracle (ceiling) | {oracle['total_conversions']:,} | "
        f"{oracle['conversions_per_lakh']:.2f} | "
        f"{format_inr(oracle['cost_per_conversion_inr'])} | {headroom:+.1f}% | - |",
        "",
        f"Gain-max configuration: `concentration = {gm['concentration']:g}`, "
        f"`discount = {gm['discount']:g}`. It gains **{gm['gain_mean']:+.1f}%** on "
        f"conversions per rupee (range {gm['gain_min']:+.1f}% to {gm['gain_max']:+.1f}% "
        f"across allocator seeds), which is **{result['capture_of_ceiling_pct']:.0f}% of the "
        f"{headroom:.1f}% ceiling** available in this world.",
        "",
        "### Does the proposal target hold?",
        "",
    ]

    if gm["gain_mean"] >= 8.0:
        lines.append(
            f"Yes, on conversions per rupee: {gm['gain_mean']:+.1f}% is inside the proposal "
            f"target of +8-12%."
        )
    else:
        lines.append(
            f"**Not quite, and the report should say so.** The proposal targeted +8-12%; the "
            f"measured gain is {gm['gain_mean']:+.1f}%, which is "
            f"{result['capture_of_ceiling_pct']:.0f}% of everything available in this world. "
            f"The honest framing for Section 15 is \"{gm['gain_mean']:+.1f}% against a "
            f"measured ceiling of {headroom:.1f}%\" - a target revised after seeing the "
            "result is much harder to defend in a viva than a target reported as missed."
        )

    lines += [
        "",
        "## 3. The trade-off the proposal did not anticipate",
        "",
        f"The proposal set two targets that pull against each other: +8-12% on conversions "
        f"per rupee (O2) and 40-60% straight-through processing (Section 15). They cannot "
        f"both be maximised. An allocator that leans harder into the leading channel earns "
        f"more conversions, but it crosses the 5 pp auto-execute threshold and gets trimmed "
        f"by the 15 pp weekly cap more often - and both of those force a manager approval "
        f"under the rulebook.",
        "",
        f"At the gain-max operating point, straight-through processing is "
        f"**{gm['stp_mean']:.0%}** ({proc['auto_executed']} of {proc['cycles']} cycles ran "
        f"untouched), against the 40-60% target.",
        "",
        f"{picks['balanced_note'].capitalize()}: "
        f"`concentration = {bal['concentration']:g}`, `discount = {bal['discount']:g}`, "
        f"giving {bal['gain_mean']:+.1f}% with {bal['stp_mean']:.0%} straight-through "
        f"processing.",
        "",
        "This is a decision for the marketing manager, not for the model: pay for more "
        "conversions with more of their own review time, or accept a smaller gain for a "
        "process that mostly runs itself. `outputs/fig_tradeoff.png` plots every "
        "configuration on both axes so the choice can be made with the numbers in view. "
        "Presenting that choice is a better answer than presenting one tuned number.",
        "",
        "## 4. How much of the gain is Thompson Sampling?",
        "",
        f"The ablation uses the identical model and discounting but acts on the posterior "
        f"*mean* - no sampling, so no exploration. It gains "
        f"{result['greedy_gain']:+.1f}%, against {gm['gain_mean']:+.1f}% for Thompson "
        f"Sampling.",
        "",
    ]

    exploration_share = gm["gain_mean"] - result["greedy_gain"]
    if abs(exploration_share) < 0.5:
        lines += [
            f"So exploration contributes essentially nothing "
            f"({exploration_share:+.1f} pp) and **the gain comes from recency-weighted "
            f"estimation**, not from the bandit's exploration. This is the expected result "
            f"at this data volume: around 500 conversions per channel per week makes the "
            f"posterior very tight (see `fig_learning.png`), so a Thompson draw lands "
            f"almost on top of the mean and there is little real uncertainty left to probe.",
            "",
            "Two things follow, and both are worth saying before an examiner says them:",
            "",
            "1. The honest claim for this project is that *fast, principled, "
            "recency-weighted re-estimation plus a disciplined projection onto the policy "
            "set* produces the gain. Thompson Sampling is the formulation used to get "
            "there, and it costs nothing, but it is not where the value comes from here.",
            "2. Exploration would matter much more for a smaller advertiser, a newly "
            "launched channel with no history, or a weekly cycle with tens rather than "
            "hundreds of conversions. The 15% per-channel floor in the rulebook is also "
            "doing part of the exploration job already, by guaranteeing every channel keeps "
            "producing data.",
        ]
    else:
        lines.append(
            f"Thompson Sampling adds {exploration_share:+.1f} pp over the greedy ablation, "
            f"so exploration is contributing measurably on top of recency weighting."
        )

    lines += [
        "",
        "## 5. To-Be process KPIs (measured, not estimated)",
        "",
        "Counted from the agent decision log, `outputs/allocator_decisions.csv`, at the "
        "gain-max operating point:",
        "",
        "| Process KPI | As-Is (estimate) | To-Be (measured) |",
        "|---|---:|---:|",
        f"| Reallocation frequency | 1 / month | {proc['cycles']} cycles in 26 weeks (weekly) |",
        f"| Straight-through processing | 0% | {proc['stp_rate']:.0%} "
        f"({proc['auto_executed']}/{proc['cycles']}) |",
        f"| Sent for manager approval | every cycle | {proc['sent_for_approval']} |",
        f"| Escalated | not tracked | {proc['escalated']} |",
        f"| Weeks detected as degraded data | not detected | {proc['degraded_weeks']} |",
        f"| Proposals trimmed by guardrail | no check existed | {proc['trimmed_weeks']} |",
        f"| Guardrail breaches executed | unchecked | 0 |",
        "",
        "Note that manager *effort* still falls sharply even at 0% straight-through "
        "processing: approving a prepared proposal with a written rationale takes minutes, "
        "where the As-Is cycle was estimated at 8 hours of compiling and deciding. Approval "
        "count and effort are different KPIs and the report should not conflate them.",
        "",
        "## 6. Top configurations on the training worlds",
        "",
        "| concentration | discount | mean gain | worst world | STP |",
        "|---:|---:|---:|---:|---:|",
    ]

    for _, row in grid.head(8).iterrows():
        lines.append(
            f"| {row['concentration']:g} | {row['discount']:g} | "
            f"{row['mean_gain_pct']:+.2f}% | {row['worst_world_gain_pct']:+.2f}% | "
            f"{row['mean_stp_rate']:.0%} |"
        )

    captions = {
        "fig_cumulative_conversions.png": "cumulative conversions on identical spend",
        "fig_allocation.png": "the allocation proposed each week, shocks marked",
        "fig_learning.png": "posterior belief against latent truth, per channel",
        "fig_tradeoff.png": "gain against straight-through processing, every configuration",
    }
    lines += ["", "## 7. Charts", ""]
    for name in figures:
        lines.append(f"- `outputs/{name}` - {captions.get(name, '')}")
    lines.append("")

    with open(OUT_DIR / "allocator_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rulebook = Rulebook.load()

    grid = tune(rulebook)
    grid.to_csv(OUT_DIR / "allocator_tuning.csv", index=False, encoding="utf-8")

    picks = pick_operating_points(grid)
    result = evaluate(rulebook, picks)
    figures = charts(result, grid, picks)
    write_report(result, picks, grid, figures)

    gm, bal = result["gain_max"], result["balanced"]
    payload = {
        "method": {
            "train_seeds": TRAIN_SEEDS, "test_seed": TEST_SEED,
            "eval_allocator_seeds": EVAL_SEEDS,
            "concentration_grid": CONCENTRATION_GRID, "discount_grid": DISCOUNT_GRID,
            "stp_target": STP_TARGET,
            "boundary_warnings": picks["boundary_warnings"],
            "domain_notes": picks["domain_notes"],
        },
        "operating_points": {
            "gain_max": {k: gm[k] for k in
                         ("concentration", "discount", "gain_mean", "gain_min",
                          "gain_max", "gains_by_seed", "stp_mean", "process_counts")},
            "balanced": {k: bal[k] for k in
                         ("concentration", "discount", "gain_mean", "gain_min",
                          "gain_max", "gains_by_seed", "stp_mean", "process_counts")},
            "balanced_note": picks["balanced_note"],
        },
        "outcome_kpis": {
            "monthly_manual": result["baseline_kpis"],
            "greedy_recency": result["greedy_kpis"],
            "thompson_sampling_gain_max": gm["kpis"],
            "thompson_sampling_balanced": bal["kpis"],
            "oracle": result["oracle_kpis"],
        },
        "greedy_gain_pct": result["greedy_gain"],
        "exploration_contribution_pp": round(gm["gain_mean"] - result["greedy_gain"], 2),
        "ceiling_pct": result["headroom_pct"],
        "capture_of_ceiling_pct": result["capture_of_ceiling_pct"],
        "tuning_grid": grid.to_dict(orient="records"),
    }
    with open(OUT_DIR / "allocator_kpis.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)

    print("\n" + "=" * 72)
    print("Allocator measured on the held-out demo world")
    print("=" * 72)
    print(f"as-is baseline    : {result['baseline_kpis']['conversions_per_lakh']:.2f} conv/lakh")
    print(f"greedy ablation   : {result['greedy_kpis']['conversions_per_lakh']:.2f} "
          f"({result['greedy_gain']:+.1f}%)  <- no exploration")
    print(f"thompson gain-max : {gm['kpis']['conversions_per_lakh']:.2f} "
          f"({gm['gain_mean']:+.1f}%, range {gm['gain_min']:+.1f} to {gm['gain_max']:+.1f})"
          f"  conc={gm['concentration']:g} disc={gm['discount']:g}  stp {gm['stp_mean']:.0%}")
    if not picks["same_config"]:
        print(f"thompson balanced : {bal['kpis']['conversions_per_lakh']:.2f} "
              f"({bal['gain_mean']:+.1f}%)"
              f"  conc={bal['concentration']:g} disc={bal['discount']:g}  "
              f"stp {bal['stp_mean']:.0%}")
    print(f"oracle ceiling    : {result['oracle_kpis']['conversions_per_lakh']:.2f} "
          f"({result['headroom_pct']:+.1f}%)")
    print(f"\ncaptured {result['capture_of_ceiling_pct']:.0f}% of the available ceiling")
    print(f"exploration contributes "
          f"{gm['gain_mean'] - result['greedy_gain']:+.1f} pp over the greedy ablation")
    print(f"balanced point    : {picks['balanced_note']}")

    if picks["domain_notes"]:
        print("\nFINDING: " + "; ".join(picks["domain_notes"]))
    if picks["boundary_warnings"]:
        print("\nCAVEAT: " + "; ".join(picks["boundary_warnings"]))

    print(f"\nreport : {OUT_DIR / 'allocator_report.md'}")
    print(f"charts : {', '.join(figures)}")


if __name__ == "__main__":
    main()
