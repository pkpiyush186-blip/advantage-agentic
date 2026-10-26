"""
Step 1 of the pipeline: generate the frozen 26-week world.

Run:  python scripts/01_generate_data.py

Writes
------
data/latent_weeks.csv   the ground truth, one row per channel-week
data/advantage.db       the same table in SQLite (the project database tool)

Nothing in this script calls an LLM or needs an API key. The dataset is
deterministic for the seed in src/world.py, so re-running it reproduces the
identical world - which is what makes every later result repeatable.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.world import (  # noqa: E402
    BASE, CHANNELS, FESTIVE_WEEKS, N_WEEKS, SEED, SHOCKS, WEEKLY_BUDGET,
    generate_latent,
)
from src.kpis import format_inr  # noqa: E402

DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "latent_weeks.csv"
DB_PATH = DATA_DIR / "advantage.db"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    latent = generate_latent(seed=SEED, n_weeks=N_WEEKS)
    latent.to_csv(CSV_PATH, index=False, encoding="utf-8")

    with sqlite3.connect(DB_PATH) as conn:
        latent.to_sql("latent_weeks", conn, if_exists="replace", index=False)

    print("=" * 72)
    print("AdVantage Retail - simulated world generated")
    print("=" * 72)
    print(f"seed            : {SEED}  (deterministic)")
    print(f"horizon         : {N_WEEKS} weeks x {len(CHANNELS)} channels = {len(latent)} rows")
    print(f"weekly budget   : {format_inr(WEEKLY_BUDGET)}")
    print(f"festive window  : weeks {FESTIVE_WEEKS.start}-{FESTIVE_WEEKS.stop - 1}")
    print(f"csv             : {CSV_PATH}")
    print(f"sqlite          : {DB_PATH}")

    print("\n--- Channel economics at the reference spend -------------------")
    print("Conversions per rupee = cvr / cpc. These are close together on")
    print("purpose: no channel is obviously best, so the split has to be learnt.\n")
    latent = latent.copy()
    latent["per_rupee_ref"] = latent["cvr"] / latent["cpc"]
    summary = latent.groupby("channel").agg(
        base_cpc=("cpc", "mean"),
        base_cvr=("cvr", "mean"),
        per_rupee_mean=("per_rupee_ref", "mean"),
        per_rupee_min=("per_rupee_ref", "min"),
        per_rupee_max=("per_rupee_ref", "max"),
    ).round(6)
    print(summary.to_string())

    print("\n--- Weeks where each channel is strongest ----------------------")
    best = latent.loc[latent.groupby("week")["per_rupee_ref"].idxmax()]
    counts = best["channel"].value_counts().to_dict()
    for channel in CHANNELS:
        weeks = sorted(best[best["channel"] == channel]["week"].tolist())
        print(f"  {channel:<8} best in {counts.get(channel, 0):>2} of {N_WEEKS} weeks  {weeks}")
    print("\n  The leader changes through the horizon. That is the whole case")
    print("  for reallocating weekly instead of once a month.")

    print("\n--- Injected shocks (each one drives a test case) ---------------")
    for week, channel, field, mult, duration, label, tc in SHOCKS:
        effect = "feed missing" if field is None else f"{field} x {mult}"
        print(f"  week {week:>2}  {channel:<8} {effect:<18} {duration}w   {label}  -> {tc}")

    print("\n--- Base parameters (illustrative, see README) -----------------")
    for channel in CHANNELS:
        b = BASE[channel]
        print(f"  {channel:<8} cpc {format_inr(b['cpc']):>8}  ctr {b['ctr']:.3%}  "
              f"cvr {b['cvr']:.2%}  aov {format_inr(b['aov']):>10}  alpha {b['alpha']}")

    print("\nDone. Next: python scripts/02_run_baseline.py")


if __name__ == "__main__":
    main()
