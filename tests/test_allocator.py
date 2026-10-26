"""
Allocator tests.

Two groups. The first checks the learning is actually Bayesian updating and
not something that merely looks like it. The second checks the allocator can
never produce an allocation the guardrail would reject - the property that
matters most in a live process, because the allocator is the only component
that proposes spending money.

Run either way:
    python -m pytest tests -q
    python tests/test_allocator.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.allocator import GreedyRecencyAllocator, LAKH, ThompsonSamplingAllocator  # noqa: E402
from src.baseline import INITIAL_JUDGMENT_SPLIT  # noqa: E402
from src.policy import Rulebook  # noqa: E402
from src.world import CHANNELS, generate_latent, run_allocation  # noqa: E402

TEST_WEEKS = 8
TEST_SEED = 1


def _rulebook() -> Rulebook:
    return Rulebook.load()


def _small_world() -> pd.DataFrame:
    return generate_latent(seed=TEST_SEED, n_weeks=TEST_WEEKS)


def _week_row(channel: str, week: int, conversions: float, spend: float,
              feed_missing: bool = False) -> dict:
    return {"week": week, "channel": channel, "conversions": conversions,
            "spend": spend, "feed_missing": feed_missing}


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def test_posterior_moves_towards_the_observed_rate():
    rb = _rulebook()
    allocator = ThompsonSamplingAllocator(rb, discount=1.0)

    before = allocator.posteriors["google"].mean

    # 400 conversions on 2 lakh of spend = 200 per lakh, above the 150 prior.
    history = pd.DataFrame([_week_row("google", 1, 400, 2 * LAKH),
                            _week_row("meta", 1, 300, 2 * LAKH),
                            _week_row("amazon", 1, 300, 2 * LAKH)])
    allocator._update_from(history, week=2)

    after = allocator.posteriors["google"].mean
    assert before < after < 200.0, f"{before} -> {after}"


def test_posterior_is_exact_bayesian_arithmetic():
    """shape = a0 + conversions, rate = b0 + spend_in_lakh, with no discounting."""
    rb = _rulebook()
    allocator = ThompsonSamplingAllocator(rb, discount=1.0)
    a0, b0 = allocator.a0, allocator.b0

    history = pd.DataFrame([_week_row(c, 1, 500, 3 * LAKH) for c in CHANNELS])
    allocator._update_from(history, week=2)

    p = allocator.posteriors["meta"]
    assert abs(p.shape - (a0 + 500)) < 1e-9
    assert abs(p.rate - (b0 + 3.0)) < 1e-9


def test_full_discounting_remembers_only_the_last_week():
    rb = _rulebook()
    allocator = ThompsonSamplingAllocator(rb, discount=0.0)

    old = pd.DataFrame([_week_row(c, 1, 900, 3 * LAKH) for c in CHANNELS])
    recent = pd.DataFrame([_week_row(c, 2, 300, 3 * LAKH) for c in CHANNELS])
    allocator._update_from(pd.concat([old, recent]), week=3)

    # Only week 2 should remain: 300 conversions on 3 lakh = 100 per lakh,
    # pulled slightly towards the prior by its small weight.
    only_recent = ThompsonSamplingAllocator(rb, discount=0.0)
    only_recent._update_from(recent, week=3)

    for c in CHANNELS:
        assert abs(allocator.posteriors[c].mean - only_recent.posteriors[c].mean) < 1e-9


def test_a_missing_feed_widens_uncertainty_instead_of_freezing_it():
    """E2: no news about a channel must make the agent less sure, not equally sure."""
    rb = _rulebook()
    allocator = ThompsonSamplingAllocator(rb, discount=0.6)

    settled = pd.DataFrame([_week_row(c, w, 500, 3 * LAKH)
                            for w in (1, 2, 3) for c in CHANNELS])
    allocator._update_from(settled, week=4)
    sd_before = {c: allocator.posteriors[c].sd for c in CHANNELS}

    gap = pd.DataFrame([
        _week_row("google", 4, 500, 3 * LAKH),
        _week_row("meta", 4, 500, 3 * LAKH),
        _week_row("amazon", 4, 0, 0, feed_missing=True),
    ])
    missing = allocator._update_from(gap, week=5)

    assert missing == ["amazon"]
    assert allocator.posteriors["amazon"].sd > sd_before["amazon"]
    assert allocator.posteriors["google"].sd <= sd_before["google"] + 1e-9


def test_allocator_never_sees_the_current_week():
    """The decision for week w may only use data from weeks before w."""
    rb = _rulebook()
    allocator = ThompsonSamplingAllocator(rb)

    history = pd.DataFrame([_week_row(c, w, 500, 3 * LAKH)
                            for w in (1, 2, 3) for c in CHANNELS])
    allocator._update_from(history, week=3)

    assert allocator._weeks_consumed == {1, 2}, allocator._weeks_consumed


# ---------------------------------------------------------------------------
# Safety: the allocator can never propose something non-compliant
# ---------------------------------------------------------------------------

def test_every_proposed_week_is_policy_compliant():
    rb = _rulebook()
    latent = _small_world()
    allocator = ThompsonSamplingAllocator(rb, start_shares=INITIAL_JUDGMENT_SPLIT)
    run = run_allocation(latent, allocator, seed=TEST_SEED)

    previous = dict(INITIAL_JUDGMENT_SPLIT)
    total = sum(previous.values())
    previous = {c: v / total for c, v in previous.items()}

    for week in sorted(run["week"].unique()):
        shares = (run[run["week"] == week]
                  .set_index("channel")["share"].to_dict())
        decision = next(d for d in allocator.decisions if d["week"] == week)

        violations = rb.check(shares, previous,
                              festive=decision["festive"],
                              degraded=decision["degraded"])
        assert not violations, f"week {week}: {[str(v) for v in violations]}"
        previous = shares


def test_shares_always_sum_to_one():
    rb = _rulebook()
    run = run_allocation(_small_world(),
                         ThompsonSamplingAllocator(rb), seed=TEST_SEED)
    by_week = run.groupby("week")["share"].sum()
    assert np.allclose(by_week.values, 1.0, atol=1e-6), by_week.to_dict()


def test_extreme_concentration_still_respects_the_floors():
    """Even told to go all-in, the projection keeps every channel above 15%."""
    rb = _rulebook()
    allocator = ThompsonSamplingAllocator(rb, concentration=50.0, discount=0.0)
    run = run_allocation(_small_world(), allocator, seed=TEST_SEED)

    assert run["share"].min() >= 0.15 - 1e-6, run["share"].min()
    assert run["share"].max() <= 0.55 + 1e-6, run["share"].max()


def test_infeasible_rulebook_holds_the_previous_allocation():
    """When no compliant split exists, hold and escalate - never guess."""
    data = copy.deepcopy(_rulebook().data)
    data["hard_limits"]["min_channel_share"] = 0.40  # 3 x 40% = 120%
    rb = Rulebook(data)

    allocator = ThompsonSamplingAllocator(rb, start_shares=INITIAL_JUDGMENT_SPLIT)
    held = dict(allocator.current)
    shares = allocator(1, pd.DataFrame())

    assert all(abs(shares[c] - held[c]) < 1e-9 for c in CHANNELS)
    assert allocator.decisions[-1]["route"] == "escalate"
    assert not allocator.decisions[-1]["feasible"]


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

def test_one_logged_decision_per_cycle():
    rb = _rulebook()
    allocator = ThompsonSamplingAllocator(rb)
    run_allocation(_small_world(), allocator, seed=TEST_SEED)

    assert len(allocator.decisions) == TEST_WEEKS
    assert len(allocator.decisions_frame()) == TEST_WEEKS


def test_process_counts_add_up():
    rb = _rulebook()
    allocator = ThompsonSamplingAllocator(rb)
    run_allocation(_small_world(), allocator, seed=TEST_SEED)

    counts = allocator.process_counts()
    assert counts["cycles"] == TEST_WEEKS
    assert (counts["auto_executed"] + counts["sent_for_approval"]
            + counts["escalated"]) == counts["cycles"]


def test_every_decision_records_a_reason_when_a_human_is_involved():
    """An approval with no stated reason is not an audit trail."""
    rb = _rulebook()
    allocator = ThompsonSamplingAllocator(rb, concentration=6.0)
    run_allocation(_small_world(), allocator, seed=TEST_SEED)

    for d in allocator.decisions:
        if d["route"] in ("approve", "escalate"):
            assert d["reasons"], f"week {d['week']} routed to {d['route']} with no reason"


# ---------------------------------------------------------------------------
# Reproducibility and the ablation
# ---------------------------------------------------------------------------

def test_same_seed_reproduces_the_same_run():
    rb = _rulebook()
    latent = _small_world()
    a = run_allocation(latent, ThompsonSamplingAllocator(rb, seed=5), seed=TEST_SEED)
    b = run_allocation(latent, ThompsonSamplingAllocator(rb, seed=5), seed=TEST_SEED)

    assert a["conversions"].sum() == b["conversions"].sum()
    assert np.allclose(a["share"].values, b["share"].values)


def test_different_seeds_give_different_runs():
    """If the seed does not matter, nothing is actually being sampled."""
    rb = _rulebook()
    latent = _small_world()
    a = run_allocation(latent, ThompsonSamplingAllocator(rb, seed=5), seed=TEST_SEED)
    b = run_allocation(latent, ThompsonSamplingAllocator(rb, seed=6), seed=TEST_SEED)

    assert not np.allclose(a["share"].values, b["share"].values)


def test_greedy_ablation_ignores_its_seed():
    """The ablation acts on the posterior mean, so sampling must not occur."""
    rb = _rulebook()
    latent = _small_world()
    a = run_allocation(latent, GreedyRecencyAllocator(rb, seed=5), seed=TEST_SEED)
    b = run_allocation(latent, GreedyRecencyAllocator(rb, seed=999), seed=TEST_SEED)

    assert np.allclose(a["share"].values, b["share"].values)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} allocator tests passed")
    sys.exit(1 if failed else 0)
