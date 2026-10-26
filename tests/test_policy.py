"""
Guardrail tests.

The guardrail is the one component that must never be wrong, so it is the one
component with tests. Each test maps to a rule in the rulebook or to a test
case in Section 13 of the proposal.

Run either way:
    python -m pytest tests -q
    python tests/test_policy.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.policy import Rulebook, equal_split  # noqa: E402

CHANNELS = ("google", "meta", "amazon")


def _rulebook() -> Rulebook:
    return Rulebook.load()


def _is_compliant(rb: Rulebook, shares, previous=None, **kw) -> bool:
    return not rb.check(shares, previous, **kw)


# ---------------------------------------------------------------------------
# Bounds and trimming
# ---------------------------------------------------------------------------

def test_equal_split_is_compliant():
    rb = _rulebook()
    assert _is_compliant(rb, equal_split(CHANNELS))


def test_trim_pulls_back_a_share_above_the_ceiling():
    rb = _rulebook()
    proposal = {"google": 0.70, "meta": 0.15, "amazon": 0.15}

    trimmed, feasible, notes = rb.trim(proposal)

    assert feasible
    assert abs(sum(trimmed.values()) - 1.0) < 1e-6
    assert trimmed["google"] <= 0.55 + 1e-9
    assert _is_compliant(rb, trimmed), rb.check(trimmed)
    assert notes, "a trim that changes the proposal should say what it changed"


def test_brand_rule_br02_caps_amazon_below_the_hard_limit():
    """BR02 caps Amazon at 50% even though the hard ceiling is 55% (TC12)."""
    rb = _rulebook()
    trimmed, feasible, _ = rb.trim({"google": 0.25, "meta": 0.20, "amazon": 0.55})

    assert feasible
    assert trimmed["amazon"] <= 0.50 + 1e-9


def test_floor_is_respected_when_a_channel_is_starved():
    rb = _rulebook()
    trimmed, feasible, _ = rb.trim({"google": 0.50, "meta": 0.48, "amazon": 0.02})

    assert feasible
    assert trimmed["amazon"] >= 0.15 - 1e-9
    assert _is_compliant(rb, trimmed)


def test_weekly_shift_cap_limits_a_big_move():
    rb = _rulebook()
    previous = equal_split(CHANNELS)
    proposal = {"google": 0.55, "meta": 0.30, "amazon": 0.15}

    trimmed, feasible, _ = rb.trim(proposal, previous_shares=previous)

    assert feasible
    assert rb.shift_pp(trimmed, previous) <= 15.0 + 1e-6
    assert _is_compliant(rb, trimmed, previous)


def test_degraded_mode_halves_the_shift_cap():
    """E2: on stale data the agent may move only half as far."""
    rb = _rulebook()
    previous = equal_split(CHANNELS)
    proposal = {"google": 0.55, "meta": 0.30, "amazon": 0.15}

    normal, _, _ = rb.trim(proposal, previous_shares=previous)
    degraded, _, _ = rb.trim(proposal, previous_shares=previous, degraded=True)

    assert rb.shift_pp(degraded, previous) <= 7.5 + 1e-6
    assert rb.shift_pp(degraded, previous) < rb.shift_pp(normal, previous)


def test_infeasible_rulebook_is_reported_not_silently_fixed():
    """If the floors cannot all be met, the agent must escalate, not guess."""
    data = copy.deepcopy(_rulebook().data)
    data["hard_limits"]["min_channel_share"] = 0.40  # 3 x 40% = 120% of budget
    rb = Rulebook(data)

    _, feasible, notes = rb.trim(equal_split(CHANNELS))

    assert not feasible
    assert notes


def test_trim_is_idempotent():
    """Trimming a compliant allocation must not move it."""
    rb = _rulebook()
    once, _, _ = rb.trim({"google": 0.70, "meta": 0.15, "amazon": 0.15})
    twice, _, _ = rb.trim(once)

    for c in CHANNELS:
        assert abs(once[c] - twice[c]) < 1e-6


# ---------------------------------------------------------------------------
# Approval routing (stage S5)
# ---------------------------------------------------------------------------

def test_small_clean_shift_is_auto_executed():
    """TC01/TC02: within 5 pp on clean data, the agent acts and reports."""
    rb = _rulebook()
    previous = equal_split(CHANNELS)
    proposal = {"google": 0.3733, "meta": 0.3133, "amazon": 0.3134}

    decision = rb.approval_decision(proposal, previous)

    assert decision["route"] == "auto_execute"
    assert decision["shift_pp"] <= 5.0
    assert decision["reasons"] == []


def test_large_shift_needs_manager_approval():
    """TC03/TC07: above the threshold the manager signs off."""
    rb = _rulebook()
    previous = equal_split(CHANNELS)
    proposal = {"google": 0.45, "meta": 0.30, "amazon": 0.25}

    decision = rb.approval_decision(proposal, previous)

    assert decision["route"] == "approve"
    assert any("threshold" in r for r in decision["reasons"])


def test_anomaly_forces_approval_even_on_a_tiny_shift():
    """TC09/TC10: a flagged week is never auto-executed."""
    rb = _rulebook()
    previous = equal_split(CHANNELS)
    proposal = {"google": 0.34, "meta": 0.33, "amazon": 0.33}

    decision = rb.approval_decision(proposal, previous, anomaly_flags=["google"])

    assert decision["route"] == "approve"
    assert decision["shift_pp"] <= 5.0
    assert any("anomaly" in r for r in decision["reasons"])


def test_trimmed_or_degraded_proposal_forces_approval():
    rb = _rulebook()
    previous = equal_split(CHANNELS)
    proposal = {"google": 0.34, "meta": 0.33, "amazon": 0.33}

    assert rb.approval_decision(proposal, previous, was_trimmed=True)["route"] == "approve"
    assert rb.approval_decision(proposal, previous, degraded=True)["route"] == "approve"


def test_infeasible_week_escalates():
    """E3 second branch: nothing complies, so a human decides."""
    rb = _rulebook()
    decision = rb.approval_decision(equal_split(CHANNELS), feasible=False)

    assert decision["route"] == "escalate"
    assert decision["escalate_to"]


def test_tc16_new_brand_rule_takes_effect_without_a_code_change():
    """
    TC16: the manager adds 'cap Meta at 40%'. Activating the rule in the
    rulebook is enough - no agent code is touched.
    """
    data = copy.deepcopy(_rulebook().data)
    before = Rulebook(data)
    proposal = {"google": 0.25, "meta": 0.50, "amazon": 0.25}
    assert before.check(proposal) == []

    for rule in data["brand_rules"]:
        if rule["id"] == "BR01":
            rule["active"] = True
    after = Rulebook(data)

    violations = after.check(proposal)
    assert any(v.rule == "BR01" for v in violations)

    trimmed, feasible, _ = after.trim(proposal)
    assert feasible and trimmed["meta"] <= 0.40 + 1e-9


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
    print(f"\n{len(tests) - failed}/{len(tests)} guardrail tests passed")
    sys.exit(1 if failed else 0)
