"""
The 16 proposal test cases, as part of the normal test run.

`scripts/05_run_test_cases.py` is the version to put on screen in the viva -
it prints the table. This is the same 16 cases wired into the test suite so a
regression in any of them fails alongside everything else, rather than only
being noticed the next time somebody runs the script by hand.

Run either way:
    python -m pytest tests -q
    python tests/test_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.testcases import SPECS, run_all  # noqa: E402

_RESULTS = None


def _results() -> list[dict]:
    global _RESULTS
    if _RESULTS is None:
        _RESULTS = run_all()
    return _RESULTS


def test_every_proposal_case_passes():
    failures = [f"{r['tc']}: {r['evidence']}" for r in _results() if not r["passed"]]
    assert not failures, "\n    " + "\n    ".join(failures)


def test_all_sixteen_cases_are_present():
    """Section 13 promises 16 cases: 8 normal, 6 exception, 2 what-if."""
    results = _results()
    assert len(results) == 16

    counts: dict[str, int] = {}
    for row in results:
        counts[row["type"]] = counts.get(row["type"], 0) + 1

    assert counts == {"Normal": 8, "Exception": 6, "What-if": 2}, counts
    assert [r["tc"] for r in results] == [f"TC{i:02d}" for i in range(1, 17)]


def test_each_case_reports_evidence():
    """A pass with nothing to show is not evidence of anything."""
    for row in _results():
        assert row["evidence"].strip(), f"{row['tc']} passed without saying what happened"


def test_every_exception_path_is_covered():
    covered = " ".join(spec[5] for spec in SPECS)
    for path in ("E1", "E2", "E3", "E4", "E5"):
        assert path in covered, f"{path} is not exercised by any test case"


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
    print(f"\n{len(tests) - failed}/{len(tests)} test-case suite checks passed")
    sys.exit(1 if failed else 0)
