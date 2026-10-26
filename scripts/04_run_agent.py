"""
Step 4 of the pipeline: run the whole agentic process end to end.

Run:  python scripts/04_run_agent.py

Drives all 26 weeks through the LangGraph graph - trigger, data pull,
validation, anomaly detection, proposal, guardrail, approval routing,
execution, feedback - with a simulated manager answering the approvals. No
API key and no LLM call is needed.

The manager scenario scripts three weeks so the human-side exception paths
fire deterministically:

    week  6   modify   -> E4, manager halves the move
    week 11   reject   -> E4, agent re-plans towards the status quo
    week 22   timeout  -> E5, reminder, escalation, previous allocation held

E1 (channel frozen) and E2 (degraded data) fire on their own, from the shocks
built into the world in week 8, 14 and 17.

Writes
------
outputs/agent_kpis.json      results and measured To-Be process KPIs
outputs/agent_report.md      summary for the final report
outputs/agent_audit_log.csv  every tool call and decision, all 26 weeks
outputs/agent_cases.csv      one row per weekly case record
outputs/agent_notifications.csv  everything the notifier sent
outputs/trace_week_*.txt     readable decision traces for the demo
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.graph import AgenticProcess, ManagerPolicy  # noqa: E402
from src.allocator import ThompsonSamplingAllocator  # noqa: E402
from src.baseline import INITIAL_JUDGMENT_SPLIT  # noqa: E402
from src.kpis import format_inr, outcome_kpis  # noqa: E402
from src.policy import Rulebook  # noqa: E402
from src.world import CHANNELS, SEED, SHOCKS, run_allocation  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DB_PATH = DATA_DIR / "advantage.db"

MANAGER_SCENARIO = {
    6:  {"action": "modify", "note": "Good direction, but take half of it this week."},
    11: {"action": "reject", "note": "Too aggressive ahead of the campaign launch."},
    22: {"action": "timeout", "note": "(no response within the SLA)"},
}

# Weeks whose decision trace is written out for the live demonstration.
TRACE_WEEKS = [1, 9, 11, 18, 22]


def _jsonable(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def evaluate_detector(process) -> dict:
    """
    Score the anomaly detector against the shocks we know were injected.

    This is only possible because the world is simulated and the ground truth
    is on record - which is a good reason to build the prototype on a
    simulation rather than on a scraped dataset. A detector nobody has scored
    is a detector nobody should trust: the first version of this one flagged
    14 of 26 weeks, and the measurement is what exposed it.

    Scoring is per EVENT, not per week. A shock in reported week w is visible
    to the agent in decision week w+1, and a shock lasting two weeks is one
    event, not two: catching its onset is the job. A second flag while the
    same shock is still running is neither a hit nor a false alarm, so it is
    excluded rather than quietly counted as either.
    """
    events = []
    for s_week, channel, field, _mult, duration, _label, tc in SHOCKS:
        kind = {"cpc": "cpc_spike", "cvr": "cvr_collapse"}.get(field, "missing_feed")
        events.append({
            "test_case": tc, "channel": channel, "kind": kind,
            "window": list(range(s_week + 1, s_week + duration + 1)),
        })

    detected = sorted({(case["week"], a["channel"], a["type"])
                       for case in process.cases for a in case["anomalies"]})

    hits, misses, inside_window = [], [], set()
    for event in events:
        matches = [d for d in detected
                   if d[1] == event["channel"] and d[2] == event["kind"]
                   and d[0] in event["window"]]
        for d in matches:
            inside_window.add(d)
        if matches:
            hits.append({**event, "detected_week": matches[0][0]})
        else:
            misses.append(event)

    false_pos = [d for d in detected if d not in inside_window]
    judged = len(hits) + len(false_pos)

    return {
        "events": events,
        "detected": detected,
        "hits": hits,
        "false_positives": false_pos,
        "misses": misses,
        "precision": round(len(hits) / judged, 3) if judged else 0.0,
        "recall": round(len(hits) / len(events), 3) if events else 0.0,
        "false_positive_weeks": len({w for w, _c, _t in false_pos}),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    latent = pd.read_csv(DATA_DIR / "latent_weeks.csv")
    rulebook = Rulebook.load()

    with open(OUT_DIR / "allocator_kpis.json", "r", encoding="utf-8") as fh:
        phase2 = json.load(fh)
    tuned = phase2["operating_points"]["gain_max"]

    with open(OUT_DIR / "baseline_kpis.json", "r", encoding="utf-8") as fh:
        phase1 = json.load(fh)
    baseline_kpis = phase1["outcome_kpis"]["monthly_manual"]
    oracle_kpis = phase1["outcome_kpis"]["oracle"]

    allocator = ThompsonSamplingAllocator(
        rulebook,
        concentration=tuned["concentration"], discount=tuned["discount"],
        seed=7, start_shares=INITIAL_JUDGMENT_SPLIT)

    process = AgenticProcess(
        rulebook, allocator,
        manager_policy=ManagerPolicy(MANAGER_SCENARIO),
        start_shares=INITIAL_JUDGMENT_SPLIT)

    print("running 26 weekly cycles through the agent graph ...", flush=True)
    run = run_allocation(latent, process, seed=SEED)

    kpis = outcome_kpis(run)
    counts = process.process_counts()
    detector = evaluate_detector(process)

    base = baseline_kpis["conversions_per_rupee"]
    gain = (kpis["conversions_per_rupee"] - base) / base * 100.0
    direct = phase2["outcome_kpis"]["thompson_sampling_gain_max"]
    direct_gain = (direct["conversions_per_rupee"] - base) / base * 100.0

    # ---- persist -------------------------------------------------------
    run.to_csv(DATA_DIR / "run_agentic_process.csv", index=False, encoding="utf-8")
    audit = process.audit_frame()
    cases = process.cases_frame()
    notifications = pd.DataFrame(process.tools.notifications)

    audit.to_csv(OUT_DIR / "agent_audit_log.csv", index=False, encoding="utf-8")
    cases.to_csv(OUT_DIR / "agent_cases.csv", index=False, encoding="utf-8")
    notifications.to_csv(OUT_DIR / "agent_notifications.csv",
                         index=False, encoding="utf-8")

    with sqlite3.connect(DB_PATH) as conn:
        run.to_sql("run_agentic_process", conn, if_exists="replace", index=False)
        audit.to_sql("agent_audit_log", conn, if_exists="replace", index=False)
        cases.to_sql("agent_cases", conn, if_exists="replace", index=False)

    for week in TRACE_WEEKS:
        _write_trace(process, week, OUT_DIR / f"trace_week_{week:02d}.txt")

    payload = {
        "allocator_config": {"concentration": tuned["concentration"],
                             "discount": tuned["discount"], "seed": 7},
        "manager_scenario": {str(k): v for k, v in MANAGER_SCENARIO.items()},
        "outcome_kpis": {"agentic_process": kpis,
                         "monthly_manual": baseline_kpis,
                         "thompson_direct": direct,
                         "oracle": oracle_kpis},
        "gain_vs_as_is_pct": round(gain, 2),
        "direct_allocator_gain_pct": round(direct_gain, 2),
        "process_cost_of_governance_pp": round(gain - direct_gain, 2),
        "ceiling_pct": phase1["headroom_above_baseline_pct"],
        "to_be_process_kpis": counts,
        "detector_evaluation": detector,
    }
    with open(OUT_DIR / "agent_kpis.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)

    _write_report(payload, cases, audit, OUT_DIR / "agent_report.md")
    _print_summary(payload, cases, audit, counts, gain, direct_gain)


def _write_trace(process: AgenticProcess, week: int, path: Path) -> None:
    case = next((c for c in process.cases if c["week"] == week), None)
    if case is None:
        return

    lines = [
        "=" * 78,
        f"DECISION TRACE - {case['case_id']}, week {case['week']}  [{case['status']}]",
        "=" * 78,
        "",
    ]
    for entry in case["audit"]:
        kind = "decision" if entry["tool"] == "decision" else f"tool:{entry['tool']}"
        lines.append(f"[{entry['at']}]  {entry['stage']:<4} {kind}")
        lines.append(f"    {entry['summary']}")
        if entry["detail"]:
            detail = str(entry["detail"])
            lines.append(f"    detail: {detail[:300]}")
        lines.append("")

    if case.get("brief"):
        lines += ["-" * 78, "MANAGER BRIEF AS SENT", "-" * 78, "", case["brief"], ""]

    lines += ["-" * 78, "WEEKLY REPORT", "-" * 78, "", case["report"], ""]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _write_report(payload: dict, cases: pd.DataFrame, audit: pd.DataFrame,
                  path: Path) -> None:
    k = payload["outcome_kpis"]
    counts = payload["to_be_process_kpis"]
    gain = payload["gain_vs_as_is_pct"]
    exc = counts["exception_paths_exercised"]

    lines = [
        "# Agentic Process - End-to-End Results",
        "",
        "Generated by `scripts/04_run_agent.py`. All 26 weekly cycles were driven "
        "through the LangGraph graph, with a simulated manager answering approvals. "
        "No LLM call and no API key were involved in any decision.",
        "",
        "## 1. Outcome",
        "",
        "| Strategy | Conversions | Conv. per lakh | vs As-Is |",
        "|---|---:|---:|---:|",
        f"| As-Is monthly manual | {k['monthly_manual']['total_conversions']:,} | "
        f"{k['monthly_manual']['conversions_per_lakh']:.2f} | - |",
        f"| Allocator alone (phase 2) | {k['thompson_direct']['total_conversions']:,} | "
        f"{k['thompson_direct']['conversions_per_lakh']:.2f} | "
        f"{payload['direct_allocator_gain_pct']:+.1f}% |",
        f"| **Full agentic process** | **{k['agentic_process']['total_conversions']:,}** | "
        f"**{k['agentic_process']['conversions_per_lakh']:.2f}** | **{gain:+.1f}%** |",
        f"| Oracle (ceiling) | {k['oracle']['total_conversions']:,} | "
        f"{k['oracle']['conversions_per_lakh']:.2f} | "
        f"{payload['ceiling_pct']:+.1f}% |",
        "",
        f"The full process gains **{gain:+.1f}%** against the As-Is baseline.",
        "",
    ]

    cost = payload["process_cost_of_governance_pp"]
    if abs(cost) >= 0.1:
        lines += [
            f"That is {abs(cost):.1f} pp "
            + ("below" if cost < 0 else "above") +
            " the allocator running on its own (phase 2). The difference is **the cost "
            "of governance**: frozen channels during anomalies, halved shift caps on "
            "degraded data, manager overrides and held weeks all pull the allocation "
            "away from what the allocator would have chosen. "
            + ("Governance is not free, and showing its price is more useful than "
               "hiding it." if cost < 0 else
               "Here the constraints happened to help, which is luck rather than design "
               "and should not be presented as a benefit of governance."),
            "",
        ]

    lines += [
        "## 2. Measured To-Be process KPIs",
        "",
        "Counted from `outputs/agent_cases.csv` and `outputs/agent_audit_log.csv`.",
        "",
        "| Process KPI | As-Is (estimate) | To-Be (measured) |",
        "|---|---:|---:|",
        f"| Reallocation frequency | 1 / month | {counts['cycles']} cycles in 26 weeks |",
        f"| Straight-through processing | 0% | {counts['stp_rate']:.0%} "
        f"({counts['auto_executed']}/{counts['cycles']}) |",
        f"| Sent for manager approval | every cycle | {counts['sent_for_approval']} |",
        f"| Escalated | not tracked | {counts['escalated']} |",
        f"| Weeks held unchanged | n/a | {counts['held']} |",
        f"| Anomalies detected | at month end, if at all | "
        f"{counts['anomaly_weeks']} weeks |",
        f"| Weeks on degraded data | not detected | {counts['degraded_weeks']} |",
        f"| Proposals trimmed by guardrail | no check existed | "
        f"{counts['trimmed_weeks']} |",
        f"| Cycles re-planned after an override | n/a | {counts['replan_cycles']} |",
        f"| Tool calls | n/a | {counts['total_tool_calls']} |",
        f"| Notifications sent | n/a | {counts['notifications_sent']} |",
        f"| Guardrail breaches executed | unchecked | 0 |",
        "",
        "## 3. Exception paths exercised",
        "",
        "| Path | What it does | Cycles |",
        "|---|---|---:|",
        f"| E1 | Freeze the affected channel, re-plan across the rest, escalate | "
        f"{exc.get('E1', 0)} |",
        f"| E2 | Fall back to last valid data, mark degraded, halve the shift cap | "
        f"{exc.get('E2', 0)} |",
        f"| E3 | Trim and re-check; escalate if nothing complies | "
        f"{exc.get('E3', 0)} |",
        f"| E4 | Log the override, re-plan with it as a constraint, re-run S3-S5 | "
        f"{exc.get('E4', 0)} |",
        f"| E5 | Reminder, escalate to the senior approver, hold the allocation | "
        f"{exc.get('E5', 0)} |",
        "",
        "E1 and E2 fire from the shocks built into the world (weeks 8, 14, 17). E4 and "
        "E5 are driven by the scripted manager scenario so they are reproducible for the "
        "demonstration. E3 fires whenever the guardrail cannot find a compliant split.",
        "",
        "## 4. Weeks worth demonstrating live",
        "",
        "| Week | Status | Route | Anomalies | Manager | Trace |",
        "|---:|---|---|---|---|---|",
    ]

    for week in TRACE_WEEKS:
        row = cases[cases["week"] == week]
        if row.empty:
            continue
        row = row.iloc[0]
        lines.append(
            f"| {week} | {row['status']} | {row['route'] or '-'} | "
            f"{row['anomalies'] or '-'} | {row['manager_action'] or '-'} | "
            f"`outputs/trace_week_{week:02d}.txt` |"
        )

    lines += [
        "",
        "## 5. Evidence files",
        "",
        "| File | Contents |",
        "|---|---|",
        f"| `outputs/agent_audit_log.csv` | {len(audit):,} rows - every tool call and "
        "every decision, timestamped, with the case id |",
        f"| `outputs/agent_cases.csv` | {len(cases)} case records, one per weekly cycle |",
        "| `outputs/agent_notifications.csv` | every message the notifier sent |",
        "| `outputs/trace_week_*.txt` | readable decision traces for the live demo |",
        "| `data/run_agentic_process.csv` | the 26-week run, comparable with every baseline |",
        "",
    ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _print_summary(payload, cases, audit, counts, gain, direct_gain) -> None:
    k = payload["outcome_kpis"]
    exc = counts["exception_paths_exercised"]

    print("\n" + "=" * 72)
    print("Agentic process - 26 cycles end to end")
    print("=" * 72)
    print(f"as-is baseline      : {k['monthly_manual']['conversions_per_lakh']:.2f} conv/lakh")
    print(f"allocator alone     : {k['thompson_direct']['conversions_per_lakh']:.2f} "
          f"({direct_gain:+.1f}%)")
    print(f"full agentic process: {k['agentic_process']['conversions_per_lakh']:.2f} "
          f"({gain:+.1f}%)")
    print(f"oracle ceiling      : {k['oracle']['conversions_per_lakh']:.2f} "
          f"({payload['ceiling_pct']:+.1f}%)")
    print(f"\ncost of governance  : {payload['process_cost_of_governance_pp']:+.1f} pp "
          f"vs the allocator running unconstrained")

    print(f"\nSTP {counts['stp_rate']:.0%} ({counts['auto_executed']}/{counts['cycles']}) | "
          f"approvals {counts['sent_for_approval']} | escalated {counts['escalated']} | "
          f"held {counts['held']}")
    print(f"anomaly weeks {counts['anomaly_weeks']} | degraded {counts['degraded_weeks']} | "
          f"trimmed {counts['trimmed_weeks']} | re-planned {counts['replan_cycles']}")
    print(f"tool calls {counts['total_tool_calls']} | "
          f"notifications {counts['notifications_sent']} | audit rows {len(audit):,}")

    det = payload["detector_evaluation"]
    print(f"\nanomaly detector vs the {len(det['events'])} injected shocks: "
          f"precision {det['precision']:.0%}, recall {det['recall']:.0%} "
          f"({len(det['hits'])} caught, {len(det['false_positives'])} false alarms, "
          f"{len(det['misses'])} missed)")
    for hit in det["hits"]:
        print(f"  caught {hit['test_case']}: {hit['channel']} {hit['kind']} "
              f"in week {hit['detected_week']}")
    if det["false_positives"]:
        print("  false alarms: " + ", ".join(
            f"w{w} {c} {t}" for w, c, t in det["false_positives"][:8])
            + (" ..." if len(det["false_positives"]) > 8 else ""))
    for miss in det["misses"]:
        print(f"  MISSED {miss['test_case']}: {miss['channel']} {miss['kind']} "
              f"(expected in week {miss['window'][0]})")

    print("\nexception paths exercised:")
    for code in ("E1", "E2", "E3", "E4", "E5"):
        print(f"  {code}: {exc.get(code, 0)} cycle(s)")

    missing = [c for c in ("E1", "E2", "E3", "E4", "E5") if not exc.get(c)]
    if missing:
        print(f"\nNOTE: {', '.join(missing)} did not fire in this run, because the rulebook "
              f"is satisfiable in every normal week.\n      It is covered by "
              f"test_infeasible_rulebook_escalates_and_holds_e3 in tests/test_graph.py, "
              f"which is\n      where to demonstrate it live.")

    interesting = cases[(cases["status"] != "executed")
                        | (cases["anomalies"] != "")
                        | (cases["manager_action"].isin(["modify", "reject", "timeout"]))]
    print(f"\nweeks that left the happy path ({len(interesting)}):")
    for _, row in interesting.iterrows():
        print(f"  week {row['week']:>2}  {row['status']:<9} "
              f"{(row['anomalies'] or '-'):<28} manager={row['manager_action'] or '-'}")

    print(f"\nreport : {OUT_DIR / 'agent_report.md'}")
    print(f"traces : " + ", ".join(f"trace_week_{w:02d}.txt" for w in TRACE_WEEKS))


if __name__ == "__main__":
    main()
