"""
AdVantage Retail - agentic budget operations dashboard.

Run:  python -m streamlit run app/dashboard.py

This is the manager-facing side of the process. The agents do the work; this
screen is where a human exercises the oversight the autonomy design calls for
at stage S5 - approve, modify or reject - and where the audit trail and the
KPIs can be read.

It runs the identical graph the batch run uses. `tests/test_live.py` asserts
that a live session and a batch run with the same answers produce the same
numbers, so what is demonstrated here and what is reported are the same
prototype.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.graph import ManagerPolicy  # noqa: E402
from src.agents.live import AWAITING, LiveSession  # noqa: E402
from src.kpis import format_inr, outcome_kpis  # noqa: E402
from src.policy import Rulebook  # noqa: E402
from src.world import CHANNELS, SEED, WEEKLY_BUDGET  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
START_SHARES = {"google": 0.40, "meta": 0.35, "amazon": 0.25}

st.set_page_config(page_title="AdVantage Agentic Budget Ops",
                   page_icon="chart_with_upwards_trend", layout="wide")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_world() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "latent_weeks.csv")


@st.cache_data
def load_baseline() -> tuple[pd.DataFrame, dict]:
    run = pd.read_csv(DATA_DIR / "run_monthly_manual.csv")
    return run, outcome_kpis(run)


@st.cache_data
def load_tuning() -> dict:
    import json
    path = OUT_DIR / "allocator_kpis.json"
    if not path.exists():
        return {"concentration": 3.0, "discount": 0.0}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["operating_points"]["gain_max"]


def missing_prerequisites() -> list[str]:
    needed = {
        DATA_DIR / "latent_weeks.csv": "python scripts/01_generate_data.py",
        DATA_DIR / "run_monthly_manual.csv": "python scripts/02_run_baseline.py",
    }
    return [cmd for path, cmd in needed.items() if not path.exists()]


def new_session(concentration: float, discount: float) -> LiveSession:
    return LiveSession(load_world(), Rulebook.load(),
                       concentration=concentration, discount=discount,
                       allocator_seed=7, world_seed=SEED,
                       start_shares=START_SHARES)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar(session: LiveSession) -> None:
    st.sidebar.title("Weekly cycle")
    st.sidebar.progress(min(session.week / session.n_weeks, 1.0),
                        text=f"Week {session.week} of {session.n_weeks}")

    if session.awaiting:
        st.sidebar.warning(f"Week {session.week} is waiting for your decision.")
    elif session.finished:
        st.sidebar.success("All weeks complete.")

    st.sidebar.divider()

    disabled = session.awaiting or session.finished
    if st.sidebar.button("Run next week", type="primary",
                         use_container_width=True, disabled=disabled):
        session.start_next_week()
        st.rerun()

    skip = st.sidebar.number_input("Fast-forward weeks", 1, 26, 4,
                                   help="Runs ahead with every proposal approved, "
                                        "so you can reach an interesting week quickly.")
    if st.sidebar.button(f"Approve the next {skip} weeks",
                         use_container_width=True, disabled=disabled):
        session.auto_run(int(skip), ManagerPolicy())
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Allocator")
    st.sidebar.code(f"concentration = {session.allocator.concentration:g}\n"
                    f"discount      = {session.allocator.discount:g}", language="text")

    limits = session.rulebook.data["hard_limits"]
    approval = session.rulebook.data["approval"]
    st.sidebar.caption("Policy in force")
    st.sidebar.code(
        f"share      {limits['min_channel_share']:.0%} - {limits['max_channel_share']:.0%}\n"
        f"weekly cap {limits['max_weekly_shift_pp']:.0f} pp\n"
        f"auto-exec  <= {approval['auto_execute_max_shift_pp']:.0f} pp\n"
        f"SLA        {approval['approval_sla_hours']}h", language="text")

    st.sidebar.divider()
    if st.sidebar.button("Reset session", use_container_width=True):
        for key in ("session", "modify_open"):
            st.session_state.pop(key, None)
        st.rerun()


# ---------------------------------------------------------------------------
# KPI header
# ---------------------------------------------------------------------------

def kpi_header(session: LiveSession) -> None:
    run = session.run_frame()
    _, baseline_kpis = load_baseline()
    counts = session.process_counts()

    cols = st.columns(5)

    if run.empty:
        cols[0].metric("Conversions", "-")
        cols[1].metric("Per lakh spent", "-")
    else:
        kpis = outcome_kpis(run)
        baseline_rate = baseline_kpis["conversions_per_lakh"]
        delta = (kpis["conversions_per_lakh"] - baseline_rate) / baseline_rate * 100.0

        cols[0].metric("Conversions", f"{kpis['total_conversions']:,}",
                       help=f"Spend so far {format_inr(kpis['total_spend_inr'])}")
        cols[1].metric("Per lakh spent", f"{kpis['conversions_per_lakh']:.1f}",
                       f"{delta:+.1f}% vs As-Is",
                       help="The As-Is manager achieved "
                            f"{baseline_rate:.1f} over the full 26 weeks.")

    cols[2].metric("Cycles run", f"{counts['cycles']}")
    cols[3].metric("Straight-through", f"{counts['stp_rate']:.0%}",
                   help=f"{counts['auto_executed']} of {counts['cycles']} cycles ran "
                        "with no human touch.")
    cols[4].metric("Needing you", f"{counts['sent_for_approval']}",
                   help=f"{counts['escalated']} escalated, {counts['held']} held "
                        "unchanged.")


# ---------------------------------------------------------------------------
# The approval step (S5)
# ---------------------------------------------------------------------------

def approval_panel(session: LiveSession) -> None:
    if not session.awaiting:
        if session.finished:
            st.success("Every weekly cycle is complete. Nothing is waiting for you.")
        else:
            st.info("Nothing is waiting for a decision. "
                    "Use **Run next week** in the sidebar to start the next cycle.")
        return

    case = session.pending
    st.subheader(f"Approval needed - {case['case_id']}, week {case['week']}")

    if case["anomalies"]:
        for anomaly in case["anomalies"]:
            st.error(f"**{anomaly['type'].replace('_', ' ').title()}** - {anomaly['detail']}")
    if case["degraded"]:
        st.warning("This week is running on stale or incomplete data. "
                   "The agent has halved its own shift cap.")
    if case["frozen_channels"]:
        st.warning(f"Held frozen while the anomaly is investigated: "
                   f"{', '.join(case['frozen_channels'])}.")

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Proposed allocation**")
        table = pd.DataFrame([
            {
                "Channel": channel,
                # Streamlit number formats take the value as given, so these
                # are stored as percentages rather than fractions.
                "Current": case["previous_shares"][channel] * 100.0,
                "Proposed": case["proposal"][channel] * 100.0,
                "Move (pp)": (case["proposal"][channel]
                              - case["previous_shares"][channel]) * 100.0,
                "Budget": case["proposal"][channel] * WEEKLY_BUDGET,
            }
            for channel in CHANNELS
        ])
        st.dataframe(
            table, hide_index=True, use_container_width=True,
            column_config={
                "Current": st.column_config.NumberColumn(format="%.1f%%"),
                "Proposed": st.column_config.NumberColumn(format="%.1f%%"),
                "Move (pp)": st.column_config.NumberColumn(format="%+.1f"),
                "Budget": st.column_config.NumberColumn(format="Rs %.0f"),
            })

        st.caption(f"Largest move {case['shift_pp']:.1f} pp. "
                   + (f"Re-planned {case['replans']}x after your override. "
                      if case["replans"] else ""))

        if case["was_trimmed"]:
            with st.expander("The guardrail changed this proposal", expanded=True):
                for note in case["trim_notes"]:
                    st.write(f"- {note}")

        st.markdown("**Why this needs your approval**")
        for reason in case["reasons"]:
            st.write(f"- {reason}")

    with right:
        st.markdown("**Brief as sent**")
        st.text(case["brief"])

    st.divider()
    decision_buttons(session, case)


def decision_buttons(session: LiveSession, case: dict) -> None:
    cols = st.columns([1, 1, 1, 3])

    if cols[0].button("Approve", type="primary", use_container_width=True):
        session.submit({"action": "approve", "note": "approved in dashboard"})
        st.session_state.pop("modify_open", None)
        st.rerun()

    if cols[1].button("Modify", use_container_width=True):
        st.session_state["modify_open"] = not st.session_state.get("modify_open", False)
        st.rerun()

    if cols[2].button("Reject", use_container_width=True):
        session.submit({"action": "reject", "note": "rejected in dashboard"})
        st.session_state.pop("modify_open", None)
        st.rerun()

    if st.session_state.get("modify_open"):
        modify_form(session, case)


def modify_form(session: LiveSession, case: dict) -> None:
    st.markdown("#### Set your own split")
    st.caption("Your split is still checked against the rulebook. The agent will "
               "re-plan onto it and come back if it has to be trimmed.")

    cols = st.columns(len(CHANNELS))
    raw = {}
    for col, channel in zip(cols, CHANNELS):
        raw[channel] = col.slider(channel, 0, 100,
                                  int(round(case["proposal"][channel] * 100)),
                                  key=f"slider_{channel}_{case['week']}")

    total = sum(raw.values()) or 1
    shares = {c: raw[c] / total for c in CHANNELS}

    preview = pd.DataFrame([
        {"Channel": c, "Your split": shares[c] * 100.0,
         "Budget": shares[c] * WEEKLY_BUDGET}
        for c in CHANNELS
    ])
    st.dataframe(preview, hide_index=True, use_container_width=True,
                 column_config={
                     "Your split": st.column_config.NumberColumn(format="%.1f%%"),
                     "Budget": st.column_config.NumberColumn(format="Rs %.0f")})

    violations = session.rulebook.check(shares, case["previous_shares"],
                                        degraded=case["degraded"])
    if violations:
        st.warning("The guardrail will trim this:")
        for violation in violations:
            st.write(f"- {violation}")
        trimmed, feasible, _ = session.rulebook.trim(
            shares, previous_shares=case["previous_shares"], degraded=case["degraded"])
        if feasible:
            st.caption("It would become " + ", ".join(
                f"{c} {trimmed[c]:.1%}" for c in CHANNELS))
    else:
        st.success("This split complies with the rulebook.")

    if st.button("Send this back to the agent", type="primary"):
        session.submit({"action": "modify", "shares": shares,
                        "note": "manager set the split in the dashboard"})
        st.session_state["modify_open"] = False
        st.rerun()


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def performance_panel(session: LiveSession) -> None:
    run = session.run_frame()
    if run.empty:
        st.info("No weeks have run yet.")
        return

    baseline_run, _ = load_baseline()

    weekly = run.groupby("week")["conversions"].sum().cumsum()
    base_weekly = (baseline_run.groupby("week")["conversions"].sum()
                   .cumsum().loc[:weekly.index.max()])

    st.markdown("**Cumulative conversions on identical spend**")
    st.line_chart(pd.DataFrame({"Agentic process": weekly,
                                "As-Is monthly manual": base_weekly}))

    left, right = st.columns(2)

    with left:
        st.markdown("**Allocation each week**")
        shares = run.pivot(index="week", columns="channel", values="share")
        st.area_chart(shares[list(CHANNELS)])

    with right:
        st.markdown("**What the allocator believes**")
        beliefs = pd.DataFrame([
            {"channel": channel,
             "conversions per lakh": session.allocator.posteriors[channel].mean,
             "uncertainty (2 sd)": session.allocator.posteriors[channel].sd * 2}
            for channel in CHANNELS
        ])
        st.dataframe(beliefs, hide_index=True, use_container_width=True,
                     column_config={
                         "conversions per lakh":
                             st.column_config.NumberColumn(format="%.0f"),
                         "uncertainty (2 sd)":
                             st.column_config.NumberColumn(format="+/- %.1f")})
        st.caption("Posterior means from the Gamma-Poisson model. These are what the "
                   "proposals are drawn from.")

    st.markdown("**Weekly case summary**")
    cases = session.cases_frame()
    if not cases.empty:
        st.dataframe(
            cases[["week", "case_id", "status", "route", "manager_action",
                   "anomalies", "exceptions", "replans", "tool_calls"]],
            hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Audit log and traces
# ---------------------------------------------------------------------------

def audit_panel(session: LiveSession) -> None:
    audit = session.audit_frame()
    if audit.empty:
        st.info("The audit log fills as cycles run.")
        return

    cols = st.columns([1, 1, 3])
    weeks = sorted(audit["week"].unique())
    week = cols[0].selectbox("Week", ["all"] + weeks, index=0)
    tools = sorted(audit["tool"].unique())
    tool = cols[1].selectbox("Entry type", ["all"] + tools, index=0)

    view = audit
    if week != "all":
        view = view[view["week"] == week]
    if tool != "all":
        view = view[view["tool"] == tool]

    st.caption(f"{len(view)} of {len(audit)} entries. Every tool call and every "
               "routing decision is recorded as it happens.")
    st.dataframe(view[["at", "case_id", "stage", "tool", "summary", "detail"]],
                 hide_index=True, use_container_width=True, height=460)

    st.download_button("Download the audit log (CSV)",
                       audit.to_csv(index=False).encode("utf-8"),
                       file_name="agent_audit_log.csv", mime="text/csv")


def trace_panel(session: LiveSession) -> None:
    if not session.cases:
        st.info("Run a week to produce a decision trace.")
        return

    weeks = [case["week"] for case in session.cases]
    week = st.selectbox("Week", weeks, index=len(weeks) - 1)
    case = next(c for c in session.cases if c["week"] == week)

    st.caption(f"{case['case_id']} - {case['status']} - "
               f"{len([e for e in case['audit'] if e['tool'] != 'decision'])} tool calls")

    for entry in case["audit"]:
        kind = "decision" if entry["tool"] == "decision" else f"tool: {entry['tool']}"
        with st.container(border=True):
            st.markdown(f"`{entry['at']}` &nbsp; **{entry['stage']}** &nbsp; {kind}")
            st.write(entry["summary"])
            if entry["detail"]:
                st.caption(str(entry["detail"])[:600])

    if case.get("report"):
        st.markdown("**Weekly report**")
        st.text(case["report"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("AdVantage Retail - Agentic Budget Operations")
    st.caption("Weekly digital advertising budget allocation across Google, Meta and "
               "Amazon Ads. Five agents propose and check; you approve.")

    blockers = missing_prerequisites()
    if blockers:
        st.error("Run these first, from the project root:")
        for command in blockers:
            st.code(command, language="bash")
        return

    if "session" not in st.session_state:
        tuned = load_tuning()
        st.session_state["session"] = new_session(
            float(tuned["concentration"]), float(tuned["discount"]))

    session: LiveSession = st.session_state["session"]

    sidebar(session)
    kpi_header(session)
    st.divider()

    label = "Approval" + (" (1)" if session.awaiting else "")
    tabs = st.tabs([label, "Performance", "Audit log", "Decision trace"])

    with tabs[0]:
        approval_panel(session)
    with tabs[1]:
        performance_panel(session)
    with tabs[2]:
        audit_panel(session)
    with tabs[3]:
        trace_panel(session)


if __name__ == "__main__":
    main()
