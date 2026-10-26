"""
The LangGraph graph, and the runner that drives one week through it.

Graph shape
-----------

    START -> S1 pull -> S2 validate
                          |
                     D1 --+-- anomaly? --> exception (E1 freeze / E2 degrade)
                          |                      |
                          +------- clean --------+--> S3 propose
                                                        |
                                                   S4 guardrail
                                                        |
                                                  D2 --+-- infeasible (E3) --> escalate
                                                        |
                                                   S5 route approval
                                                        |
                                           D3 --+-- auto --> S6 execute
                                                 |
                                                 +-- approve --> await manager
                                                                     |
                                                   approve ----------+--> S6 execute
                                                   modify/reject (E4) --> back to S3
                                                   timeout (E5) ---------> hold
                                                        |
                                 escalate / hold -------+--> S7 report -> END

The same week can take six different paths. Which one it takes is decided
from the case record at run time by the three `route_*` functions, which is
the difference between an agentic process and a script with an if-statement
in it: no path is privileged, and the orchestrator re-plans rather than
failing when something goes wrong.

Running without an API key
--------------------------
The graph needs no LLM to run. Every routing decision is made from the case
state, and the manager brief is generated from a template. `narrate()` can
rewrite that brief more fluently with Claude if a key is present, but it is
off by default and it never supplies a number or makes a decision - a brief
that authorises spending should not contain figures invented by a language
model.
"""

from __future__ import annotations

import os

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..allocator import ThompsonSamplingAllocator
from ..policy import Rulebook
from ..world import CHANNELS, FESTIVE_WEEKS, WEEKLY_BUDGET
from .core import CaseRecord, ToolBelt, new_case
from .nodes import ProcessAgents


def build_graph(agents: ProcessAgents, *, use_checkpointer: bool = True):
    """Wire the stages, the three decision points and the exception paths."""
    graph = StateGraph(CaseRecord)

    graph.add_node("s1_pull", agents.s1_trigger_and_pull)
    graph.add_node("s2_validate", agents.s2_validate_and_detect)
    graph.add_node("exception", agents.handle_exception)
    graph.add_node("s3_propose", agents.s3_propose)
    graph.add_node("s4_guardrail", agents.s4_guardrail)
    graph.add_node("s5_route", agents.s5_route_approval)
    graph.add_node("await_manager", agents.s5_await_manager)
    graph.add_node("timeout", agents.handle_timeout)
    graph.add_node("escalate", agents.handle_escalation)
    graph.add_node("s6_execute", agents.s6_execute)
    graph.add_node("hold", agents.hold_previous)
    graph.add_node("s7_report", agents.s7_feedback_and_report)

    graph.add_edge(START, "s1_pull")
    graph.add_edge("s1_pull", "s2_validate")

    # D1
    graph.add_conditional_edges("s2_validate", agents.route_after_validation,
                                {"exception": "exception", "propose": "s3_propose"})
    graph.add_edge("exception", "s3_propose")
    graph.add_edge("s3_propose", "s4_guardrail")

    # D2
    graph.add_conditional_edges("s4_guardrail", agents.route_after_guardrail,
                                {"escalate": "escalate", "route": "s5_route"})

    # D3
    graph.add_conditional_edges("s5_route", agents.route_after_approval,
                                {"escalate": "escalate", "human": "await_manager",
                                 "execute": "s6_execute"})

    # E4 / E5
    graph.add_conditional_edges("await_manager", agents.route_after_manager,
                                {"replan": "s3_propose", "execute": "s6_execute",
                                 "timeout": "timeout", "hold": "hold"})

    graph.add_edge("timeout", "hold")
    graph.add_edge("escalate", "hold")
    graph.add_edge("s6_execute", "s7_report")
    graph.add_edge("hold", "s7_report")
    graph.add_edge("s7_report", END)

    return graph.compile(checkpointer=MemorySaver() if use_checkpointer else None)


# ---------------------------------------------------------------------------
# The simulated manager
# ---------------------------------------------------------------------------

class ManagerPolicy:
    """
    Stands in for the human during a 26-week batch run.

    In the live demo the same decision comes from a real person through
    LangGraph `interrupt()`; the graph is identical either way, which is the
    point - the human-in-the-loop step is part of the process design, not a
    feature of the demo harness.

    `scenario` scripts specific weeks so the exception paths can be exercised
    deterministically for the test cases (TC04 modify, TC13 reject, TC14 no
    response).
    """

    def __init__(self, scenario: dict[int, dict] | None = None,
                 default_action: str = "approve"):
        self.scenario = scenario or {}
        self.default_action = default_action
        self.answered: list[dict] = []

    def __call__(self, case: CaseRecord) -> dict:
        # A re-plan is the manager looking at a revised proposal they asked
        # for, so they accept it rather than rejecting their own correction.
        if case.get("replans", 0) > 0:
            decision = {"action": "approve",
                        "note": "revised proposal accepted after override"}
        else:
            decision = dict(self.scenario.get(case["week"],
                                              {"action": self.default_action}))

        if decision["action"] == "modify" and not decision.get("shares"):
            decision["shares"] = self._halve_the_move(case)

        self.answered.append({"week": case["week"], **decision})
        return decision

    @staticmethod
    def _halve_the_move(case: CaseRecord) -> dict:
        """A manager who wants the move, but only half of it."""
        prev, prop = case["previous_shares"], case["proposal"]
        halved = {c: prev[c] + 0.5 * (prop[c] - prev[c]) for c in CHANNELS}
        total = sum(halved.values())
        return {c: v / total for c, v in halved.items()}


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

class AgenticProcess:
    """
    Drives one week through the graph, and presents itself as an allocator.

    Because `__call__(week, history) -> shares` matches the signature of every
    other strategy, `world.run_allocation` replays the whole agentic process
    against the identical world as the baselines, with the same noise stream
    and the same measurement code. The comparison in the report is therefore
    the process against the baseline, not two different measurements.
    """

    name = "agentic_process"
    description = "LangGraph multi-agent process: S1-S7, D1-D3, E1-E5"

    def __init__(self, rulebook: Rulebook, allocator: ThompsonSamplingAllocator, *,
                 manager_policy: ManagerPolicy | None = None,
                 human_in_the_loop: bool = False,
                 start_shares: dict[str, float] | None = None,
                 weekly_budget: float = WEEKLY_BUDGET,
                 budget_schedule: dict[int, float] | None = None,
                 festive_weeks=FESTIVE_WEEKS):
        self.rulebook = rulebook
        self.allocator = allocator
        self.tools = ToolBelt(rulebook, allocator)
        self.agents = ProcessAgents(self.tools,
                                    manager_policy=manager_policy or ManagerPolicy(),
                                    human_in_the_loop=human_in_the_loop,
                                    weekly_budget=weekly_budget)
        self.graph = build_graph(self.agents)
        self.festive_weeks = set(festive_weeks)

        # Finance can change the total mid-season (TC15). The agent never
        # changes it - the rulebook forbids that - it re-plans against it.
        self.weekly_budget = weekly_budget
        self.budget_schedule = dict(budget_schedule or {})

        start = dict(start_shares or {c: 1.0 / len(CHANNELS) for c in CHANNELS})
        total = sum(start.values())
        self.current = {c: v / total for c, v in start.items()}

        self.cases: list[CaseRecord] = []

    def budget_for(self, week: int) -> float:
        return float(self.budget_schedule.get(week, self.weekly_budget))

    def __call__(self, week: int, history: pd.DataFrame) -> dict[str, float]:
        self.agents.history = history

        case = new_case(week, self.current, festive=week in self.festive_weeks,
                        weekly_budget=self.budget_for(week))
        # The recursion limit is a deliberate safety net, not a formality. The
        # re-plan edge makes this graph cyclic, and a cycle that fails to
        # terminate should stop loudly after a handful of steps rather than
        # spin for ten thousand.
        config = {"configurable": {"thread_id": f"week-{week}"},
                  "recursion_limit": 40}
        final = self.graph.invoke(case, config=config)

        self.cases.append(final)
        self.current = dict(final["executed"])
        return dict(final["executed"])

    # -- reporting -------------------------------------------------------

    def audit_frame(self) -> pd.DataFrame:
        rows = [entry for case in self.cases for entry in case["audit"]]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["detail"] = frame["detail"].astype(str).str.slice(0, 400)
        return frame

    def cases_frame(self) -> pd.DataFrame:
        rows = []
        for case in self.cases:
            decision = case.get("manager_decision") or {}
            row = {
                "week": case["week"], "case_id": case["case_id"],
                "status": case["status"], "route": case.get("route", ""),
                "degraded": case["degraded"],
                "was_trimmed": case["was_trimmed"],
                "feasible": case["feasible"],
                "replans": case["replans"],
                "anomalies": "; ".join(a["type"] for a in case["anomalies"]),
                "exceptions": "; ".join(sorted({a["exception"]
                                                for a in case["anomalies"]})),
                "frozen": ";".join(case["frozen_channels"]),
                "manager_action": decision.get("action", ""),
                "approval_reasons": "; ".join(case["approval_reasons"]),
                "escalations": " | ".join(case.get("escalations", [])),
                "tool_calls": sum(1 for e in case["audit"] if e["tool"] != "decision"),
            }
            for c in CHANNELS:
                row[f"executed_{c}"] = round(case["executed"][c], 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def process_counts(self) -> dict:
        cases = self.cases
        n = len(cases)

        def count(pred) -> int:
            return sum(1 for c in cases if pred(c))

        auto = count(lambda c: c.get("route") == "auto_execute"
                     and c["status"] == "executed")
        exceptions: dict[str, int] = {}
        for case in cases:
            for code in sorted({a["exception"] for a in case["anomalies"]}):
                exceptions[code] = exceptions.get(code, 0) + 1
        for case in cases:
            if case["replans"]:
                exceptions["E4"] = exceptions.get("E4", 0) + 1
            if (case.get("manager_decision") or {}).get("action") == "timeout":
                exceptions["E5"] = exceptions.get("E5", 0) + 1
            if not case["feasible"]:
                exceptions["E3"] = exceptions.get("E3", 0) + 1

        return {
            "cycles": n,
            "auto_executed": auto,
            "sent_for_approval": count(lambda c: c.get("route") == "approve"),
            "escalated": count(lambda c: c["status"] == "escalated"),
            "held": count(lambda c: c["status"] == "held"),
            "stp_rate": round(auto / n, 4) if n else 0.0,
            "degraded_weeks": count(lambda c: c["degraded"]),
            "trimmed_weeks": count(lambda c: c["was_trimmed"]),
            "anomaly_weeks": count(lambda c: bool(c["anomalies"])),
            "replan_cycles": count(lambda c: c["replans"] > 0),
            "total_tool_calls": sum(
                1 for c in cases for e in c["audit"] if e["tool"] != "decision"),
            "notifications_sent": len(self.tools.notifications),
            "exception_paths_exercised": exceptions,
        }


# ---------------------------------------------------------------------------
# Optional LLM narration
# ---------------------------------------------------------------------------

def narrate(brief: str, model: str = "claude-opus-5-5") -> str:
    """
    Rewrite a manager brief more fluently with Claude.

    Optional and off by default. It is given the finished brief and asked only
    to re-word it: it supplies no figures, makes no decision and changes no
    allocation. If no API key is configured the original brief is returned
    unchanged, so the process never depends on it.
    """
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return brief

    try:
        import anthropic
    except ImportError:
        return brief

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=("You rewrite internal marketing-budget briefs so a busy manager can act "
                "on them in under a minute. Keep every number exactly as given, change no "
                "figure, add no recommendation that is not already in the text, and keep "
                "it under 180 words."),
        messages=[{"role": "user", "content": brief}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
