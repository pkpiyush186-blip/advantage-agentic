"""
Live session driver - the same graph, paused for a real human.

In the 26-week batch run a `ManagerPolicy` answers the approvals so the
process can be measured. Here the graph stops at stage S5 and waits for an
actual person. Nothing about the agents changes: `human_in_the_loop=True`
swaps the simulated manager for LangGraph `interrupt()`, the graph is
compiled with a checkpointer so the paused cycle survives between HTTP
requests, and the decision arrives back as `Command(resume=...)`.

That is the point worth making in the viva: the human-in-the-loop step is
part of the process design, not something bolted onto the demo. The batch run
and the dashboard execute identical code and, given identical answers,
produce identical numbers - `tests/test_live.py` asserts exactly that.

One subtlety the driver has to handle: a single week can interrupt more than
once. If the manager modifies or rejects, the orchestrator re-plans (E4) and
comes straight back for another decision, so `submit()` can return another
pause rather than a finished week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from langgraph.types import Command

from ..allocator import ThompsonSamplingAllocator
from ..policy import Rulebook
from ..world import CHANNELS, SEED, WEEKLY_BUDGET, observe
from .core import new_case
from .graph import AgenticProcess

AWAITING = "awaiting_manager"
WEEK_COMPLETE = "week_complete"
SEASON_COMPLETE = "season_complete"


class LiveSession:
    """Drives the agent graph one week at a time, pausing for a human at S5."""

    def __init__(self, latent: pd.DataFrame, rulebook: Rulebook, *,
                 concentration: float = 3.0, discount: float = 0.0,
                 allocator_seed: int = 7, world_seed: int = SEED,
                 start_shares: dict[str, float] | None = None,
                 weekly_budget: float = WEEKLY_BUDGET):
        self.latent = latent
        self.rulebook = rulebook
        self.weekly_budget = weekly_budget
        self.n_weeks = int(latent["week"].max())

        self.allocator = ThompsonSamplingAllocator(
            rulebook, concentration=concentration, discount=discount,
            seed=allocator_seed, start_shares=start_shares)
        self.process = AgenticProcess(
            rulebook, self.allocator, human_in_the_loop=True,
            start_shares=start_shares, weekly_budget=weekly_budget)

        # Same noise stream as world.run_allocation, so a live session and a
        # batch run of the same weeks are directly comparable.
        self.rng = np.random.default_rng(world_seed + 1)

        self.week = 0
        self.rows: list[dict] = []
        self.pending: dict | None = None
        self._config: dict | None = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def awaiting(self) -> bool:
        return self.pending is not None

    @property
    def finished(self) -> bool:
        return self.week >= self.n_weeks and not self.awaiting

    @property
    def cases(self) -> list:
        return self.process.cases

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def run_frame(self) -> pd.DataFrame:
        """The run so far, in the same shape every baseline produces."""
        frame = self.history()
        if frame.empty:
            return frame
        frame = frame.copy()
        clicks = frame["clicks"].replace(0, np.nan)
        frame["cpc_observed"] = (frame["spend"] / clicks).round(4)
        frame["cvr_observed"] = (frame["conversions"] / clicks).round(6)
        frame["conv_per_rupee"] = (
            frame["conversions"] / frame["spend"].replace(0, np.nan)).round(8)
        return frame

    # ------------------------------------------------------------------
    # Driving the graph
    # ------------------------------------------------------------------

    def start_next_week(self) -> dict:
        if self.awaiting:
            raise RuntimeError("a decision is still outstanding for this week")
        if self.week >= self.n_weeks:
            return {"state": SEASON_COMPLETE}

        self.week += 1
        self.process.agents.history = self.history()

        case = new_case(self.week, self.process.current,
                        festive=self.week in self.process.festive_weeks,
                        weekly_budget=self.process.budget_for(self.week))
        self._config = {"configurable": {"thread_id": f"week-{self.week}"},
                        "recursion_limit": 40}

        return self._advance(self.process.graph.invoke(case, config=self._config))

    def submit(self, decision: dict) -> dict:
        """Hand the manager answer back to the paused graph."""
        if not self.awaiting:
            raise RuntimeError("nothing is waiting for a decision")

        self.pending = None
        return self._advance(
            self.process.graph.invoke(Command(resume=dict(decision)),
                                      config=self._config))

    def _advance(self, result: dict) -> dict:
        interrupts = result.get("__interrupt__")
        if interrupts:
            # A re-plan can pause again in the same week, so this is a loop,
            # not a one-shot handoff.
            self.pending = dict(interrupts[0].value)
            return {"state": AWAITING, "payload": self.pending}

        case = {k: v for k, v in result.items() if k != "__interrupt__"}
        self._observe(case)

        self.process.cases.append(case)
        self.process.current = dict(case["executed"])

        return {"state": WEEK_COMPLETE, "case": case}

    # ------------------------------------------------------------------
    # The world responds
    # ------------------------------------------------------------------

    def _observe(self, case: dict) -> None:
        week_latent = self.latent[self.latent["week"] == case["week"]].set_index("channel")

        shares = dict(case["executed"])
        total = sum(shares.values())
        shares = {c: v / total for c, v in shares.items()}

        for channel in CHANNELS:
            row = week_latent.loc[channel]
            result = observe(row, shares[channel] * self.weekly_budget,
                             self.rng, self.weekly_budget)
            self.rows.append({
                "week": int(case["week"]),
                "channel": channel,
                "share": round(float(shares[channel]), 6),
                **result,
                "festive": bool(row["festive"]),
                "feed_missing": bool(row["feed_missing"]),
                "shock": row["shock"],
            })

    # ------------------------------------------------------------------
    # Convenience for the dashboard
    # ------------------------------------------------------------------

    def auto_run(self, weeks: int, policy) -> list[dict]:
        """
        Fast-forward some weeks with a policy answering for the manager.

        The dashboard uses this to skip ahead to an interesting week without
        clicking through every approval.
        """
        outcomes = []
        for _ in range(weeks):
            if self.finished:
                break
            step = self.start_next_week() if not self.awaiting else {"state": AWAITING}
            while step["state"] == AWAITING:
                step = self.submit(policy(dict(self.pending)))
            outcomes.append(step)
        return outcomes

    def process_counts(self) -> dict:
        return self.process.process_counts()

    def audit_frame(self) -> pd.DataFrame:
        return self.process.audit_frame()

    def cases_frame(self) -> pd.DataFrame:
        return self.process.cases_frame()
