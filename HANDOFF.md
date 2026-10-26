# Handoff — state of this project as of 27 Oct 2026

Everything about *what was built* is in `README.md`. This file records only the
things that existed in the chat and nowhere else: what is finished, what is
open, and what will bite you.

## What is done

| Piece | Where |
|---|---|
| Python system (world, allocator, policy, 7-stage LangGraph agent graph) | `src/` |
| Streamlit dashboard with live human approval | `app/dashboard.py` |
| 61 unit tests + the 16 proposal test cases, all passing | `tests/`, `scripts/05_run_test_cases.py` |
| n8n version — 20 nodes, 24 connections | `n8n/advantage_agentic_workflow.json` |
| Final report (16 sections, per guidelines §7B) | https://claude.ai/artifact/83iVZguW1rRtct5GKBUZMg |
| GitHub repo (public) | https://github.com/pkpiyush186-blip/advantage-agentic |

Headline result: **+8.8% conversions** on identical spend (47,341 vs 43,528),
against an oracle ceiling of +12.7% — so it captures 72% of what was
achievable. 0 policy breaches across 26 weeks.

## Open items

**1. n8n import — not yet verified.**
The JavaScript in all 12 Code nodes was executed for real against the live
dataset and works. What was never checked is n8n's own node configuration. If
the import fails it will be one of exactly four things, in this order of
likelihood:
- the `IF` node condition format (D1, D2)
- the `Switch` node condition format ("Route the decision")
- the `Wait` node's form parameters ("Await manager decision")
- the HTTP Request node's options (S1)

Paste the error into a new chat and it is a quick fix. The logic is not the
suspect.

**2. The LLM orchestrator — promised but not built.**
Proposal §9.2 says "the orchestrator is an LLM-driven planner (LangGraph)".
What exists is deterministic routing: the decisions D1/D2/D3 are rules, not a
model. This is defensible in a viva — a budget system that can hallucinate a
15% shift is worse than one that cannot — but it *is* a deviation from the
submitted proposal, so say so before an examiner finds it.

Building it would need an `ANTHROPIC_API_KEY` (none is set). Cost under $1 on
Haiku. The deterministic rules would stay underneath as the safety net.
`src/agents/graph.py` has a `narrate()` method — the only LLM reference in the
codebase — which is off by default.

**3. The dashboard is a local process, not a link.**
`localhost:8501` only exists while something is running it. Start it yourself:

```
cd C:\Users\pkpiy\Downloads\AdVantage_Agentic
python -m streamlit run app/dashboard.py
```

Streamlit's session state resets on every page reload, so weeks you have
already run disappear if you refresh. That is Streamlit, not a bug here.

**4. The repo is public.** Needed for Streamlit Cloud. After the viva:

```
gh repo edit pkpiyush186-blip/advantage-agentic --visibility private --accept-visibility-change-consequences
```

## Three findings to have ready for the viva

These are the honest ones, and examiners reward them:

1. **Monthly manual re-planning beats a frozen split by only +0.4%.** Nearly
   all of the gain comes from reacting *weekly*, not from reacting at all.
2. **Exploration contributes nothing** (−0.01pp). The greedy ablation matches
   Thompson Sampling. The gain is recency-weighted re-estimation; the tuned
   discount came out at 0, which is a finding about this world, not a bug.
3. **O2 (+8–12% ROAS) and the 40–60% straight-through-processing target cannot
   both be met.** At gain-max, STP is 0%; at the balanced setting, STP is 42%
   but the gain drops to +5.6%. Governance and autonomy trade against each
   other, and the project reports the trade-off curve rather than picking the
   number that looks good.

The largest assumption in the whole thing is `ROTATION_AMPLITUDE = 0.45` in
`src/world.py` — how strongly channel performance rotates over a 13-week
cycle. It sets the size of the prize. It is documented as an assumption in the
report, not tuned to hit the target.

## To open a new chat

> Continue the AdVantage Agentic MBA project. Read HANDOFF.md and README.md in
> C:\Users\pkpiy\Downloads\AdVantage_Agentic first.
