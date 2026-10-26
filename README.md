# Agentic AI for Digital Advertising Operations

Budget Planning, Allocation, Monitoring and Reallocation — a multi-agent system with a
reinforcement-learning allocation engine.

**Course:** Advanced AI for Business Automation · Formative Assessment Project · Dr. Neha Advani
**Team:** Prashant Kumar · Aditya Gupta · Sandeep Ojha (MBA Jan 2026, Group 5)
**Case:** AdVantage Retail Pvt. Ltd. — a fictional mid-sized Indian D2C fashion brand

---

## Status

| Phase | Deliverable | State |
|---|---|---|
| 1 | Simulated world, policy rulebook, As-Is baseline, guardrail tests | ✅ complete |
| 2 | Thompson Sampling allocator, tuned and measured | ✅ complete |
| 3 | LangGraph agent graph (S1–S7, D1–D3, E1–E5) | ✅ complete |
| 4 | Streamlit dashboard, live human-in-the-loop approval | ✅ complete |
| 5 | All 16 test cases as one runnable suite | ✅ complete |

The whole prototype needs **no API key and makes no LLM calls**. This is the week 3–11
milestone from Section 18 of the proposal. 61 tests pass, and all 16 proposal test cases
pass.

## How to run

```bash
pip install -r requirements.txt
python scripts/01_generate_data.py     # build the frozen 26-week world
python scripts/02_run_baseline.py      # measure the As-Is baseline + the ceiling
python scripts/03_run_allocator.py     # tune and measure the allocator (~90s)
python scripts/04_run_agent.py         # run all 26 cycles through the agent graph
python scripts/05_run_test_cases.py    # the 16 proposal test cases, with a table
python tests/test_policy.py            # 14 guardrail tests
python tests/test_allocator.py         # 15 allocator tests
python tests/test_graph.py             # 18 graph / exception-path tests
python tests/test_live.py              # 10 human-in-the-loop tests
python tests/test_cases.py             #  4 checks over the 16 proposal cases

python -m streamlit run app/dashboard.py   # the manager dashboard
```

Everything is deterministic for the seed in `src/world.py`, so re-running reproduces
identical numbers. That is deliberate — it is what makes the results in the report
checkable by someone else.

## Layout

```
config/policy_rulebook.json   the rules the Guardrail agent reads at run time
src/world.py                  the simulated world (ground truth + response model)
src/policy.py                 guardrail engine: bounds, checks, trimming, approval routing
src/baseline.py               static split, monthly manual As-Is, oracle upper bound
src/allocator.py              discounted Thompson Sampling + the greedy ablation
src/agents/core.py            the case record and the eight tools
src/agents/nodes.py           the seven stage agents and the exception handlers
src/agents/graph.py           the LangGraph wiring, simulated manager, runner
src/agents/live.py            live session driver - pauses the graph for a real human
app/dashboard.py              the Streamlit manager dashboard
src/testcases.py              the 16 proposal test cases, as executable scenarios
src/kpis.py                   outcome KPIs (measured) and process KPIs (declared)
scripts/01_generate_data.py   generates data/latent_weeks.csv + data/advantage.db
scripts/02_run_baseline.py    writes outputs/baseline_kpis.json + baseline_report.md
scripts/03_run_allocator.py   tunes on training worlds, evaluates on the demo world
scripts/04_run_agent.py       drives all 26 cycles through the graph, writes the audit log
scripts/05_run_test_cases.py  runs the 16 proposal cases and prints the pass/fail table
tests/test_policy.py          guardrail tests, one per rule / test case
tests/test_allocator.py       learning, safety, audit-trail and reproducibility tests
tests/test_graph.py           routing, all five exception paths, audit trail
tests/test_live.py            pause/resume, and live-vs-batch parity
tests/test_cases.py           the 16 proposal cases, wired into the test run
```

---

## 1. The data

### 1.1 What it is

A simulated 26-week history for three paid channels — Google Ads, Meta Ads, Amazon Ads.
The data is **synthetic**, generated in Python. No real account data, no personal data and
no confidential data is used anywhere in this project, in line with Section 9 of the
guidelines.

The world is stored as *latent weekly parameters*, not as fixed observed metrics:

| Variable | Meaning |
|---|---|
| `cpc` | cost per click for that channel-week (₹) |
| `ctr` | click-through rate |
| `cvr` | conversion rate per click |
| `aov` | average order value (₹) |
| `alpha` | diminishing-returns exponent for the channel |
| `festive` | inside the festive window |
| `feed_missing` | the data feed for that channel-week did not arrive |
| `shock` | label of any injected shock |

Observed metrics (impressions, clicks, conversions, revenue) are **derived from whatever
spend an allocation chooses**, not stored. This is the key design decision in the data
layer: it means the manual baseline and the agentic run can be replayed against the
*identical* world, so the before/after comparison is a controlled test rather than two
different simulations. Without it, any measured "gain" could just be two different runs
of luck.

### 1.2 Response model

For weekly spend *S* on a channel:

```
clicks      = S / cpc
clicks_ref  = s_ref / cpc                        (s_ref = equal split)
conversions = cvr · clicks_ref^(1−α) · clicks^α
```

`α < 1` gives **diminishing returns** — doubling spend less than doubles conversions.
This is what creates an interior optimum. Without it, the best allocation would always be
"put everything in this week's strongest channel", there would be nothing to balance, and
the project would not need an allocator. Observed conversions are then drawn from
`Poisson(mean)`, so the agent sees noisy feedback and never the latent truth — which is
the reason a Thompson Sampling allocator has a job at all.

### 1.3 Calibration

Base values are set to plausible ranges for Indian D2C fashion e-commerce (CPC ₹12–22,
CVR 1.8–3.3%, AOV ₹1,800–2,400). They are **illustrative, not sourced from a specific
report**. Conversion rates are set so that conversions per rupee (`cvr / cpc`) is an exact
tie at 0.0015 across all three channels at the reference spend. The tie is deliberate: if
one channel were permanently best, a static split would be the right answer and the
project premise would collapse.

On top of that:

- **Mean-reverting random walk** on CPC and CVR (σ = 5% / 7%, reversion 0.45).
- **Slow rotation** in channel effectiveness, 13-week period, one phase per channel,
  amplitude 0.45 — so the strongest channel keeps changing (Google leads 10 weeks, Amazon
  11, Meta 5).
- **Festive window** weeks 19–23: CVR ×1.35, CPC ×1.25 (demand and competition both rise).
- **Three injected shocks**, each one existing to drive a specific test case:

| Week | Channel | Shock | Test case |
|---|---|---|---|
| 8–9 | Google | CPC ×1.60 | TC09 |
| 14–15 | Meta | CVR ×0.40 (tracking break) | TC10 |
| 17 | Amazon | data feed missing | TC11 |

### 1.4 Quality issues and limitations — read this before quoting any result

These are real limitations, not disclaimers:

1. **The rotation amplitude is the most consequential assumption in the project.** The
   gain any allocator can earn is bounded by how much the channels actually differ week to
   week. It is set from realism (a channel being ~2× more efficient than another in a given
   week is ordinary once CPC and CVR both move), **not** from the target in the proposal.
   A reader who disagrees with 0.45 should change `ROTATION_AMPLITUDE` and re-run — the
   headroom moves with it, and `scripts/02_run_baseline.py` prints the ceiling every time
   precisely so this stays visible.
2. **No cross-channel effects.** No halo, no cannibalisation, no view-through. Real
   channels interact; here they are independent.
3. **No competitor reaction.** Nothing bids against us in response to our own moves.
4. **Attribution is perfect.** Every conversion is attributed to the channel that bought
   the click. Real attribution is noisy and contested, which is exactly what TC10 (a
   suspected tracking break) gestures at but does not model.
5. **Linear-in-spend within a week, apart from `α`.** No creative fatigue, no frequency
   caps, no audience exhaustion.
6. **Conversions are the objective, not profit.** AOV differs by channel, so the
   revenue-maximising split is not the same as the conversion-maximising one. The proposal
   optimises conversions per rupee, so that is what is optimised; ROAS is reported
   alongside but never optimised.
7. **Simulated data cannot validate business ROI.** The results show whether the mechanism
   works, and in which direction. They are not an estimate of what AdVantage Retail would
   actually earn.

## 2. The policy rulebook

`config/policy_rulebook.json` holds the limits, brand rules and approval matrix. Two
design decisions:

- **The rules are data, not code.** The Guardrail agent retrieves them at run time. This
  is the "knowledge retrieval" capability claimed in the proposal, and it is what makes
  TC16 ("a new brand rule caps Meta at 40%") a one-line data change — `test_policy.py`
  proves the cap binds without any agent code being touched.
- **The guardrail contains no LLM.** It is deterministic Python. The agent that proposes
  an allocation is never the agent that approves it, and the approving side must give the
  same answer every time. A guardrail that occasionally hallucinates is not a guardrail.

Core limits: per-channel share 15–55%, weekly shift cap 15 pp, auto-execute threshold
5 pp, approval SLA 48 h, Amazon capped at 50% by brand rule BR02. Degraded data halves
the shift cap.

The central operation is a **projection**: given a desired allocation, find the nearest
allocation satisfying every bound that still sums to 100% of the budget. If none exists,
the week is infeasible and escalates to a human (exception path E3).

## 3. Measured baseline results

Full numbers in [outputs/baseline_report.md](outputs/baseline_report.md). Headline, over
26 weeks at an identical ₹11,50,000 per week:

| Strategy | Conversions | Conv. per lakh | Cost per conversion |
|---|---:|---:|---:|
| Static split (set once on judgment) | 43,361 | 145.02 | ₹690 |
| **Monthly manual (the As-Is baseline)** | **43,528** | **145.58** | **₹687** |
| Oracle (perfect foresight, upper bound) | 49,043 | 164.02 | ₹610 |

Three findings that matter for the report:

**Monthly manual reallocation beats a frozen split by only +0.4%.** The As-Is manager
reviews the split six times in 26 weeks and gains almost nothing. The reason is visible in
the data: the manager chases *last month's* winner, and with a 13-week rotation the
previous month's leader is often already past its peak. This is a stronger argument for
the project than the proposal makes — the problem is not that the manager reallocates too
rarely, it is that monthly chasing of a rotating signal adds almost no value.

**The ceiling is +12.7%.** Perfect weekly foresight, inside the same guardrails, earns
12.7% more conversions per rupee than the As-Is manager. No allocator can do better than
that in this world.

**The proposal's +8–12% target is therefore reachable but demanding.** It requires the
allocator to capture 63–94% of the theoretical ceiling, while respecting a 15 pp weekly
cap and learning from noisy feedback. Phase 2 measures what it actually achieves.

## 4. Allocator results (phase 2)

Full numbers in [outputs/allocator_report.md](outputs/allocator_report.md). The allocator
is discounted Thompson Sampling over a Gamma–Poisson model of conversions per rupee,
projected onto the policy set by the guardrail. Hyperparameters were tuned on **eight
separate training worlds** and evaluated once on the demo world, which never picks a
hyperparameter.

| Strategy | Conv. per lakh | vs As-Is | STP |
|---|---:|---:|---:|
| As-Is monthly manual | 145.58 | — | 0% |
| Greedy recency (ablation) | 158.82 | +9.1% | — |
| **Thompson Sampling (gain-max)** | **158.70** | **+9.1%** | 1% |
| Thompson Sampling (balanced) | 153.91 | +5.6% | 42% |
| Oracle (ceiling) | 164.02 | +12.7% | — |

**Four findings, all of which should be said before an examiner says them:**

**1. +9.1% meets the proposal target**, and captures 72% of the 12.7% ceiling. Range
across five allocator seeds is +8.8% to +9.5%, so it is not one lucky run.

**2. Exploration contributes essentially nothing.** The ablation — identical model and
discounting, acting on the posterior *mean* instead of a sample, so no exploration at all
— gains +9.1% too. At roughly 500 conversions per channel per week the posterior is very
tight (see `fig_learning.png`), so a Thompson draw lands almost on top of the mean. The
honest claim is that the gain comes from **fast, recency-weighted re-estimation plus a
disciplined projection onto the policy set**; Thompson Sampling is the formulation used to
get there and it costs nothing, but it is not where the value is. Exploration would matter
far more for a smaller advertiser, a new channel with no history, or a cycle with tens
rather than hundreds of conversions. The 15% per-channel floor is also already doing part
of the exploration job by guaranteeing every channel keeps producing data.

**3. The best memory setting is no memory at all.** The optimal `discount` is 0 — estimate
from last week alone. The grid spans the parameter's whole domain [0, 1], so this is a
finding about the world rather than a gap in the search: because channel advantage
rotates, evidence older than about a week is not merely less useful, it is actively
misleading. That is also the clean explanation for why the As-Is manager, who effectively
carries a month of memory, gains only +0.4% over a frozen split.

**4. The proposal set two targets that cannot both be met.** O2 wanted +8–12% on
conversions per rupee; Section 15 wanted 40–60% straight-through processing. An allocator
that leans harder into the leading channel earns more conversions but crosses the 5 pp
auto-execute threshold and gets trimmed by the 15 pp cap more often — and both force a
manager approval under the rulebook. At the gain-max point STP is 1%; the balanced point
reaches 42% STP but only +5.6%. `fig_tradeoff.png` plots every configuration on both axes.

This is a decision for the marketing manager, not for the model: pay for more conversions
with more of their own review time, or accept a smaller gain for a process that mostly
runs itself. Presenting that choice is a stronger answer than presenting one tuned number.

Note that manager *effort* falls sharply even at 1% STP — approving a prepared proposal
with a written rationale takes minutes, against an estimated 8 hours of compiling and
deciding in the As-Is. Approval count and effort are different KPIs and the report should
not conflate them.

## 5. The agentic process (phase 3)

Full numbers in [outputs/agent_report.md](outputs/agent_report.md). All 26 cycles run
through a LangGraph graph: S1 pull → S2 validate → S3 propose → S4 guardrail → S5 approval
→ S6 execute → S7 report, with decision points D1/D2/D3 and five exception paths. Because
the runner presents itself as an allocator (`__call__(week, history) -> shares`), the whole
process is replayed against the same frozen world as every baseline, with the same
measurement code.

| | Conv. per lakh | vs As-Is |
|---|---:|---:|
| As-Is monthly manual | 145.58 | — |
| Allocator alone (phase 2) | 158.70 | +9.0% |
| **Full agentic process** | **158.33** | **+8.8%** |
| Oracle (ceiling) | 164.02 | +12.7% |

**The cost of governance is −0.2 pp.** Freezing channels during anomalies, halving the
shift cap on degraded data, manager overrides and one held week together cost two tenths of
a percentage point against the allocator running unconstrained. That number is worth
quoting: governance is not free, and showing its price is more useful than hiding it.

**Measured, not estimated:** 26 cycles, 218 tool calls, 386 audit rows, 33 notifications,
3 anomaly weeks, 1 degraded week, 2 re-planned cycles, 1 held week, 0 guardrail breaches
executed.

### The anomaly detector is scored, not assumed

Because the world is simulated, the ground truth is on record, so the detector can be
measured rather than trusted. It now runs at **100% precision and 100% recall** against the
three injected shocks, catching each in exactly the week it becomes visible (TC09 week 9,
TC10 week 15, TC11 week 18).

It did not start there. The first version flagged **14 of 26 weeks** — a detector that cries
wolf, which in practice means a manager stops reading the alerts. Three defects, each found
by measuring:

1. **It compared levels against a trailing median.** These channels trend by design, so the
   rotation itself looked like an anomaly. Fixed by detecting sudden *week-on-week change*
   instead of deviation from a level.
2. **Observed conversion rate was confounded by the agent's own spending.** Under
   diminishing returns a channel run at a 45% share converts at a different observed rate
   from the same channel at 20%, so the agent's own budget moves registered as channel
   anomalies. Fixed by normalising each week back to an equal-split reference spend.
   Observed CPC needs no correction — spend cancels out of it exactly.
3. **A two-week shock hid its own second week.** The standard deviation used for the
   "is this unusual?" test was inflated by the first week of the shock. Fixed with the
   median absolute deviation.

The flag now requires the move to be both larger than a fixed percentage *and* unusual for
that channel. This is the single most useful thing in the project to walk an examiner
through, because it shows measurement driving design rather than decorating it.

### Two bugs worth knowing about before the viva

**Routing decisions were being silently discarded.** D1, D2 and D3 were originally computed
inside LangGraph's conditional-edge functions. LangGraph keeps what a *node* returns and
discards what an *edge* writes, so the decisions never reached the case record or the audit
log — and the re-plan counter never incremented, so a manager who rejected a proposal was
asked forever and the graph span until it hit the recursion limit. All three decisions are
now made and logged inside nodes; the edges only read them.

**A rejection could end in silent execution.** After two rejections the agent proposed a
move small enough to fall under the 5 pp auto-execute threshold and executed it *without
asking again*. The rulebook now lists `after_override` under
`always_require_approval_on`: once a manager has overridden a cycle, that cycle cannot
auto-execute. This is exactly the kind of hole an examiner probes — "what stops it just
doing something smaller anyway after I say no?"

### Exception paths

| Path | Behaviour | Where it fires |
|---|---|---|
| E1 | Freeze the affected channel, re-plan across the rest, escalate | weeks 9, 15 |
| E2 | Fall back to the last valid week, mark degraded, halve the shift cap | week 18 |
| E3 | Trim and re-check; escalate and hold if nothing complies | `tests/test_graph.py` |
| E4 | Log the override, re-plan with it as a constraint, re-run S3–S5 | weeks 6, 11 |
| E5 | Reminder, escalate to the senior approver, hold the allocation | week 22 |

E3 does not fire in a normal run because the rulebook is satisfiable every week. It is
covered by `test_infeasible_rulebook_escalates_and_holds_e3`, which is where to demonstrate
it live.

### For the demonstration

`outputs/trace_week_*.txt` holds readable decision traces — every tool call and routing
decision, timestamped, with the manager brief as it was sent. Week 18 (E2, degraded data)
is the best single trace to show: the feed is missing, the agent falls back, halves its own
shift cap, proposes 68% to Google, and the guardrail cuts it to 55% citing four specific
violations.

## 6. The dashboard (phase 4)

```bash
python -m streamlit run app/dashboard.py
```

This is the manager-facing side of the process, and it is what the live demonstration
should be run from. Four tabs:

- **Approval** — the S5 decision. Proposed split against current, the budget in rupees per
  channel, the anomalies flagged this week, what the guardrail trimmed and why, and the
  brief exactly as it was sent. Approve, Modify or Reject.
- **Performance** — cumulative conversions against the As-Is baseline on identical spend,
  the allocation each week, what the allocator currently believes about each channel, and a
  per-week case summary.
- **Audit log** — every tool call and routing decision, filterable by week and by entry
  type, downloadable as CSV.
- **Decision trace** — the full stage-by-stage trace for any week.

**Modify is the moment worth demonstrating.** Set your own split with the sliders and the
screen tells you *before you send it* which rules it breaks and what the guardrail would
trim it to. Send it anyway and the agent re-plans onto your split, the guardrail still
trims it, and it comes back to you — a manager override does not bypass the rulebook.
`test_a_modification_is_honoured_but_still_checked` asserts that.

### It is the same prototype, not a demo mode

`human_in_the_loop=True` swaps the simulated manager for LangGraph's `interrupt()`; the
graph is compiled with a checkpointer so a paused cycle survives between requests, and the
decision returns as `Command(resume=...)`. Everything else is identical code.

`test_live_matches_the_batch_run_given_the_same_answers` runs the same weeks twice — once
through the batch runner, once through the live session answering identically — and asserts
the conversions and every allocation match exactly. If the dashboard and the report ever
disagreed, neither could be trusted; this is how that is prevented rather than hoped for.

A single week can pause **more than once**: reject or modify, and the orchestrator re-plans
and comes straight back for another decision. The driver handles that as a loop, not a
one-shot handoff.

## 7. The 16 test cases (phase 5)

```bash
python scripts/05_run_test_cases.py
```

**16 of 16 pass** — 8 normal, 6 exception, 2 what-if. Full table in
[outputs/test_case_results.md](outputs/test_case_results.md). Each case constructs a
scenario the agent has not seen, runs one real weekly cycle through the real graph, and
checks what it did against Section 13 of the proposal. Nothing is stubbed. The script
exits non-zero on any failure, so it belongs in the submission checklist.

| TC | What the agent did |
|---|---|
| TC01 | 0.2 pp shift, auto-executed, 12 audit entries |
| TC02 | Meta up, 1.1 pp, auto-executed |
| TC03 | 9.7 pp — above the 5 pp threshold, clear of the cap, routed to the manager |
| TC04 | Manager halved the move; override logged, re-planned once |
| TC05 | Weakest channel still funded at exactly the 15% floor |
| TC06 | Uniform festive uplift moved the split 0.9 pp and breached nothing |
| TC07 | Cap-bound 15.0 pp shift, approved, executed |
| TC08 | Google spent 70% of its budget; capped at 24.5% and re-planned |
| TC09 | CPC spike flagged, Google frozen, budget re-planned, manager notified |
| TC10 | Conversion collapse flagged, Meta held pending investigation |
| TC11 | Feed missing; fell back to last valid week, cap halved to 8 pp |
| TC12 | Allocator wanted 100%; trimmed to the 55% ceiling exactly |
| TC13 | Rejection logged, re-planned towards the status quo |
| TC14 | No reply in 48h; reminder, escalation to the Senior Marketing Head, split held |
| TC15 | Budget cut to ₹9,20,000; re-planned and executed with every limit respected |
| TC16 | Allocator wanted Meta at 94%; executed at 40%, BR01 cited |

**Two cases needed capability that did not exist, so it was built rather than the checks
being softened.** TC08 (pacing) now flags a channel that cannot spend what it was given
and caps it at roughly what it delivered — a capped channel is not misbehaving, it simply
cannot absorb the budget, so it is capped rather than frozen. TC15 moved the weekly total
onto the case record so finance can change it mid-season as a data change; the agent still
may not change the total itself, because the rulebook forbids it.

**A passing suite deserves scrutiny, so it got some.** The first version passed 16/16 too,
with four weak checks: TC03 and TC07 were the same test written twice (both saturated the
15 pp cap); TC12 named the 55% ceiling but started from an equal split, where the shift cap
binds first and the ceiling is never exercised; TC16's check had an `or` that could pass
without confirming the rule was cited; and TC15 compared rupee totals with a one-rupee
tolerance. All four are now tightened, and all still pass — TC03 lands at 9.7 pp, which is
the "+9 pp" the proposal predicted.

## 8. As-Is process KPIs

These are **estimates** for the fictional case, exactly as Section 6 of the proposal
states — they are carried in the code with an `estimated=True` flag so an estimate is
never printed as a measurement. The To-Be equivalents will be *counted* from the agent
case log in phase 5.

| Process KPI | As-Is | Basis |
|---|---:|---|
| Reallocation frequency | 1 / month | estimate |
| Mean detection lag | 15 days | estimate |
| Manager + analyst effort per cycle | 8 hours | estimate |
| Human touchpoints per cycle | 5 | estimate |
| Straight-through processing | 0% | by definition |
| Cost per cycle | ₹4,000 | 8 h × ₹500/h |
| Guardrail breaches checked | not systematically | the As-Is process has no check |

## 9. Acknowledgement of AI assistance

Per Section 9 of the guidelines, to be adapted to the institution's required format.

The code in `src/`, `scripts/` and `tests/` was written with the assistance of
Claude (Anthropic), working from the team's own submitted project proposal. The team
specified the case, the process, the seven stages, the autonomy levels, the exception
paths, the rulebook limits and the 16 test cases; the AI assistance was in implementing
those specifications in Python and in calibrating the simulation. All design decisions
recorded in this README — the choice of a frozen latent world, the diminishing-returns
response model, the deterministic guardrail, the oracle upper bound, and the revision of
the +8–12% target against measured headroom, the train/test split for hyperparameters, and
the greedy ablation that shows exploration is not what produces the gain — are documented
here so the team can explain and defend each one.

No AI tool received confidential, personal or proprietary data at any point.

## 10. What is left

The prototype deliverable (Section 7C of the guidelines) is complete: it runs the process
end to end, uses tools, keeps memory, pauses for human approval, handles exceptions, logs
everything, and reports KPIs against a baseline. What remains is written work and
rehearsal, not code:

- **The final report.** Sections 1–8 above map onto most of it. The parts that still need
  writing by hand are the business and industry background, the implementation roadmap and
  change management (proposal Section 17), and the ROI case.
- **Two numbers the report must not dodge.** Straight-through processing is 0% at the
  gain-max operating point against a 40–60% target, and the +8.8% gain sits against a
  measured ceiling of +12.7%. Both are defensible; neither is improved by being quiet
  about it.
- **Rehearse the live demo.** Week 18 (E2) is the single best trace. The Modify flow in
  the dashboard is the best interactive moment, because the guardrail visibly overrules
  the manager.
- **A dry run on the presentation laptop.** "Demonstration readiness" is a mark of its own,
  and a Streamlit app that needs restarting mid-demo costs more than it should.
