# n8n version — AdVantage Retail agentic budget operations

`advantage_agentic_workflow.json` is the same seven-stage process as the Python
prototype, rebuilt as an n8n workflow you can import and run.

**20 nodes · 24 connections · 7 stages · 3 decision points · 5 exception paths.**

---

## How to run it

Node.js on this machine is broken (`node --version` prints nothing), so running
n8n locally needs that fixed first. The quickest route is the browser.

### Option A — n8n Cloud (no install)

1. Sign up at **n8n.io** (free trial)
2. **Workflows → Import from File**
3. Choose `n8n/advantage_agentic_workflow.json`
4. Click **Execute workflow**

### Option B — locally, once Node works

```bash
npx -y n8n
```

Then open `http://localhost:5678` and import the same file.

---

## What it does

| Node | Stage | Agent |
|---|---|---|
| Manual trigger / Weekly trigger | — | fires Monday 09:00, or on demand |
| S1 · Performance DB | S1 | Data Monitor — a real HTTP call to the live dataset |
| S1 · Parse and pull | S1 | Data Monitor — falls back to the last valid week per channel |
| S2 · Validate and detect | S2 | Data Monitor — anomaly detection |
| **D1 · Anomaly?** | D1 | Orchestrator decision |
| E1/E2 · Handle exception | — | freeze the channel, or degrade and halve the cap |
| S3 · Propose allocation (RL) | S3 | Allocation agent — Thompson Sampling |
| S4 · Guardrail check | S4 | Guardrail agent — limits, brand rules, projection |
| **D2 · Compliant?** | D2 | Orchestrator decision |
| E3 · Escalate | — | nothing complies, hand to a human |
| S5 · Route for approval | S5 | Reporting agent — writes the manager brief |
| **D3 · Needs a human?** | D3 | Orchestrator decision |
| Await manager decision | — | **a real form: approve / reject / timeout** |
| E4/E5 · Manager decision | — | re-plan on reject, escalate on timeout |
| S6 · Execute budget | S6 | mock budget API |
| Hold previous split | — | the safe fallback: no budget moves |
| S7 · Report and audit | S7 | audit log + weekly report |

The **HTTP Request node is a genuine tool call** — it fetches the 26-week
dataset live from this repository, so the workflow is not reading values typed
into it.

The **Route the decision → S3** connection is the E4 re-plan loop. A rejected
proposal goes back to the Allocation agent with the rejection as a new
constraint, capped at two re-plans so a disagreement cannot loop forever. That
backward edge is the thing most n8n workflows do not have, and it is worth
pointing at during the viva.

## What to demonstrate

The workflow defaults to **week 18**, where Amazon's data feed is missing.
Running it shows:

1. The HTTP node pulling real data
2. D1 branching to the **exception** output
3. E2 marking the case degraded and halving the shift cap from 15 pp to 7.5 pp
4. S4 trimming the allocator's proposal, naming each rule it breached
5. The approval form appearing, because stale data always needs a human
6. S7 returning the audit trail

To see other paths, change `const WEEK = 18;` in the **Open case** node:

| Week | What happens |
|---|---|
| 9 | Google CPC spike — E1, channel frozen |
| 15 | Meta conversion collapse — E1 |
| 18 | Amazon feed missing — E2, degraded |
| 2 | A quiet week — clean path |

Choose **reject** on the approval form to watch the E4 loop re-plan and come
back for a second decision.

## Honest notes

**The JavaScript is tested; the node wiring is not.**

All 12 Code nodes were executed in a real JavaScript engine against the live
dataset, driving week 18 end to end plus the E3, E4 and E5 branches. No errors,
and the behaviour matches the Python system:

| | Python | n8n port |
|---|---|---|
| Missing feed detected | amazon, week 17 | amazon, week 17 |
| Degraded mode, cap halved | 15 → 7.5 pp | 15 → 7.5 pp |
| Fallback to last valid week | yes | yes |
| Guardrail trimmed Google to | 55.0% | 55.0% |
| Route | approve | approve |
| Reject → re-plan | pulls back to status quo | pulls back to status quo |
| Timeout → hold | executed = previous | executed = previous |

The proposals differ by a percentage point or two between the two versions
because Thompson Sampling *samples* — a different random draw gives a slightly
different split. That is the algorithm working, not a porting error.

**What is still untested** is n8n's own node configuration: the `IF` and
`Switch` condition format, the `Wait` node's form parameters, and the HTTP
Request node's options. Those are n8n-version-specific and can only be checked
by importing. If something fails on import it will be one of those four, not the
logic.

**This is a port, not the original.** These parts of the Python system could
not come across and remain only there:

- The hyperparameter tuning — 80 configurations across 8 training worlds
- The greedy ablation showing exploration contributes nothing
- The anomaly detector scored at 100% precision and recall
- 61 unit tests and the 16 proposal test cases
- The oracle upper bound that the +8.8% result is measured against

**Thompson Sampling here replays history each run.** The Python allocator
carries its posteriors forward across weeks in memory. n8n has no state between
executions, so this version rebuilds the posterior from the dataset each time it
runs. The result is the same; the mechanism is less elegant.

**Which to present.** The Python system is the one with the measured results,
the test suite and the report behind it. This workflow is the same process shown
on a canvas. Run both if you have time; lead with whichever your examiner asks
to see.
