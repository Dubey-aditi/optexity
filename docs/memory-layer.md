# Memory layer

An `agentic_task` solves a task with an LLM choosing every step — and solves it again
from scratch on every run. The memory layer records what the agent did, turns it into
deterministic steps, and saves them back to the recording on the dashboard. The next
call replays those steps with no LLM. When a step later breaks and the LLM fallback
rescues it, the layer repairs that one step and saves the repair.

```
dashboard: agentic_task ─► worker runs it (LLM) ─► actions recorded in memory
                                                         │
                     learned automation ◄─ plan / compile┘   (locators verified on the live page)
                            │
                            ▼  PATCH /api/v1/update_recording
dashboard: deterministic steps ─► next call: no LLM
                            │
       a step breaks ─► LLM fallback rescues it ─► self-heal patches that step ─► synced
```

## Quick start

```bash
# terminal 1 — a worker that learns, plans with the LLM, self-heals and syncs
HEAL=1 REFINE=1 scripts/serve.sh

# terminal 2 — run a dashboard automation; inputs default to its examples
scripts/fire.sh <endpoint>
scripts/fire.sh <endpoint> first_name=Jane postal_code=110001
```

The first call of an endpoint whose automation is an `agentic_task` runs the agent and
syncs the learned steps. Every later call runs those steps. `scripts/serve.sh` looks
for `.env` in the repo or one level above it, and for the `optexity` CLI in the active
virtualenv, a `.venv` in or above the repo, or `PATH`.

| `serve.sh` flag | Effect |
|---|---|
| (none) | learn: a successful `agentic_task` becomes deterministic steps, synced |
| `HEAL=1` | self-heal: a step the LLM fallback rescued gets the verified locator it found, synced |
| `REFINE=1` | the LLM plans the learned automation (see [Planning](#planning)) |
| `LEARN=0` | plain worker, nothing learned |
| `LOCAL=x.json` | run `automations/x.json` instead of the dashboard copy; learned output is logged, never synced |
| `PORT=9001` | another port |

## How it works

### 1. Record — `browser_use/cache/` (browser-use fork)

A hook in `Tools.act`, the one point every browser-use action passes through,
snapshots the element before the action and records it afterwards
(`ActionTraceRecorder`). Capturing there, rather than from the agent's history, keeps
the full DOM node with its accessibility data, and the page is still in the state the
LLM saw, so locators can be checked against it.

For each action it derives ranked locator candidates — test id, id, name, aria-label,
exact role and name, placeholder, CSS class, exact text, a stable-id ancestor, xpath —
and **counts how many elements each one matches on the live page**, piercing open
shadow roots and inside the element's own frame. Only a candidate matching exactly one
element is ever used. Role and text locators are emitted with `exact=True`, so
"Uttar Pradesh" cannot also match "Uttar Pradesh East"; where the count approximates
Playwright's matching it errs high, so it can cost a usable locator but never make an
ambiguous one look unique.

Each action is tagged with the automation node that caused it
(`set_trace_context(node=…)`), so one node can be repaired without recompiling the
rest. Recording is in memory (`BROWSER_USE_ACTION_TRACE_MEMORY=1`) or to a JSONL file
(`BROWSER_USE_ACTION_TRACE_PATH`), and off unless one is set.

### 2. Learn — `optexity/cache/learning.py`

Runs in the worker that ran the task, after a **successful** run, from the in-memory
trace: nothing is written to disk. A failed run is not learned from; its wrong turns
would make the next run fail faster.

- **An `agentic_task`** is replaced by deterministic nodes — planned by the LLM with
  `REFINE=1`, otherwise compiled by rules (`compiler.py`).
- **A step the LLM fallback rescued** (with `HEAL=1`) gets the verified locator of the
  element the LLM chose. Every other node is left alone, so a run that needed no LLM
  changes nothing.

The result is validated as an `Automation` and PATCHed to the recording.

### Planning

`optexity/cache/planner.py`. The LLM decides what the automation should be — which
actions are real steps and which were exploration, which values a caller will want to
vary, how each step is described. **It never writes a locator**: its plan refers to
recorded actions and their verified locators by index. The builder then checks
everything the plan claims and rejects it, with the reasons, if:

- a recorded action is neither a step nor dropped with a reason, or steps are out of order;
- a step uses a locator that was not verified unique for that action;
- an input's value is not what the agent actually typed, picked or clicked;
- a value still sits inside a locator that nothing templated (a calendar cell
  `Choose Friday, January 1st, 1999` keeps picking that date).

A rejected plan goes back with the problems, up to three attempts; after that the
rule-based compiler takes over, with one refinement pass (`refine.py`) on its output.
When the plan leaves a parameter-bound click to the LLM although a verified locator
names its value exactly, the builder takes that locator.

### Inputs

Inputs are created from the recording, never from the model: a typed value, a picked
option, or a clicked element's own label. A clicked choice keeps a command only if a
verified exact locator names it, which is then templated —
`get_by_role("radio", name="{gender[0]}", exact=True)`; otherwise that step uses the
LLM on every run and the report says so.

A new input changes the endpoint's contract: once synced, the dashboard requires it
from every caller. The learning log prints the call to make, and `scripts/fire.sh`
reads the endpoint's current inputs on every call.

### Rule-based compiler

`optexity/cache/compiler.py` (`optexity compile-cache`): drops non-mutating actions
(`done`, `scroll`, `screenshot`, …), failed actions, clicks browser-use refuses (a
`<select>`, a file input), a click immediately before typing into the same field, and
inputs overwritten later; emits a node per remaining action with its best verified
locator; templates a typed value that equals a declared input. A step with no verified
locator keeps only its `prompt_instructions`, Optexity's own LLM-fallback shape.

## Configuration

| Variable | Where read | Effect |
|---|---|---|
| `OPTEXITY_LEARN` | worker | learn from successful agentic runs |
| `OPTEXITY_SELF_HEAL` | worker | repair steps the LLM fallback rescued |
| `OPTEXITY_LEARN_REFINE` | worker | plan with the LLM |
| `OPTEXITY_LOCAL_AUTOMATION` | worker | dev: run a local automation file instead of the dashboard's |
| `OPTEXITY_DUMP_AUTOMATION` | worker | dev: save the automation the dashboard sent |
| `BROWSER_USE_ACTION_TRACE_MEMORY` / `_PATH` | browser-use | record actions in memory / to a JSONL file |

## CLI

```bash
optexity compile-cache --trace run.jsonl --out automation.json \
    --input-parameters '{"first_name": ["Aditi"]}' [--llm --objective "the task"]
optexity learn --automation source.json --trace run.jsonl --out learned.json [--llm]
```

To capture a trace file from a run: start the worker with
`OPTEXITY_LEARN=0 BROWSER_USE_ACTION_TRACE_PATH=/path/run.jsonl`.

## Results

Node execution time; browser start-up and uploads are excluded, since caching cannot
change them.

**roboform** (`automations/test_automation*.json`) — agentic 34.6 s and 15 LLM calls;
cached 4.0 s (re-measured 3.8 s) and 0. Three runs chose three different DOM index sets
and compiled to the same four steps.

**demoqa practice form** (react-select dropdowns, date picker, radios) — learning run
59.5 s; replay 9.4 s, 0 LLM. A locator broken on purpose: one LLM fallback, healed,
next run 10.0 s, 0 LLM. The planned automation replayed with values never recorded
(`Jane`, `Male`, `Agra`, `15 Mar 1995`): 13/13 commands on try 1, 0 LLM.

**SauceDemo checkout** (multi-step: login → products → cart → checkout form →
overview → done; `automations/saucedemo_*.json`). The task was a goal only — *"Buy
the Sauce Labs Backpack on this store as the standard test user. Ship it to Aditi
Dubey, postal code 560001"* — with no steps and no inputs declared.

| Run | Result |
|---|---|
| Learn | agent 7 steps, 43.5 s; planner 1 attempt, ~5.7k tokens, 11 steps, all `data-test` locators, 0 left to the LLM; inputs `username`, `password`, `first_name`, `last_name`, `postal_code` created |
| Replay with `Jane` / `Doe` / `110001` | 11/11 on try 1, 0 LLM, ~4.7 s of step time across 5 pages |
| Checkout locator broken on the dashboard | one LLM fallback; steps 6–10 on try 1 on the following pages; healed and synced |

## Fixes to existing code

Found while building this; each was reproduced first.

| Problem | Fix |
|---|---|
| An agent without vision is offered a screenshot tool it cannot use, and loops on it until `max_steps` — browser-use only removes the tool for the `Tools` it builds itself | `handle_agentic_task.py`: exclude `screenshot` when `use_vision` is off |
| An agent that gives up or runs out of steps is reported as a successful task | `run_interaction.py`: fail the node unless the agent finished and reported success |
| browser-use's judge adds an LLM call per run and fails on Anthropic models (JPEG screenshot labelled PNG) | `use_judge=False` |
| Claude's agent output sometimes misplaces fields around otherwise correct actions; Anthropic does not enforce a non-strict tool schema | `chat_litellm.py`: report why parsing failed; repair the structure, never the actions |
| browser-use key names (`Enter`) are rejected by `KeyPressAction` (`enter`) | the compiler maps them |
| After a login, Chrome's password dialog freezes the page for Playwright and CDP alike, in headed runs only; the flags meant to stop it are not passed on the default Playwright launch path | `actual_browser.py`: password manager and leak detection off in the profile preferences |

The recorder also differs from the existing locator heuristic in three places: a
leading-digit name like `04fullname` is not treated as generated; `react-` is generated
only when a counter follows (`react-select-3-input`, not
`react-datepicker__year-select`); and browser-internal roles (`LabelText`) are not used
with `get_by_role`.

## Limits

- **Extraction is not learned.** An agent's `extract` actions are dropped; a task whose
  output is read from the page needs an `extraction_action` added by hand.
- **Inputs are plain `input_parameters`.** A learned password is stored as an ordinary
  input; it should be a `secure_parameter`.
- **A date picked from a calendar** keeps its day step on the LLM, because the day
  cell's label carries the weekday. A typed date is fully deterministic.
- **Accessible names that are a field's value** (`get_by_role("textbox", name="01 Oct
  2026")`) are verified unique and would be chosen for a field with nothing better.
- **A broken locator costs `max_tries` × `max_timeout_seconds_per_try`** (about 11 s by
  default) before the fallback starts — once, since the heal removes it.
- `get_args()` passes three separate `--disable-features` switches; Chromium keeps only
  the last value of a repeated switch, so they should be merged into one.
