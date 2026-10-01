#!/usr/bin/env bash
# Start a local Optexity worker that learns from every run and saves what it learns
# back to the dashboard. Automations always come from the dashboard; nothing is
# written to disk. See docs/memory-layer.md.
#
#   scripts/serve.sh               learn: an agentic_task that succeeds is compiled
#                                  into deterministic steps and synced to its recording
#   HEAL=1 scripts/serve.sh        also self-heal: a step whose locator broke, and that
#                                  the LLM fallback rescued, gets the new locator synced
#   REFINE=1 scripts/serve.sh      an LLM plans the learned automation from the
#                                  recorded run: which steps to keep, which values
#                                  become inputs, how each step reads. Locators still
#                                  come only from the recording; a plan that does not
#                                  check out falls back to the rule-based compiler
#   LEARN=0 scripts/serve.sh       a plain worker: run automations, learn nothing
#   LOCAL=x.json scripts/serve.sh  run automations/x.json instead of the dashboard's
#                                  copy (the endpoint you call must take the same
#                                  inputs). What it learns is logged, never synced
#   PORT=9001 scripts/serve.sh     listen on another port (default 9000)
#
# Flags combine: HEAL=1 REFINE=1 scripts/serve.sh
#
# Finds the .env in the repo or one directory above it (override with ENV_PATH), and
# the optexity CLI in the active virtualenv, a .venv in or above the repo, or PATH.
set -euo pipefail

SCRIPT="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SCRIPT")/.." && pwd)"
PORT="${PORT:-9000}"

# The one variable that cannot live in .env: it tells the settings where .env is.
if [[ -z "${ENV_PATH:-}" ]]; then
    for candidate in "$REPO/.env" "$REPO/../.env"; do
        if [[ -f "$candidate" ]]; then
            ENV_PATH="$(cd "$(dirname "$candidate")" && pwd)/.env"
            break
        fi
    done
fi
[[ -n "${ENV_PATH:-}" ]] || { echo "no .env in $REPO or its parent; set ENV_PATH" >&2; exit 1; }
export ENV_PATH

OPTEXITY_BIN="optexity"
for candidate in "${VIRTUAL_ENV:-/nonexistent}/bin/optexity" "$REPO/.venv/bin/optexity" "$REPO/../.venv/bin/optexity"; do
    if [[ -x "$candidate" ]]; then
        OPTEXITY_BIN="$candidate"
        break
    fi
done

export OPTEXITY_LEARN="${LEARN:-1}"
export OPTEXITY_SELF_HEAL="${HEAL:-0}"
export OPTEXITY_LEARN_REFINE="${REFINE:-0}"

# Clear the dev-only overrides so every run uses, and syncs to, the dashboard copy.
# Empty rather than unset: python-dotenv refills an unset variable from .env.
export OPTEXITY_LOCAL_AUTOMATION="" OPTEXITY_DUMP_AUTOMATION="" BROWSER_USE_ACTION_TRACE_PATH=""
if [[ -n "${LOCAL:-}" ]]; then
    OPTEXITY_LOCAL_AUTOMATION="$REPO/automations/$LOCAL"
    [[ -f "$OPTEXITY_LOCAL_AUTOMATION" ]] || { echo "no such automation: $OPTEXITY_LOCAL_AUTOMATION" >&2; exit 1; }
    export OPTEXITY_LOCAL_AUTOMATION
fi

if pid="$(lsof -ti "tcp:$PORT" 2>/dev/null)" && [[ -n "$pid" ]]; then
    echo "port $PORT is already in use (pid $pid). Stop it with:  kill $pid" >&2
    exit 1
fi

on() { [[ "$1" == "1" ]] && echo "on" || echo "off"; }
echo "▶ Optexity worker on :$PORT"
echo "  learn       $(on "$OPTEXITY_LEARN")"
echo "  self-heal   $(on "$OPTEXITY_SELF_HEAL")"
echo "  llm plan    $(on "$OPTEXITY_LEARN_REFINE")"
[[ -n "${LOCAL:-}" ]] && echo "  source      automations/$LOCAL (local: never synced)"

exec "$OPTEXITY_BIN" inference --port "$PORT" --child_process_id 0
