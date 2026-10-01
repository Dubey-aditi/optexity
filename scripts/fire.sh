#!/usr/bin/env bash
# Run a dashboard automation on the local worker started by scripts/serve.sh.
#
#   scripts/fire.sh <endpoint>                        every input at the example
#                                                     value the automation declares
#   scripts/fire.sh <endpoint> city=Boston state=MA   overriding some of them
#   PORT=9001 scripts/fire.sh <endpoint>              a worker on another port
#
# The inputs are read from the dashboard on every call, so they follow the
# automation as learning changes it — a refined one may declare new inputs.
# Logs appear in the serve.sh terminal; this only queues the task.
set -euo pipefail

SCRIPT="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SCRIPT")/.." && pwd)"
[[ $# -ge 1 ]] || { sed -n '2,11p' "$SCRIPT" | sed 's/^# \{0,1\}//'; exit 1; }

if [[ -z "${ENV_PATH:-}" ]]; then
    for candidate in "$REPO/.env" "$REPO/../.env"; do
        [[ -f "$candidate" ]] && { ENV_PATH="$candidate"; break; }
    done
fi
[[ -n "${ENV_PATH:-}" ]] || { echo "no .env in $REPO or its parent; set ENV_PATH" >&2; exit 1; }

exec python3 - "$ENV_PATH" "${PORT:-9000}" "$@" <<'PY'
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

env_path, port, endpoint, *overrides = sys.argv[1:]

settings = {}
for line in open(env_path):
    key, sep, value = line.strip().partition("=")
    if sep and not key.startswith("#"):
        settings[key.strip()] = value.strip().strip('"').strip("'")
if not settings.get("OPTEXITY_API_KEY"):
    sys.exit(f"OPTEXITY_API_KEY is not set in {env_path}")


def request(url: str, payload: dict | None = None, headers: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"{url}: HTTP {error.code}: {error.read().decode()[:300]}")
    except urllib.error.URLError as error:
        sys.exit(f"{url}: {error.reason} (is scripts/serve.sh running?)")


server = settings.get("SERVER_URL", "https://api.optexity.com").rstrip("/")
query = urllib.parse.urlencode({"endpoint_name": endpoint})
recordings = request(
    f"{server}/api/v1/get_recordings?{query}",
    headers={"x-api-key": settings["OPTEXITY_API_KEY"]},
)
# The endpoint_name filter is a substring match; take only the exact name.
match = [r for r in recordings if r.get("endpoint_name") == endpoint]
if not match:
    sys.exit(f"no automation named {endpoint!r} on the dashboard")
inputs = dict(match[0]["automation"]["parameters"].get("input_parameters") or {})

for override in overrides:
    name, sep, value = override.partition("=")
    if not sep:
        sys.exit(f"expected name=value, got {override!r}")
    if name not in inputs:
        sys.exit(f"{endpoint} has no input {name!r}; it takes: {', '.join(inputs) or 'none'}")
    inputs[name] = [value]

print(f"▶ {endpoint}  inputs {json.dumps(inputs)}")
reply = request(
    f"http://localhost:{port}/inference",
    {"endpoint_name": endpoint, "input_parameters": inputs, "unique_parameter_names": []},
)
print(f"  queued task {reply.get('task_id')}" if reply.get("success") else f"  {reply}")
PY
