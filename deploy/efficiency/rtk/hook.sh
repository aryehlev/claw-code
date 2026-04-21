#!/usr/bin/env bash
# claw PreToolUse hook that routes Bash calls through `rtk` for token-
# compressed output.
#
# Wiring:
#   1. install rtk:   cargo install --git https://github.com/rtk-ai/rtk rtk
#                     (or grab a release binary for your platform)
#   2. copy this file: cp deploy/efficiency/rtk/hook.sh ~/.claw/hooks/rtk.sh
#                      chmod +x ~/.claw/hooks/rtk.sh
#   3. point .claw.json at it — see settings.json.example next to this file.
#
# How it works:
#   claw runs PreToolUse hooks for EVERY tool call, not just Bash. This
#   wrapper short-circuits for non-Bash tools and only rewrites the
#   command when it matches a known rtk-supported tool family. All other
#   tool calls pass through untouched.
#
# Input:
#   HOOK_TOOL_NAME         — the tool claw is about to invoke ("bash" etc)
#   HOOK_TOOL_INPUT        — the raw tool input string (JSON)
#   stdin (JSON payload)   — same tool_input plus claw metadata
#
# Output (stdout, JSON):
#   { "hookSpecificOutput": { "updatedInput": { "command": "<rewritten>" } } }
#
# Anything else on stdout is ignored by claw. Non-zero exit = hook failed
# (tool call blocked), so we exit 0 on every non-transform path.

set -euo pipefail

# Tools claw calls "Bash" or "bash" depending on registry casing.
case "${HOOK_TOOL_NAME:-}" in
    bash|Bash) ;;
    *) exit 0 ;;
esac

# Skip if rtk isn't installed; pass through untouched rather than
# breaking the tool call.
if ! command -v rtk >/dev/null 2>&1; then
    exit 0
fi

# Pull the command out of the tool input JSON. Using python3 because it's
# almost always present; fall back to jq if you've got it.
command_in="$(
    python3 - <<'PY' "$HOOK_TOOL_INPUT"
import json, sys
try:
    payload = json.loads(sys.argv[1])
except Exception:
    print("", end="")
    sys.exit(0)
print(payload.get("command", ""), end="")
PY
)"

if [ -z "$command_in" ]; then
    exit 0
fi

# Peek at the first token to decide if rtk knows the tool. rtk's
# supported tool families (v0.28+):
#   git gh gt cargo go golangci-lint npm pnpm npx ruff pytest pip mypy
#   rspec rubocop rake dotnet playwright vitest jest docker kubectl aws
first_word="${command_in%% *}"
first_word="${first_word##*/}"  # strip any leading path

case "$first_word" in
    git|gh|gt|cargo|go|golangci-lint|npm|pnpm|npx|ruff|pytest|pip|mypy|\
    rspec|rubocop|rake|dotnet|playwright|vitest|jest|docker|kubectl|aws)
        rewritten="rtk $command_in"
        ;;
    *)
        exit 0
        ;;
esac

# Emit the replacement. claw merges this into the tool input before it
# runs the command, so the model sees rtk-compressed output.
python3 - <<PY
import json
print(json.dumps({
    "hookSpecificOutput": {
        "updatedInput": {"command": ${rewritten@Q}}
    }
}))
PY
