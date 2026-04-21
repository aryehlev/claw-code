# Efficiency integrations for claw

Third-party tools that layer on top of claw to cut token usage and
wall-clock cost. Everything here is **external** — claw calls out to a
separately-installed binary or skill via its existing hook / skill
mechanisms. No Rust changes in claw itself; each integration lands and
unlands by editing `.claw.json`.

| Tool | What it compresses | Install |
|---|---|---|
| [RTK](https://github.com/rtk-ai/rtk) | Tool-call output (`git`, `cargo`, `npm`, `docker`, …). 60-90% on common dev commands | `rtk/` — PreToolUse hook wrapper |
| [Caveman](https://github.com/juliusbrussee/caveman) | Model response text via terse-style rewrites. ~65% on verbose turns | `caveman/` — user-level skill copy |

Both are opt-in per-workspace. Mix and match; they don't conflict.

## How they fit with what's already in claw

claw already ships:

- **Eval-driven router** (`.claw.json` → `router.mode: eval-driven`) — picks a cheaper candidate model per turn using a success-rate scoreboard.
- **Memory** (`.claw.json` → `memory`) — Zep-backed recall of prior facts, so the model doesn't need to re-see them every turn.
- **Prompt cache** — Anthropic's native prompt caching on the Anthropic path.
- **External cache/router sidecars** (`deploy/router/`) — GPTCache + RouteLLM for semantic cache + pre-trained routing.

Routing/cache cut **which requests run** and **what goes in**. RTK and
Caveman cut **how big each request and response is**. Stacking all four
is where the real savings come from:

```
  user turn
     │
     ▼
  Caveman (per-turn) ──► shorter model output, shorter user text
     │
     ▼
  Router (scoreboard) ─► cheapest capable model
     │
     ▼
  Cache (sidecar / prompt-cache)
     │
     ▼
  Provider
     │
     ▼
  Tool call? ─► RTK PreToolUse hook ─► compressed output back to model
     │
     ▼
  Memory ingest (Zep) ──► persisted facts for future recall
```

## Per-integration

- [`rtk/`](./rtk/) — drop-in `PreToolUse` hook wrapping `bash` tool calls.
- [`caveman/`](./caveman/) — manual skill copy (claw has no `SessionStart` event, so upstream's auto-activate doesn't apply — invoke via `/caveman`).

## License notes

Both tools have their own licenses (see upstream repos). The wrappers
here (`rtk/hook.sh`, the install instructions) are part of claw's
repository.
