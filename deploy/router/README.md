# Router

claw supports two routing modes, selected via `router.mode` in `.claw.json`.

## Mode A — `external` (sidecar proxy)

Requests flow through a sidecar that both caches and routes:

```
claw --> gptcache:8100 --> routellm:6060 --> Anthropic / OpenAI / ...
```

| Service | Port | Purpose |
|---|---|---|
| `gptcache` | 8100 | Semantic cache in front of the router. Hashes/embeds prompts, serves hits locally, forwards misses downstream. |
| `routellm` | 6060 | Pre-trained classifier that picks between a "strong" and "weak" model per request. Ships with an MT-Bench / MMLU / GSM8K eval harness. |

### Bring up

```sh
export ANTHROPIC_API_KEY=sk-ant-...
docker compose -f deploy/router/docker-compose.yml up -d
```

Enable in `.claw.json`:

```json
{
  "router": {
    "enabled": true,
    "mode": "external",
    "baseUrl": "http://127.0.0.1:8100/v1",
    "apiKey": "sk-router",
    "model": "router-mf-0.11593"
  }
}
```

`router-mf-0.11593` is a RouteLLM model specifier — `mf` (matrix
factorization) is the router, `0.11593` is the strong-model threshold.
Lower = more requests to the strong model.

## Mode B — `eval-driven` (in-process, no sidecar required)

claw picks the upstream model itself per turn using a scoreboard of
prior outcomes. No external routing classifier, no opaque decisions —
the router's choice is explainable (`warm_start` / `explore` /
`exploit`) and the scoreboard is a plain JSON file you can inspect.

Request flow:

```
claw (picks model from scoreboard) --> Anthropic / OpenAI / ...
```

Enable in `.claw.json`:

```json
{
  "router": {
    "enabled": true,
    "mode": "eval-driven",
    "candidates": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"],
    "epsilonPercent": 10,
    "minSamples": 5,
    "scoreboardPath": "~/.local/share/claw/router-scoreboard.json"
  }
}
```

### How selection works

For each turn, the router computes a coarse prompt bucket by length
(`short` < 512 chars, `medium` < 6000 chars, `long` ≥ 6000) and picks
one candidate using:

1. **Warm-start** — any candidate with fewer than `minSamples`
   observations in the current bucket gets picked so the scoreboard
   accumulates baseline data.
2. **Explore** — with probability `epsilonPercent` / 100 pick a random
   candidate, to keep adapting when the provider mix shifts.
3. **Exploit** — otherwise pick the candidate with the highest
   Laplace-smoothed success rate; tie-breaker is lower average latency.

Each turn logs one line to stderr:

```
[router] bucket=short selected=claude-haiku-4-5 reason=exploit
```

Outcome (success = stream completed without error, plus latency and
token counts) is recorded back to the scoreboard and persisted
atomically (write-then-rename) after every turn.

### Inspecting the scoreboard

```sh
jq . ~/.local/share/claw/router-scoreboard.json
```

Structure: `{bucket: {model: {successes, failures, total_latency_ms, total_input_tokens, total_output_tokens}}}`.

### Seeding from session history

Use `claw eval` to replay historical sessions; each turn's result is
recorded into the scoreboard so the greedy phase has data to rank on.
(See `claw eval --help`.)

### Pairing with a cache

Eval-driven mode still works with the GPTCache sidecar — just add
`"baseUrl": "http://127.0.0.1:8100/v1"` and `"apiKey": "sk-router"` to
the `router` block. claw's selected model flows through GPTCache on
the way to the provider, so you keep semantic caching without the
RouteLLM classifier. RouteLLM becomes optional.

## Evaluation

### Option A — replay your own session history with `claw eval`

Point the router at a saved session and replay every user turn through
the proxy. claw emits one JSONL record per turn and a summary line on
stderr. This measures routing behavior against *your* prompts rather
than a public benchmark.

```sh
claw eval --session latest --output eval.jsonl
# stderr: [eval] turns=42 ok=42 failed=0 avg_latency_ms=318 tokens_in=... \
#         models[claude-haiku-4-5=31,claude-opus-4-6=11]
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--session <id\|latest>` | `latest` | Session to replay (same references as `claw --resume`). |
| `--output <path>` | stdout | Write JSONL records to `<path>` instead of stdout. |
| `--max-turns <n>` | all turns | Cap the number of user turns replayed. |
| `--output-format {text,json}` | `text` | When `json`, also print a pretty summary object on stdout. |

Each JSONL record:

```json
{"turn_index": 3, "user_input": "...", "requested_model": "router-mf-0.11593",
 "routed_model": "claude-haiku-4-5", "latency_ms": 284,
 "input_tokens": 1840, "output_tokens": 92, "ok": true}
```

Requires `router.enabled: true` in `.claw.json` — `claw eval` always
goes through the configured proxy and never touches upstream providers
directly.

### Option B — RouteLLM's own benchmark harness

RouteLLM ships MT-Bench / MMLU / GSM8K scripts inside the container:

```sh
docker exec -it claw-routellm \
  python -m routellm.evals.evaluate \
    --routers mf \
    --strong-model claude-opus-4-6 \
    --weak-model claude-haiku-4-5 \
    --benchmark mt-bench
```

Results land under the `routellm-results` volume.

## Observing router decisions

When routing is enabled claw prints one line per turn on stderr:

```
[router] user_model=opus routed_to=claude-haiku-4-5
```

Pipe that to a log aggregator to build cost-vs-quality curves against
your own session corpus.
