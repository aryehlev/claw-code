# Router sidecars

`docker-compose.yml` brings up the two sidecars that back claw's `router`
config block:

| Service | Port | Purpose |
|---|---|---|
| `gptcache` | 8100 | Semantic cache in front of the router. Hashes/embeds prompts, serves hits locally, forwards misses downstream. |
| `routellm` | 6060 | Trained prompt classifier that picks between a "strong" and "weak" model per request. Ships with an eval harness on MT-Bench / MMLU / GSM8K. |

Request flow:

```
claw --> gptcache:8100 --> routellm:6060 --> Anthropic / OpenAI / ...
```

## Bring up

```sh
export ANTHROPIC_API_KEY=sk-ant-...
docker compose -f deploy/router/docker-compose.yml up -d
```

Then enable routing in `.claw.json`:

```json
{
  "router": {
    "enabled": true,
    "baseUrl": "http://127.0.0.1:8100/v1",
    "apiKey": "sk-router",
    "model": "router-mf-0.11593"
  }
}
```

`router-mf-0.11593` is a RouteLLM model specifier — `mf` (matrix
factorization) is the router, `0.11593` is the strong-model threshold.
Lower = more requests to the strong model. Tune via eval.

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
