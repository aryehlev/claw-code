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

RouteLLM publishes its own eval harness; run it inside the container:

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
