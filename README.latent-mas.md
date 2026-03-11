# LatentMAS for OpenClaw

This branch adds a minimal local-only LatentMAS execution path to OpenClaw.

The goal is to reduce text-token overhead during multi-step reasoning by moving
the internal collaboration phase from visible text into model latent states.

## Status

This is an experimental first pass, not a production-ready integration.

What works now:

- Model-level `latentMas` config in OpenClaw model compat settings
- Agent dispatch from the normal OpenClaw run path into a latent runner
- Local Python bridge that runs a 4-agent latent workflow:
  - Planner
  - Critic
  - Refiner
  - Judger
- Final answer delivery back through OpenClaw's normal payload/session flow
- Best-effort reuse of recent conversation context from the current session transcript

What does not work yet:

- Image inputs
- Client tools / tool calling
- Long-lived model daemon reuse
- Native vLLM hidden-state sidecar integration
- Deep integration with OpenClaw subagent spawning

## High-Level Design

The branch does not replace OpenClaw's general provider stack.

Instead, it introduces a parallel execution path for models marked as
LatentMAS-enabled:

1. OpenClaw resolves the selected model as usual.
2. If the model has `compat.latentMas.enabled: true`, or the provider is
   `latentmas`, the request is routed to a local latent runner.
3. The latent runner gathers the current prompt and recent session history.
4. A Python bridge loads a local Hugging Face causal LM and runs latent
   collaboration across four internal agents.
5. Only the final Judger response is returned to OpenClaw.
6. OpenClaw persists the user turn and final assistant turn to the transcript.

This keeps OpenClaw's channel/session/result plumbing intact while isolating the
latent reasoning experiment to a single execution branch.

## Files

Core feature files:

- `src/latent-mas/config.ts`
- `src/latent-mas/run.ts`
- `src/latent-mas/bridge.ts`
- `assets/latent-mas/bridge.py`

Integration points:

- `src/commands/agent.ts`
- `src/config/types.models.ts`
- `src/config/zod-schema.core.ts`

## Configuration

Add a model provider entry with `compat.latentMas`.

Example:

```yaml
models:
  providers:
    latentmas:
      baseUrl: "http://127.0.0.1/latent"
      api: "openai-responses"
      models:
        - id: "Qwen/Qwen3-4B"
          name: "Qwen3 4B Latent"
          reasoning: true
          input: ["text"]
          cost:
            input: 0
            output: 0
            cacheRead: 0
            cacheWrite: 0
          contextWindow: 32768
          maxTokens: 4096
          compat:
            latentMas:
              enabled: true
              latentSteps: 12
              prompt: "sequential"
              device: "cuda"
              torchDtype: "bfloat16"
              maxNewTokens: 1024
              temperature: 0.6
              topP: 0.95
              historyTurns: 6
```

Notes:

- `baseUrl` is currently only a schema placeholder. The execution path is local
  Python/Hugging Face, not HTTP.
- If `modelName` is omitted, the model `id` is used as the Hugging Face model name.
- `provider: latentmas` enables the branch automatically, but any provider can
  also opt in with `compat.latentMas.enabled: true`.

## Runtime Requirements

The latent runner currently requires:

- A local Python interpreter (`python3` or `python`)
- `torch`
- `transformers`
- A local causal LM that supports:
  - `use_cache=True`
  - hidden state output
  - input embeddings

Recommended first target:

- Qwen-family causal models on a local GPU

## Request Constraints

The current implementation is text-only.

If a dedicated `latentmas/*` model receives:

- image input, or
- client tools

the run will fail fast with an explicit error instead of silently falling back.

For non-dedicated providers that opt into `compat.latentMas`, unsupported
request types continue through the normal OpenClaw execution path.

## Current Tradeoffs

This branch is intentionally conservative.

Pros:

- Minimal blast radius
- Keeps OpenClaw's normal agent/session delivery flow
- Clean experiment boundary

Cons:

- Model load cost is currently per-call
- No tool use inside latent collaboration
- No full parity with the default OpenClaw agent runtime

## Suggested Next Steps

The next practical milestones are:

1. Replace the per-call Python process with a long-lived local latent service.
2. Add a native vLLM/HF sidecar API for hidden-state and KV-cache operations.
3. Reintroduce tool use after the latent planning phase.
4. Connect the latent branch to OpenClaw's subagent/session-spawn workflow.

## Validation Notes

In the current workspace, the feature was added and a Python AST parse check was
performed for `assets/latent-mas/bridge.py`.

Full TypeScript/Vitest validation was not run here because the workspace did not
include `node_modules` and no global `tsc` was available.
