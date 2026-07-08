## Commands

- `bun run test` — run tests
- `bun run build` — ESM + CJS bundles
- `bun run lint` — full type check
- `bun run typecheck` — quick `--noEmit` check
- `bun run dev <example>` — run an example

## Structure

`src/index.ts` is the single entry point. Key files:

- `llm.ts` — `createLLM()` factory
- `structured.ts` — `llm.structured()` pipeline (parse → repair → Zod validate → self-heal)
- `generate.ts` — `llm.generate()` text generation
- `generate-shared.ts` / `generate-model-call.ts` / `generate-output.ts` — model-call pipeline shared by `structured()` and `generate()` (streaming snapshots, think-tag normalization, debug/observe)
- `prompt.ts` — `` prompt`...` `` tag + `prompt()` fluent builder
- `conversation.ts` / `image.ts` — multi-turn history and multimodal content helpers
- `schema-builder.ts` — `s.*` Zod helpers
- `parse.ts` / `extract.ts` — JSON extraction heuristics + validation
- `mcp.ts` — `createMCPClient()`
- `providers/` — `openai-compatible` (Responses API), `openai-compatible-legacy` (Chat Completions), and `anthropic-compatible` adapters; MCP runtime; SSE parsing; `registry.ts` for `createLLM` wiring
- `types.ts` — re-exports the shared types defined in `type-definitions/`

Tests in `tests/` (Bun native), examples in `examples/` (run via `bun run dev <name>`), output in `dist/`.

## API

- `createLLM({ provider, model, transport })` → client
- `llm.structured(schema, prompt, opts)` → `{ data, text, reasoning, attempts }`
- `llm.generate(prompt, opts)` → `{ text, reasoning, attempts }`
- `llm.embed(text)` → `{ embeddings: number[][] }`
- Streaming via `opts.stream.onData(event)` — `event.delta` (incremental) + `event.snapshot` (accumulated)
