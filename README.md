# extrait

High-level LLM text generation and structured JSON extraction with validation, repair, and streaming.

<p align="left">
  <a href="https://www.npmjs.com/package/extrait">
    <img src="https://img.shields.io/npm/v/extrait?logo=npm&style=plastic" alt="npm package">
  </a>
</p>

## Features

- Multi-candidate JSON extraction from LLM responses
- Automatic repair with jsonrepair
- Zod schema validation and coercion
- Optional self-healing for validation failures
- Streaming support
- MCP tools
- Vector embeddings (OpenAI-compatible + Voyage AI)
- Multimodal images from paths, URLs, bytes or blobs

## Installation

Install `extrait` with your preferred package manager.

```bash
bun add extrait
# or
npm install extrait
# or
deno add npm:extrait
```

## Quick Start

`openai-compatible` targets the Responses API. Use `openai-compatible-legacy`
for servers that only expose Chat Completions.

```typescript
import { createLLM, prompt, s } from "extrait";
import { z } from "zod";

const llm = createLLM({
  provider: "openai-compatible",
  model: "mistralai/ministral-3-3b",
  baseURL: "http://localhost:1234/v1",
  apiKey: process.env.LLM_API_KEY ?? "local-demo-key",
});

const text = "Beat two eggs, add flour and milk, then cook thin pancakes in a hot pan.";

const RecipeSchema = s.schema(
  "Recipe",
  z.object({
    title: s.string().min(1).describe("Short recipe title"),
    ingredients: s.array(s.string()).min(1).describe("Ingredient list"),
  })
);

const result = await llm.structured(
  RecipeSchema,
  prompt`Extract a simple recipe from this text: """${text}"""`
);

console.log(result.data);
```

## Examples at a Glance

These examples cover the most common usage patterns in the repository.

- [`examples/simple.ts`](examples/simple.ts) - Basic structured output with streaming
- [`examples/generate.ts`](examples/generate.ts) - High-level text generation
- [`examples/logprobs.ts`](examples/logprobs.ts) - Token log probabilities and alternatives
- [`examples/streaming.ts`](examples/streaming.ts) - Real-time partial output and snapshot updates
- [`examples/calculator-tool.ts`](examples/calculator-tool.ts) - Structured extraction with MCP tools
- [`examples/streaming-turns-with-tools.ts`](examples/streaming-turns-with-tools.ts) - Streaming MCP turns, transitions, and reasoning blocks
- [`examples/conversation.ts`](examples/conversation.ts) - Multi-turn prompts and multimodal content
- [`examples/image-analysis.ts`](examples/image-analysis.ts) - Vision input with structured output
- [`examples/embeddings.ts`](examples/embeddings.ts) - Embeddings and similarity workflows

```bash
bun run dev simple "Bun.js runtime"
bun run dev generate "Bun.js runtime"
bun run dev streaming
bun run dev calculator-tool
```

## API Reference

The sections below cover the main building blocks of the library.

### Create an LLM Client

Use `createLLM()` to configure the provider, model, transport, and client defaults.

```typescript
const llm = createLLM({
  provider: "openai-compatible" | "openai-compatible-legacy" | "anthropic-compatible",
  model: "gpt-5-nano",
  baseURL: "http://localhost:1234/v1",     // required
  apiKey: process.env.LLM_API_KEY,         // optional
  transport: {                             // optional advanced settings
    path: "/v1/responses",                 // optional provider endpoint override
    embeddingPath: "/v1/embeddings",       // optional embedding endpoint (openai-compatible only)
    headers: { "x-trace-id": "docs-demo" }, // optional extra headers
    defaultBody: { user: "docs-demo" },    // optional provider body defaults
    version: "2023-06-01",                 // anthropic-compatible only
    defaultMaxTokens: 4096,                // anthropic-compatible only (default: 1024)
    defaultMaxToolRounds: 10,              // optional cap on MCP tool rounds (default: 100)
    fetcher: fetch,                        // optional custom fetch implementation
  },
  defaults: {
    mode: "loose" | "strict",             // loose allows repair
    selfHeal: 1,                          // optional retry attempts
    debug: false,                         // optional structured debug output
    // or:
    // debug: { enabled: true, verbose: true },
    systemPrompt: "You are a helpful assistant.",
    timeout: {
      request: 30_000,
      tool: 10_000,
    },
  },
});
```

The endpoint and credentials always live at the top level (`baseURL`, `apiKey`); `transport` is reserved for advanced settings such as endpoint paths, headers, body defaults, and a custom `fetcher`.

`baseURL` is **required** for the built-in providers. There is no implicit `https://api.openai.com` / `https://api.anthropic.com` fallback, so a client that was never pointed at an endpoint fails at construction instead of shipping your key to a vendor host. For request-specific options such as `stream`, `request`, `schemaInstruction`, and parse tuning, see the sections below.

Common setup patterns:

```typescript
// OpenAI-compatible gateway or local endpoint
const llm = createLLM({
  provider: "openai-compatible",
  model: "local-model",
  baseURL: process.env.LLM_BASE_URL ?? "http://localhost:1234/v1", // never left undefined
  apiKey: process.env.LLM_API_KEY ?? "local-demo-key",
});

// Chat Completions-only gateway
const legacy = createLLM({
  provider: "openai-compatible-legacy",
  model: "local-model",
  baseURL: "http://localhost:1234",
  apiKey: "local-demo-key",
});

// Anthropic-compatible endpoint with explicit API version
const anthropic = createLLM({
  provider: "anthropic-compatible",
  model: "claude-3-5-sonnet-latest",
  baseURL: "http://localhost:8080",
  apiKey: process.env.LLM_API_KEY,
  transport: {
    version: "2023-06-01",
  },
});
```

### Defining Schemas

Use the `s` wrapper around Zod for schema names, descriptions, and a more ergonomic authoring flow.

```typescript
import { s } from "extrait";
import { z } from "zod";

const Schema = s.schema(
  "SchemaName",
  z.object({
    // String fields
    text: s.string().min(1).describe("Field description"),
    optional: s.string().optional(),
    withDefault: s.string().default("value"),

    // Numbers
    count: s.number().int().min(0).max(100),
    score: s.number().min(0).max(1),

    // Arrays
    items: s.array(s.string()).min(1).max(10),

    // Nested objects
    nested: z.object({
      field: s.string(),
    }),

    // Enums (use native Zod)
    category: z.enum(["a", "b", "c"]),

    // Booleans
    flag: s.boolean(),
  })
);
```

### Making Structured Calls

`structured()` accepts a schema plus either a tagged prompt, a fluent prompt builder, or a raw message payload.

```typescript
// Simple prompt
const result = await llm.structured(
  Schema,
  prompt`Your prompt with ${variables}`
);

// Multi-part prompt
const result = await llm.structured(
  Schema,
  prompt()
    .system`You are an expert assistant.`
    .user`Analyze: """${input}"""`
);

// Multi-turn conversation
const conversationResult = await llm.structured(
  Schema,
  prompt()
    .system`You are an expert assistant.`
    .user`Hello`
    .assistant`Hi, how can I help?`
    .user`Analyze: """${input}"""`
);

// With options
const result = await llm.structured(
  Schema,
  prompt`Your prompt`,
  {
    mode: "loose",
    selfHeal: 1,
    debug: true,
    systemPrompt: "You are a helpful assistant.",
    stream: {
      to: "stdout",
      onData: (event) => {
        if (event.delta.text) {
          console.log("New visible text:", event.delta.text);
        }
        if (event.delta.reasoning) {
          console.log("New reasoning text:", event.delta.reasoning);
        }

        console.log("Current visible text:", event.snapshot.text);
        console.log("Current reasoning:", event.snapshot.reasoning);
        console.log("Current structured snapshot:", event.snapshot.data);

        if (event.done) {
          console.log("Streaming done.");
        }
      },
    },
    request: {
      signal: AbortSignal.timeout(30_000),  // optional AbortSignal
      reasoningEffort: "medium",            // optional reasoning effort hint
    },
    timeout: {
      request: 30_000,  // ms per LLM HTTP request
      tool: 10_000,     // ms per MCP tool call
    },
  }
);
```

`prompt()` builds an ordered `messages` payload. Use ``prompt`...` `` for a single string prompt, or the fluent builder for multi-turn conversations. The `LLMMessage` type is exported if you need to type your own message arrays.

In `stream.onData`, the event is split into two layers:

- `event.delta.text` is only the newly received visible text since the previous event.
- `event.delta.reasoning` is only the newly received reasoning text since the previous event.
- `event.snapshot.text` is the full visible text accumulated so far.
- `event.snapshot.reasoning` is the full normalized reasoning accumulated so far.
- `event.snapshot.data` is the best structured JSON snapshot that can be parsed from the stream so far. It may stay unchanged while `event.delta.text` continues to grow.

Typical usage is:

- render `event.delta.text` directly to a terminal or chat UI
- optionally render `event.delta.reasoning` in a separate reasoning panel
- use `event.snapshot.data` to drive partial structured UI state
- use `event.snapshot.text` / `event.snapshot.reasoning` when you need the full accumulated state instead of only the latest increment

You can also pass provider request options through `request`:

```typescript
const result = await llm.structured(
  Schema,
  prompt`Summarize this document: """${text}"""`,
  {
    request: {
      temperature: 0,
      maxTokens: 800,
      body: { user: "demo-user" },
    },
  }
);
```

### Making Text Calls

`generate()` is the high-level API for non-structured generation. It accepts the same prompt shapes as `structured()`, but does not inject any schema or parse the output.

```typescript
// Simple prompt
const result = await llm.generate(
  prompt`Write a short summary of ${topic}.`
);

// Multi-message prompt
const result = await llm.generate(
  prompt()
    .system`You are a concise assistant.`
    .user`Summarize: """${text}"""`
);

// Raw messages payload
const result = await llm.generate({
  prompt: {
    messages: [
      { role: "user", content: "Say hello in one sentence." },
    ],
  },
});
```

Streaming mirrors `structured()`, except the snapshot only contains `text` and `reasoning`:

```typescript
const result = await llm.generate(
  prompt`Explain ${topic} in one short paragraph.`,
  {
    stream: {
      enabled: true,
      onData: (event) => {
        process.stdout.write(event.delta.text);

        console.log("Full text so far:", event.snapshot.text);
        console.log("Full reasoning so far:", event.snapshot.reasoning);

        if (event.done) {
          console.log("Streaming done.");
        }
      },
    },
  }
);
```

Provider request options and MCP tools still go through `request`:

```typescript
const result = await llm.generate(
  prompt`Use tools if needed and answer the user clearly.`,
  {
    request: {
      temperature: 0,
      maxTokens: 800,
      reasoningEffort: "medium",
      mcpClients: [calculatorMCP],
      maxToolRounds: 10,
    },
  }
);
```

On `openai-compatible`, this is sent as `reasoning: { effort }`; on `openai-compatible-legacy`, as `reasoning_effort`. OpenAI effort values, including the distinct `xhigh` and `max` levels, are forwarded as-is. On `anthropic-compatible`, `minimal` maps to `low`, `none` disables thinking, and supported effort values are sent as `output_config.effort` while auto-enabling `thinking: { type: "adaptive" }`; the thinking content comes back in `result.reasoning`.

For existing history or multi-turn conversations, pass `messages` directly:

```typescript
const messages = conversation("You are a helpful assistant.", [
  { role: "user", text: "What is the speed of light?" },
  { role: "assistant", text: "Approximately 299,792 km/s in a vacuum." },
  { role: "user", text: "How long does light take to reach Earth from the Sun?" },
]);

const result = await llm.generate({ prompt: { messages } });
```

Use `llm.adapter.complete(...)` or `llm.adapter.stream(...)` only when you need the raw low-level provider interface.

### Token Logprobs

Set `request.topLogprobs` (0-20) to get token-level probabilities on `openai-compatible` and `openai-compatible-legacy`. The result exposes `logprobs.content` — one entry per generated token, each with its chosen token, logprob, and the requested number of alternatives.

```typescript
const result = await llm.generate(prompt`Answer with one word: yes or no?`, {
  request: { topLogprobs: 3 },
});

for (const token of result.logprobs?.content ?? []) {
  const probability = Math.exp(token.logprob);
  console.log(token.token, probability, token.top_logprobs);
}
```

On `openai-compatible` (Responses API) this sends `top_logprobs` and automatically adds the `message.output_text.logprobs` include; on `openai-compatible-legacy` it sends `logprobs: true` + `top_logprobs`. Streaming accumulates logprobs across chunks. Anthropic has no logprobs API, so `anthropic-compatible` never returns them.

Some Responses-dialect servers (e.g. llama.cpp) also require the Chat Completions-style flag; pass it through `transport.defaultBody: { logprobs: true }`. See [examples/logprobs.ts](examples/logprobs.ts).

### Images (multimodal)

Two functions build the content blocks that spread into a message:

| | Accepts | I/O |
|---|---|---|
| `images(...)` | data URL, `http(s)` URL, `URL`, `Uint8Array` / `ArrayBuffer` / `Buffer`, `{ base64, mimeType }` | none — synchronous |
| `await loadImages(...)` | everything above **plus** file paths, `file:` URLs and `Blob` / `File` | reads files and blobs |

Both take a single source or an array, and always return an `LLMImageContent[]`.

```typescript
import { loadImages, prompt } from "extrait";

const content = await loadImages("photo.png");

const result = await llm.structured(Schema,
  prompt()
    .system`You are a vision assistant.`
    .user([{ type: "text", text: "Describe this image." }, ...content])
);
```

```typescript
import { images, loadImages } from "extrait";

// No I/O needed — synchronous
images("https://example.com/photo.png");          // remote URL, passed through as-is
images("data:image/png;base64,iVBORw0KG...");     // data URL, passed through byte for byte
images(await sharp(path).resize(512).toBuffer()); // raw bytes, mime type sniffed
images({ base64, mimeType: "image/png" });        // explicit pair

// Needs I/O
await loadImages(["a.png", "b.jpg"]);             // file paths
await loadImages(new Blob([bytes]));              // Blob / File

// Mix freely — order is preserved
await loadImages(["local.png", "https://example.com/remote.jpg"]);
```

**How a `string` is resolved**, in order:

1. starts with `data:` → used as-is;
2. starts with `http://` or `https://` → forwarded to the provider as-is (`extrait` does not fetch it — some models reject remote URLs);
3. otherwise → treated as a **file path**, so it requires `loadImages()`. Passing one to `images()` throws an explicit error.

A bare base64 string is not accepted: it is indistinguishable from a path. Pass `{ base64, mimeType }`.

**Mime type detection**, in order: the explicit `mimeType` → magic bytes (PNG, JPEG, WebP, GIF, AVIF/HEIC, BMP) → file extension → error. `sniffMimeType(bytes)` is exported if you need it on its own.

Providers commonly cap images around 5 MB and 8000 px. Since nothing is resized for you, resize upstream when it matters:

```typescript
import sharp from "sharp";
import { images } from "extrait";

const buf = await sharp("photo.png")
  .resize(512, 512, { fit: "inside", withoutEnlargement: true })
  .toBuffer();

const content = images(buf);
```

See [examples/image-analysis.ts](examples/image-analysis.ts).

### Conversations (multi-turn history)

Use `conversation()` to build a `LLMMessage[]` from an existing conversation history. This is the idiomatic way to pass prior turns to the LLM.

```typescript
import { conversation } from "extrait";

const messages = conversation("You are a helpful assistant.", [
  { role: "user",      text: "What is the speed of light?" },
  { role: "assistant", text: "Approximately 299,792 km/s in a vacuum." },
  { role: "user",      text: "How long does light take to reach Earth from the Sun?" },
]);

// High-level text generation
const response = await llm.generate({ prompt: { messages } });

// Or to structured extraction
const result = await llm.structured(Schema, { messages });
```

Entries with `images` produce multimodal content automatically:

```typescript
const messages = conversation("You are a vision assistant.", [
  {
    role: "user",
    text: "What is in this image?",
    images: ["https://example.com/photo.png"], // any synchronous source
  },
]);
```

### Result Object

Successful `generate()` calls return normalized text/reasoning plus request metadata:

```typescript
{
  text: string,
  reasoning: string,
  attempts: GenerateAttempt[],
  usage?: {
    inputTokens?: number,
    outputTokens?: number,
    totalTokens?: number,
    cost?: number,
  },
  finishReason?: string,
  logprobs?: LLMLogprobs,          // when request.topLogprobs is set
  reasoningBlocks?: ReasoningBlock[], // per-turn reasoning for multi-round MCP streams
}
```

Each `attempts` entry includes:

```typescript
{
  attempt: number,
  via: "complete" | "stream",
  text: string,
  reasoning: string,
  usage?: LLMUsage,
  finishReason?: string,
}
```

Successful `structured()` calls return validated data plus normalized text/reasoning and trace metadata.

```typescript
{
  data: T,                      // Validated data matching schema
  text: string,                 // Visible model text, without inline <think> blocks
  reasoning: string,            // Normalized reasoning across dedicated fields and inline <think>
  json: unknown | null,         // Parsed JSON before validation
  attempts: StructuredAttempt<T>[], // One entry per parse / self-heal attempt
  usage?: {
    inputTokens?: number,
    outputTokens?: number,
    totalTokens?: number,
    cost?: number,
  },
  finishReason?: string,        // provider vocabulary: "completed" (Responses), "stop" (Chat Completions), "end_turn" (Anthropic)
  logprobs?: LLMLogprobs,          // when request.topLogprobs is set
  reasoningBlocks?: ReasoningBlock[], // per-turn reasoning for multi-round MCP streams
}
```

Each `attempts` entry includes:

```typescript
{
  attempt: number,
  selfHeal: boolean,
  via: "complete" | "stream",
  text: string,
  reasoning: string,
  json: unknown | null,
  candidates: string[],
  repairLog: string[],
  zodIssues: z.ZodIssue[],
  success: boolean,
  usage?: LLMUsage,
  finishReason?: string,
  parsed: ParseLLMOutputResult<T>,
}
```

Legacy inline `<think>...</think>` blocks are still supported, but the high-level `structured()` API now folds them into `reasoning` internally instead of exposing block metadata.

### Error Handling

Catch `StructuredParseError` when repair and validation still fail.

```typescript
import { StructuredParseError } from "extrait";

try {
  const result = await llm.structured(Schema, prompt`...`);
} catch (error) {
  if (error instanceof StructuredParseError) {
    console.error("Validation failed");
    console.error("Attempt:", error.attempt);
    console.error("Zod issues:", error.zodIssues);
    console.error("Repair log:", error.repairLog);
    console.error("Candidates:", error.candidates);
  }
}
```

### Embeddings

Generate vector embeddings using `llm.embed()`. It always returns `number[][]` — one vector per input string.

```typescript
// Create a dedicated embedder client (recommended)
const embedder = createLLM({
  provider: "openai-compatible",
  model: "text-embedding-3-small",
  baseURL: "http://localhost:1234/v1",
  apiKey: process.env.LLM_API_KEY,
});

// Single string
const { embeddings, model, usage } = await embedder.embed("Hello world");
const vector: number[] = embeddings[0];

// Multiple strings in one request
const { embeddings } = await embedder.embed(["text one", "text two", "text three"]);
// embeddings[0], embeddings[1], embeddings[2] — one vector each

// Optional: override model or request extra options per call
const { embeddings } = await embedder.embed("Hello", {
  model: "text-embedding-ada-002",
  dimensions: 512,              // supported by text-embedding-3-* models
  body: { user: "user-id" },    // pass-through to provider
  signal: AbortSignal.timeout(10_000), // optional cancellation / timeout
});
```

**Result shape:**

```typescript
{
  embeddings: number[][];  // one vector per input
  model: string;
  usage?: { inputTokens?: number; totalTokens?: number };
  raw?: unknown;           // full provider response
}
```

**Anthropic / Voyage AI**

Anthropic does not provide a native embedding API. Their recommended solution is [Voyage AI](https://api.voyageai.com), which uses the same OpenAI-compatible format:

```typescript
const embedder = createLLM({
  provider: "openai-compatible",
  model: "voyage-3",
  baseURL: "https://api.voyageai.com",
  apiKey: process.env.LLM_API_KEY,
});

const { embeddings } = await embedder.embed(["query", "document"]);
```

Requests default to `encoding_format: "float"`. If your endpoint rejects that value, override it per call via `body`, or drop it entirely with `transport.defaultBody: { encoding_format: undefined }`.

Calling `llm.embed()` on an `anthropic-compatible` adapter throws a descriptive error pointing to Voyage AI.

### MCP Tools

Attach MCP clients at request time to let the model call tools during structured generation.

```typescript
import { createMCPClient } from "extrait";

const mcpClient = await createMCPClient({
  id: "calculator",
  transport: {
    type: "stdio",
    command: "bun",
    args: ["run", "examples/calculator-mcp-server.ts"],
  },
});

const result = await llm.structured(
  Schema,
  prompt`Calculate 14 + 8`,
  {
    request: {
      mcpClients: [mcpClient],
      maxToolRounds: 5,       // default: 100 (also configurable via transport.defaultMaxToolRounds)
      toolDebug: {
        enabled: true,
        includeRequest: true,
        includeResult: true,
      },
      onToolExecution: (execution) => {
        console.log(execution.name, execution.durationMs);
      },
      // Optional: transform tool output before it is sent back to the LLM
      transformToolOutput: (output, execution) => {
        return { ...output, source: execution.name };
      },
      // Optional: transform tool arguments before the tool is called
      transformToolArguments: (args, call) => args,
      // Optional: transform the full MCP call payload, including _meta
      transformToolCallParams: (params, call) => ({
        ...params,
        _meta: {
          source: "extrait-docs",
          clientId: call.clientId,
        },
      }),
      // Optional: custom error message when an unknown tool is called
      unknownToolError: (toolName) => `Tool "${toolName}" is not available.`,
    },
  }
);

await mcpClient.close?.();
```

`transformToolArguments()` only receives the tool input object. `transformToolCallParams()` runs after it and receives the full `MCPCallToolParams` payload that will be sent to the MCP client:

```typescript
type MCPCallToolParams = {
  name: string;
  arguments?: Record<string, unknown>;
  _meta?: Record<string, unknown>;
};
```

MCP toolsets may change between tool rounds. Providers call `listTools()` before each round, so a client can expose additional tools after a previous `callTool()` result, or remove tools that are no longer available.

Use `transformToolCallParams()` when you need to attach MCP-specific metadata, override the final remote tool name, or otherwise change the full request passed to `client.callTool()`. This hook is exported as `LLMToolCallParamsTransformer`.

### Timeouts

Use `timeout` to set per-request and per-tool-call time limits without managing `AbortSignal` manually.

```typescript
const result = await llm.structured(Schema, prompt`...`, {
  timeout: {
    request: 30_000,  // abort the LLM HTTP request after 30s
    tool: 5_000,      // abort each MCP tool call after 5s
  },
});
```

Both fields are optional. `timeout.request` creates an `AbortSignal.timeout` internally; it is ignored if you also pass `request.signal` (your signal takes precedence). `timeout.tool` wraps each MCP client transparently.

You can also set defaults on the client:

```typescript
const llm = createLLM({
  provider: "openai-compatible",
  model: "gpt-5-nano",
  baseURL: process.env.LLM_BASE_URL ?? "http://localhost:1234/v1",
  apiKey: process.env.LLM_API_KEY,
  defaults: {
    timeout: { request: 60_000 },
  },
});
```

## Examples

Run repository examples with `bun run dev <example-name>`.

Available examples:
- `generate` - High-level text generation ([generate.ts](examples/generate.ts))
- `logprobs` - Token log probabilities and verbose debug output ([logprobs.ts](examples/logprobs.ts))
- `streaming` - Real LLM streaming + snapshot self-check ([streaming.ts](examples/streaming.ts))
- `streaming-with-tools` - Real text streaming with MCP tools + self-check ([streaming-with-tools.ts](examples/streaming-with-tools.ts))
- `streaming-turns-with-tools` - Streaming MCP turns, transitions, and reasoning blocks ([streaming-turns-with-tools.ts](examples/streaming-turns-with-tools.ts))
- `abort-signal` - Start a generation then cancel quickly with `AbortSignal` ([abort-signal.ts](examples/abort-signal.ts))
- `timeout` - Set per-request and per-tool timeouts via the `timeout` option ([timeout.ts](examples/timeout.ts))
- `simple` - Basic structured output with streaming ([simple.ts](examples/simple.ts))
- `sentiment-analysis` - Enum validation, strict mode ([sentiment-analysis.ts](examples/sentiment-analysis.ts))
- `data-extraction` - Complex nested schemas, self-healing ([data-extraction.ts](examples/data-extraction.ts))
- `multi-step-reasoning` - Chained structured calls ([multi-step-reasoning.ts](examples/multi-step-reasoning.ts))
- `calculator-tool` - MCP tool integration ([calculator-tool.ts](examples/calculator-tool.ts))
- `image-analysis` - Multimodal structured extraction from an image file ([image-analysis.ts](examples/image-analysis.ts))
- `conversation` - Multi-turn conversation history and inline image messages ([conversation.ts](examples/conversation.ts))
- `simulated-tools` - Inject fake tool calls/results into conversation context without real execution ([simulated-tools.ts](examples/simulated-tools.ts))
- `embeddings` - Vector embeddings, cosine similarity, and semantic comparison ([embeddings.ts](examples/embeddings.ts))

Pass arguments after the example name:
```bash
bun run dev generate "Why Bun is fast"
bun run dev logprobs "Answer with one word: yes or no?"
bun run dev streaming
bun run dev streaming-with-tools
bun run dev abort-signal 120 "JSON cancellation demo"
bun run dev timeout 5000
bun run dev simple "Bun.js runtime"
bun run dev sentiment-analysis "I love this product."
bun run dev multi-step-reasoning "Why is the sky blue?"
bun run dev embeddings "the cat sat on the mat" "a feline rested on the rug"
```

## Environment Variables

These environment variables are used across the examples and common client setups.

- `LLM_PROVIDER` - `openai-compatible`, `openai-compatible-legacy`, or `anthropic-compatible`
- `LLM_BASE_URL` - API endpoint (required, no default)
- `LLM_MODEL` - Model name (default: `gpt-5-nano`)
- `LLM_API_KEY` - API key for the provider
- `STRUCTURED_DEBUG=1` - Enable debug output
  By default, structured debug prints `text` (public visible output) and
  `reasoning` (normalized reasoning). `parseSource` (the internal source used by
  parsing and self-heal) is only printed when `debug.verbose` is enabled.

## Testing

Run the test suite with Bun.

```bash
bun run test
```
