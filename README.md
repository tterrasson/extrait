# extrait

Structured JSON extraction from LLMs with validation, repair, and streaming.

<p align="left">
  <a href="https://www.npmjs.com/package/extrait">
    <img src="https://img.shields.io/npm/v/extrait?logo=npm&style=plastic" alt="npm package">
  </a>
</p>

**Features:**
- Multi-candidate JSON extraction from LLM responses
- Automatic repair with jsonrepair
- Zod schema validation and coercion
- Optional self-healing for validation failures
- Streaming support
- MCP tools
- Vector embeddings (OpenAI-compatible + Voyage AI)

## Installation

```bash
bun add extrait
# or
npm install extrait
# or
deno add npm:extrait
```

## Quick Start

```typescript
import { createLLM, prompt, s } from "extrait";
import { z } from "zod";

const llm = createLLM({
  provider: "openai-compatible",
  model: "gpt-5-nano",
  transport: { apiKey: process.env.LLM_API_KEY },
});

const SummarySchema = s.schema(
  "Summary",
  z.object({
    summary: s.string().min(1).describe("One-sentence summary"),
    tags: s.array(s.string()).default([]).describe("Keywords"),
  })
);

const result = await llm.structured(
  SummarySchema,
  prompt`Summarize this: """${text}"""`
);

console.log(result.data);
```

## API Reference

### Creating an LLM Client

```typescript
const llm = createLLM({
  provider: "openai-compatible" | "anthropic-compatible",
  model: "gpt-5-nano",
  transport: {
    baseURL: "https://api.openai.com",   // optional
    apiKey: process.env.LLM_API_KEY,     // optional
  },
  defaults: {
    mode: "loose" | "strict",            // loose allows repair
    selfHeal: 0 | 1 | 2,                 // retry attempts
    debug: false,                        // show repair logs
    timeout: { request: 30_000 },        // optional default timeouts
  },
});
```

### Defining Schemas

Use the `s` wrapper around Zod for enhanced schema building:

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
        console.log("Partial data:", event.data);
        if (event.done) {
          console.log("Streaming done.");
        }
      },
    },
    request: {
      signal: abortController.signal,  // optional AbortSignal
    },
    timeout: {
      request: 30_000,  // ms per LLM HTTP request
      tool: 10_000,     // ms per MCP tool call
    },
  }
);
```

`prompt()` builds an ordered `messages` payload. Use ``prompt`...` `` for a single string prompt, or the fluent builder for multi-turn conversations. The `LLMMessage` type is exported if you need to type your own message arrays.

### Images (multimodal)

Use `images()` to build base64 image content blocks for vision-capable models.

```typescript
import { images, prompt } from "extrait";
import { readFileSync } from "fs";

const base64 = readFileSync("photo.png").toString("base64");
const img = { base64, mimeType: "image/png" };

// With prompt() builder — pass LLMMessageContent array to .user() or .assistant()
const result = await llm.structured(Schema,
  prompt()
    .system`You are a vision assistant.`
    .user([{ type: "text", text: "Describe this image." }, ...images(img)])
);

// With raw messages array
const result = await llm.structured(Schema, {
  messages: [
    {
      role: "user",
      content: [
        { type: "text", text: "Describe this image." },
        ...images(img),
      ],
    },
  ],
});

// Multiple images
const content = [
  { type: "text", text: "Compare these two images." },
  ...images([
    { base64: base64A, mimeType: "image/png" },
    { base64: base64B, mimeType: "image/jpeg" },
  ]),
];
```

`images()` accepts a single `{ base64, mimeType }` object or an array, and always returns an `LLMImageContent[]` that spreads directly into a content array.

### Conversations (multi-turn history)

Use `conversation()` to build a `LLMMessage[]` from an existing conversation history. This is the idiomatic way to pass prior turns to the LLM.

```typescript
import { conversation } from "extrait";

const messages = conversation("You are a helpful assistant.", [
  { role: "user",      text: "What is the speed of light?" },
  { role: "assistant", text: "Approximately 299,792 km/s in a vacuum." },
  { role: "user",      text: "How long does light take to reach Earth from the Sun?" },
]);

// Pass to adapter directly
const response = await llm.adapter.complete({ messages });

// Or to structured extraction
const result = await llm.structured(Schema, { messages });
```

Entries with `images` produce multimodal content automatically:

```typescript
const messages = conversation("You are a vision assistant.", [
  {
    role: "user",
    text: "What is in this image?",
    images: [{ base64, mimeType: "image/png" }],
  },
]);
```

### Result Object

```typescript
{
  data: T,                      // Validated data matching schema
  raw: string,                  // Raw LLM response
  thinkBlocks: ThinkBlock[],    // Extracted <think> blocks
  json: unknown | null,         // Parsed JSON before validation
  attempts: AttemptTrace[],     // Self-heal attempts
  usage?: {
    inputTokens?: number,
    outputTokens?: number,
    totalTokens?: number,
    cost?: number,
  },
  finishReason?: string,        // e.g., "stop"
}
```

### Error Handling

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
  transport: { apiKey: process.env.OPENAI_API_KEY },
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
  transport: {
    baseURL: "https://api.voyageai.com",
    apiKey: process.env.VOYAGE_API_KEY,
  },
});

const { embeddings } = await embedder.embed(["query", "document"]);
```

Calling `llm.embed()` on an `anthropic-compatible` adapter throws a descriptive error pointing to Voyage AI.

### MCP Tools

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
      maxToolRounds: 5,
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
      // Optional: custom error message when an unknown tool is called
      unknownToolError: (toolName) => `Tool "${toolName}" is not available.`,
    },
  }
);

await mcpClient.close?.();
```

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
  transport: { apiKey: process.env.LLM_API_KEY },
  defaults: {
    timeout: { request: 60_000 },
  },
});
```

## Examples

Run examples with: `bun run dev <example-name>`

Available examples:
- `streaming` - Real LLM streaming + snapshot self-check ([streaming.ts](examples/streaming.ts))
- `streaming-with-tools` - Real text streaming with MCP tools + self-check ([streaming-with-tools.ts](examples/streaming-with-tools.ts))
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

- `LLM_PROVIDER` - `openai-compatible` or `anthropic-compatible`
- `LLM_BASE_URL` - API endpoint (optional)
- `LLM_MODEL` - Model name (default: `gpt-5-nano`)
- `LLM_API_KEY` - API key for the provider
- `STRUCTURED_DEBUG=1` - Enable debug output

## Testing

```bash
bun run test
```
