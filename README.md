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
import { images } from "extrait";
import { readFileSync } from "fs";

const base64 = readFileSync("photo.png").toString("base64");

// Single image
const result = await llm.structured(Schema, {
  messages: [
    {
      role: "user",
      content: [
        { type: "text", text: "Describe this image." },
        ...images({ base64, mimeType: "image/png" }),
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

Pass arguments after the example name:
```bash
bun run dev streaming
bun run dev streaming-with-tools
bun run dev abort-signal 120 "JSON cancellation demo"
bun run dev timeout 5000
bun run dev simple "Bun.js runtime"
bun run dev sentiment-analysis "I love this product."
bun run dev multi-step-reasoning "Why is the sky blue?"
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
