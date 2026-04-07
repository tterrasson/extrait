/**
 * Streaming With Tools Example (real LLM call)
 *
 * Demonstrates:
 * - High-level text streaming with `llm.generate(...)`
 * - MCP tool usage during generation
 * - Post-tool streaming (not only a final buffered chunk)
 * - A lightweight self-check to validate tools + streaming behavior
 *
 * Usage: bun run dev streaming-with-tools [topic]
 */

import { createLLM, createMCPClient } from "@/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "anthropic-compatible";

const model = process.env.LLM_MODEL ?? "gpt-5-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;
const debugEnabled = process.env.STRUCTURED_DEBUG === "1";

if (!apiKey) {
  console.error("Missing LLM_API_KEY in environment.");
  console.error("Set it before running: bun run dev streaming-with-tools");
  process.exit(1);
}

const llm = createLLM({
  provider,
  model,
  transport: {
    baseURL,
    apiKey,
  },
  defaults: {
    debug: debugEnabled,
  },
});

if (!llm.adapter.stream) {
  console.error(`Provider "${provider}" does not support streaming.`);
  process.exit(1);
}

const calculatorMCP = await createMCPClient({
  id: "calculator",
  transport: {
    type: "stdio",
    command: "bun",
    args: ["run", "examples/calculator-mcp-server.ts"],
  },
});

const expectedMathResult = 128;
const requestPrompt = [
  "You are a precise calculator assistant.",
  "Use the calculate MCP tool to compute ((15 * 7) + 23). Do not do mental math.",
  "Reply with a single sentence stating the result, e.g. 'The result is 128.'",
].join("\n");

const tokens: string[] = [];
const toolExecutions: Array<{ name?: string; error?: string; durationMs?: number }> = [];
let chunkCount = 0;
let started = false;
let completed = false;

console.log(`Provider: ${provider}`);
console.log(`Model: ${model}`);

console.log(`Expected calculation result: ${expectedMathResult}`);
console.log("\nStreaming text output:\n");

try {
  const result = await llm.generate(
    requestPrompt,
    {
      request: {
        temperature: 0,
        mcpClients: [calculatorMCP],
        maxToolRounds: 8,
        toolDebug: debugEnabled
          ? {
              enabled: true,
              includeRequest: true,
              includeResult: true,
              includeResultOnError: true,
              pretty: false,
            }
          : false,
        onToolExecution: (execution) => {
          toolExecutions.push({
            name: execution.name,
            error: execution.error,
            durationMs: execution.durationMs,
          });
        },
      },
      stream: {
        enabled: true,
        onData: (event) => {
          if (!started) {
            started = true;
          }
          if (event.delta.text.length > 0) {
            tokens.push(event.delta.text);
            process.stdout.write(event.delta.text);
          }
          chunkCount += 1;
          if (event.done) {
            completed = true;
          }
        },
      },
    },
  );

  const streamedText = tokens.join("");

  if (!started) {
    throw new Error("Streaming failed: onStart was not called.");
  }

  if (!completed) {
    throw new Error("Streaming failed: onComplete was not called.");
  }

  if (toolExecutions.length === 0) {
    throw new Error("Tools validation failed: no MCP tool execution captured.");
  }

  if (toolExecutions.some((execution) => execution.error)) {
    throw new Error(`Tools validation failed: ${JSON.stringify(toolExecutions)}`);
  }

  if (!toolExecutions.some((execution) => execution.name === "calculate")) {
    throw new Error(`Tools validation failed: expected calculate tool call, got ${JSON.stringify(toolExecutions)}`);
  }

  if (tokens.length === 0) {
    throw new Error("Streaming failed: no token was emitted.");
  }

  if (streamedText.trim().length === 0) {
    throw new Error("Streaming failed: streamed text is empty.");
  }

  const normalizedStream = streamedText.trim();
  const normalizedFinal = result.text.trim();
  const streamIncludesFinal = normalizedFinal.length === 0 || normalizedStream.endsWith(normalizedFinal);
  if (!streamIncludesFinal) {
    throw new Error(
      [
        "Streaming mismatch: final response is not aligned with streamed output.",
        `stream=${JSON.stringify(streamedText)}`,
        `final=${JSON.stringify(result.text)}`,
      ].join("\n"),
    );
  }

  const streamHasExpectedNumber = new RegExp(`\\b${expectedMathResult}\\b`).test(streamedText);
  const finalHasExpectedNumber = new RegExp(`\\b${expectedMathResult}\\b`).test(result.text);
  if (!streamHasExpectedNumber || !finalHasExpectedNumber) {
    throw new Error(
      [
        "Tools validation failed: expected computed number not found.",
        `expected=${expectedMathResult}`,
        `stream=${JSON.stringify(streamedText)}`,
        `final=${JSON.stringify(result.text)}`,
      ].join("\n"),
    );
  }

  console.log("\n");
  console.log(`Chunks: ${chunkCount}`);
  console.log(`Tokens: ${tokens.length}`);
  console.log(`Tool executions: ${toolExecutions.length}`);
  console.log(
    "Tool details:",
    toolExecutions.map((execution, index) => ({
      index: index + 1,
      name: execution.name,
      durationMs: execution.durationMs,
      status: execution.error ? "error" : "ok",
    })),
  );
  console.log("Usage:", result.usage ?? {});
  console.log("Finish reason:", result.finishReason ?? "unknown");
  console.log("Final text:", JSON.stringify(result.text));
  console.log("\n✅ Tools + text streaming check passed.");
} catch (error) {
  console.error("\n❌ Streaming-with-tools example failed:", error);
  process.exit(1);
} finally {
  await calculatorMCP.close?.();
}
