/**
 * Streaming With Tools Example (real LLM call)
 *
 * Demonstrates:
 * - Raw text streaming with `llm.adapter.stream(...)`
 * - MCP tool usage during generation
 * - Post-tool streaming (not only a final buffered chunk)
 * - A lightweight self-check to validate tools + streaming behavior
 *
 * Usage: bun run dev streaming-with-tools [topic]
 */

import { createLLM, createMCPClient } from "../src/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "anthropic-compatible";

const model = process.env.LLM_MODEL ?? "gpt-5-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;

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

const userInput = process.argv.slice(3).join(" ").trim();
const topic = userInput || "the benefits of text streaming generation";
const expectedMathResult = 128;
const requestPrompt = [
  "You are a precise assistant.",
  "Use MCP tools to compute ((15 * 7) + 23).",
  "Do not do mental math.",
  "Then write one concise English sentence that includes the computed number.",
  `Topic: """${topic}"""`,
].join("\n");

const tokens: string[] = [];
const toolExecutions: Array<{ name?: string; error?: string; durationMs?: number }> = [];
let chunkCount = 0;
let started = false;
let completed = false;

console.log(`Provider: ${provider}`);
console.log(`Model: ${model}`);
console.log(`Topic: ${topic}`);
console.log(`Expected calculation result: ${expectedMathResult}`);
console.log("\nStreaming text output:\n");

try {
  const result = await llm.adapter.stream(
    {
      prompt: requestPrompt,
      temperature: 0,
      mcpClients: [calculatorMCP],
      maxToolRounds: 8,
      onToolExecution: (execution) => {
        toolExecutions.push({
          name: execution.name,
          error: execution.error,
          durationMs: execution.durationMs,
        });
      },
    },
    {
      onStart: () => {
        started = true;
      },
      onToken: (token) => {
        tokens.push(token);
        process.stdout.write(token);
      },
      onChunk: () => {
        chunkCount += 1;
      },
      onComplete: () => {
        completed = true;
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
