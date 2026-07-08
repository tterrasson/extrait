/**
 * Generate Example
 *
 * Demonstrates:
 * - High-level unstructured text generation with `llm.generate(...)`
 * - Prompt strings and normalized reasoning extraction
 * - Simple request options forwarding
 *
 * Usage: bun run dev generate [topic]
 */

import { createLLM, prompt } from "@/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "openai-compatible-legacy"
  | "anthropic-compatible";

const model = process.env.LLM_MODEL ?? "gpt-5-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;
const debugEnabled = process.env.STRUCTURED_DEBUG === "1";

if (!apiKey) {
  console.error("Missing LLM_API_KEY in environment.");
  console.error("Set it before running: bun run dev generate");
  process.exit(1);
}

const topic = process.argv.slice(3).join(" ").trim() || "why Bun is fast";

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

console.log(`Provider: ${provider}`);
console.log(`Model: ${model}`);
console.log(`Topic: ${topic}`);
console.log("\nGenerated text:\n");

const result = await llm.generate(
  prompt`
    Write a concise explanation in 3 short paragraphs about:
    ${topic}
  `,
  {
    request: {
      temperature: 0.4,
    },
  },
);

console.log(result.text);

if (result.reasoning.length > 0) {
  console.log("\nReasoning:");
  console.log(result.reasoning);
}

console.log("\nUsage:", result.usage ?? {});
console.log("Finish reason:", result.finishReason ?? "unknown");
