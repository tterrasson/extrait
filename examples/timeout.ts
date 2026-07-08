/**
 * Timeout Example (real LLM call)
 *
 * Demonstrates:
 * - Setting per-request and per-tool timeouts via `timeout`
 * - Timeout as a first-class option (no manual AbortSignal needed)
 *
 * Usage: bun run dev timeout [request-timeout-ms]
 * Default timeout: 5000ms
 */

import { z } from "zod";
import { createLLM, prompt, s } from "@/index";

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
  console.error("Set it before running: bun run dev timeout");
  process.exit(1);
}

const requestTimeoutMs = Number(process.argv[3] ?? "5000");

if (!Number.isFinite(requestTimeoutMs) || requestTimeoutMs <= 0) {
  console.error(`Invalid timeout: "${process.argv[3] ?? ""}"`);
  process.exit(1);
}

const llm = createLLM({
  provider,
  model,
  transport: { baseURL, apiKey },
  defaults: {
    mode: "strict",
    selfHeal: false,
    debug: debugEnabled,
  },
});

const SummarySchema = s.schema(
  "Summary",
  z.object({
    topic: s.string().min(1).describe("The topic being summarized"),
    summary: s.string().min(1).describe("A concise summary"),
  }),
);

console.log(`Provider: ${provider}`);
console.log(`Model: ${model}`);
console.log(`Request timeout: ${requestTimeoutMs}ms`);
console.log("Launching generation...\n");

try {
  const result = await llm.structured(
    SummarySchema,
    prompt`Summarize the concept of "structured data extraction from LLMs" in a few sentences.`,
    {
      timeout: {
        request: requestTimeoutMs,
      },
    },
  );

  console.log("Result:", JSON.stringify(result.data, null, 2));
  console.log(`\nTokens used: ${result.usage?.totalTokens ?? "unknown"}`);
} catch (error) {
  if (isTimeoutError(error)) {
    console.log(`Request timed out after ${requestTimeoutMs}ms.`);
    console.log("Try a larger value, e.g. `bun run dev timeout 10000`.");
    process.exit(0);
  }

  console.error("Request failed:", error);
  process.exit(1);
}

function isTimeoutError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const name = "name" in error ? (error as { name: unknown }).name : undefined;
  return name === "AbortError" || name === "TimeoutError";
}
