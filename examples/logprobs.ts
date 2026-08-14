/**
 * Logprobs + Debug Example
 *
 * Demonstrates:
 * - Requesting token log probabilities from an OpenAI-compatible Responses endpoint
 * - Enabling verbose request/response debug output
 * - Inspecting chosen tokens and their most likely alternatives
 *
 * Usage: bun run dev logprobs [--legacy-logprobs] [prompt]
 *
 * Pass --legacy-logprobs for Responses servers (e.g. llama.cpp) that reject a
 * bare `top_logprobs` and require the Chat Completions-style `logprobs: true`
 * flag; it is forwarded through `transport.defaultBody`.
 */

import { createLLM, type LLMTokenLogprob } from "@/index";
import { requireBaseURL } from "./env";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "openai-compatible-legacy";
const model = process.env.LLM_MODEL ?? "my-model-id";
const baseURL = requireBaseURL();
const apiKey = process.env.LLM_API_KEY;

if (!apiKey) {
  console.error("Missing LLM_API_KEY in environment.");
  console.error("Set it before running: bun run dev logprobs");
  process.exit(1);
}

const args = process.argv.slice(3);
const legacyLogprobs = args.includes("--legacy-logprobs");
const input = args.filter((arg) => arg !== "--legacy-logprobs").join(" ").trim()
  || "Answer with one word: yes or no?";
const llm = createLLM({
  provider,
  model,
  baseURL,
  apiKey,
  ...(legacyLogprobs ? { transport: { defaultBody: { logprobs: true } } } : {}),
});

const result = await llm.generate(input, {
  debug: {
    enabled: true,
    verbose: true,
  },
  request: {
    topLogprobs: 3,
  },
});

console.log("\nGenerated text:", result.text);

const tokens = result.logprobs?.content ?? [];
if (tokens.length === 0) {
  console.log("\nNo logprobs returned. Check that the selected model supports them.");
  process.exit(0);
}

console.log("\nToken probabilities:");
console.table(tokens.map(formatToken));

function formatToken(entry: LLMTokenLogprob) {
  return {
    token: JSON.stringify(entry.token),
    logprob: entry.logprob.toFixed(4),
    probability: `${(Math.exp(entry.logprob) * 100).toFixed(2)}%`,
    alternatives: (entry.top_logprobs ?? [])
      .filter((alternative) => alternative.token !== entry.token)
      .map(
        (alternative) =>
          `${JSON.stringify(alternative.token)} (${(Math.exp(alternative.logprob) * 100).toFixed(2)}%)`,
      )
      .join(", "),
  };
}
