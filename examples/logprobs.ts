/**
 * Logprobs + Debug Example
 *
 * Demonstrates:
 * - Requesting token log probabilities from an OpenAI-compatible chat endpoint
 * - Enabling verbose request/response debug output
 * - Inspecting chosen tokens and their most likely alternatives
 *
 * Usage: bun run dev logprobs [prompt]
 */

import { createLLM, type LLMTokenLogprob } from "@/index";

const model = process.env.LLM_MODEL ?? "gpt-4.1-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;

if (!apiKey) {
  console.error("Missing LLM_API_KEY in environment.");
  console.error("Set it before running: bun run dev logprobs");
  process.exit(1);
}

const input = process.argv.slice(3).join(" ").trim() || "Answer with one word: yes or no?";
const llm = createLLM({
  provider: "openai-compatible",
  model,
  transport: { baseURL, apiKey },
});

const result = await llm.generate(input, {
  debug: {
    enabled: true,
    verbose: true,
  },
  request: {
    body: {
      logprobs: true,
      top_logprobs: 3,
    },
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
