/**
 * Decision Models Example
 *
 * Demonstrates:
 * - Scoring options with a llama.cpp decision model via llm.decide() (/v1/systemone)
 * - The three question types: choice, noul (yes/no) and score
 * - Acting on confident answers and escalating the rest
 *
 * Requires a llama.cpp server running a decision model, e.g.:
 *   llama serve -hf ggml-org/Kev-4B-GGUF
 *   LLM_BASE_URL=http://localhost:8080 bun run dev decision
 *
 * Usage: bun run dev decision [message]
 */

import { createLLM } from "@/index";
import { requireBaseURL } from "./env";

const llm = createLLM({
  provider: "openai-compatible",
  model: process.env.DECISION_MODEL ?? process.env.LLM_MODEL ?? "ggml-org/Kev-4B-GGUF",
  baseURL: requireBaseURL(),
  apiKey: process.env.LLM_API_KEY,
});

const message = process.argv[3] ?? "I was charged twice for my order last week and nobody has replied.";

const { answers, model, usage } = await llm.decide(`Customer message: ${message}`, {
  route: {
    type: "choice",
    instructions: "Which team should handle this?",
    criteria: {
      billing: "payments, charges, refunds, invoices",
      shipping: "delivery, tracking, lost or late parcels",
      technical: "bugs, errors, login problems",
    },
  },
  angry: {
    type: "noul",
    instructions: "Is the customer angry?",
    criteria: { true: "frustrated or angry", false: "calm or neutral" },
  },
  urgency: {
    type: "score",
    instructions: "How urgent is this?",
    criteria: ["can wait", "this week", "today", "right now"],
  },
});

console.log(`Model: ${model} (${usage?.inputTokens ?? "?"} input tokens)\n`);

const { route, angry, urgency } = answers;
const urgencyLabel = urgency.legend[String(Math.round(urgency.score))];

console.log(`Route:   ${route.choice} (confidence ${route.confidence.toFixed(2)})`);
for (const [team, probability] of Object.entries(route.probabilities)) {
  console.log(`  ${team.padEnd(10)} ${(probability * 100).toFixed(1)}%`);
}
console.log(`Angry:   ${(angry.noul * 100).toFixed(1)}% yes`);
console.log(`Urgency: ${urgency.score.toFixed(2)} ≈ "${urgencyLabel}"`);
console.log();

if (route.confidence >= 0.5) {
  console.log(`→ Routed to ${route.choice}.`);
} else {
  console.log("→ Low confidence: sending to a human.");
}
