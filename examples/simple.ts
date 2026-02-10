/**
 * Simple Example
 *
 * Demonstrates:
 * - Basic structured output with a small schema
 * - Streaming output to stdout
 * - English prompt and field descriptions
 * - Error handling with StructuredParseError
 *
 * Usage: bun run dev simple [topic]
 */

import { z } from "zod";
import { createLLM, prompt, s, StructuredParseError } from "../src/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "anthropic-compatible";

const model = process.env.LLM_MODEL ?? "gpt-5-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;
const debugEnabled = process.env.STRUCTURED_DEBUG === "1";

const llm = createLLM({
  provider,
  model,
  transport: {
    baseURL,
    apiKey,
  },
  defaults: {
    mode: "loose",
    selfHeal: 1,
    debug: debugEnabled,
  },
});

const SimpleSchema = s.schema(
  "SimpleResponse",
  z.object({
    topic: s.string().min(1).describe("Main topic in 2 to 4 words."),
    description: s.string().min(1).describe("Short plain-English description."),
    tags: s.array(s.string()).default([]).describe("1 to 3 relevant keywords."),
  })
);

const userInput = process.argv.slice(3).join(" ").trim(); // Skip "bun", "examples/runner.ts", "simple"
const topic = userInput || "Bun.js runtime";

try {
  const result = await llm.structured(
    SimpleSchema,
    prompt`
      Create a simple structured response about: """${topic}"""
      Keep all text concise and in English.
    `,
    {
      stream: {
        to: "stdout",
        onData: (event) => {
          if (event.done) {
            console.log("\n[stream] snapshot complete:", JSON.stringify(event.data));
          }
        },
      },
    },
  );

  console.log("\n\nStructured result:");
  console.log(JSON.stringify(result.data, null, 2));
  console.log("Usage:", result.usage ?? {});
  console.log("Finish reason:", result.finishReason ?? "unknown");
} catch (error) {
  if (error instanceof StructuredParseError) {
    console.error("Structured parsing failed.");
    console.error("Attempt:", error.attempt);
    console.error("Candidates:", error.candidates.length);
    console.error("Zod issues:", error.zodIssues ?? []);
    console.error("Repair log:", error.repairLog ?? []);
    process.exit(1);
  }

  throw error;
}
