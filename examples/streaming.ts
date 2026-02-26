/**
 * Streaming Example (real LLM call)
 *
 * Demonstrates:
 * - Unified structured streaming with `onData`
 * - Progressive partial snapshots while the model is generating
 * - A lightweight self-check to verify streaming behavior
 *
 * Usage: bun run dev streaming [text to analyze]
 */

import { z } from "zod";
import { createLLM, prompt, s, StructuredParseError } from "../src/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "anthropic-compatible";

const model = process.env.LLM_MODEL ?? "gpt-5-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;

if (!apiKey) {
  console.error("Missing LLM_API_KEY in environment.");
  console.error("Set it before running: bun run dev streaming");
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
    mode: "strict",
    selfHeal: false,
  },
});

const SentimentSchema = s.schema(
  "SentimentStreaming",
  z.object({
    sentiment: z
      .enum(["positive", "negative", "neutral"])
      .describe("Overall sentiment in lowercase"),
    confidence: s.number().min(0).max(1).describe("Confidence between 0 and 1"),
  }),
);

const userInput = process.argv.slice(3).join(" ").trim();
const textToAnalyze = userInput || "I love this product, it works very well and saves me time.";

const snapshots: Array<{ data: unknown; done: boolean }> = [];

console.log(`Provider: ${provider}`);
console.log(`Model: ${model}`);
console.log("\nStreaming raw output:");

try {
  const result = await llm.structured(
    SentimentSchema,
    prompt`
      Analyze the sentiment of the following text and return strictly valid JSON:

      """
      ${textToAnalyze}
      """
    `,
    {
      request: {
        temperature: 0,
      },
      stream: {
        enabled: true,
        to: "stdout",
        onData: (event) => {
          snapshots.push({
            data: event.data,
            done: event.done,
          });
        },
      },
    },
  );

  if (snapshots.length === 0) {
    throw new Error("Streaming failed: no snapshot was emitted.");
  }

  const finalSnapshot = snapshots[snapshots.length - 1];
  if (!finalSnapshot?.done) {
    throw new Error("Streaming failed: missing final done=true snapshot.");
  }

  if (finalSnapshot.data === null || typeof finalSnapshot.data !== "object") {
    throw new Error("Streaming failed: final snapshot has no structured data.");
  }

  const finalData = finalSnapshot.data as Record<string, unknown>;
  if (finalData.sentiment !== result.data.sentiment || finalData.confidence !== result.data.confidence) {
    throw new Error(
      [
        "Streaming mismatch between final snapshot and parsed result.",
        `snapshot=${JSON.stringify(finalSnapshot.data)}`,
        `result=${JSON.stringify(result.data)}`,
      ].join("\n"),
    );
  }

  console.log("\n\nSnapshots:");
  for (const snapshot of snapshots) {
    console.log(`- done=${snapshot.done} data=${JSON.stringify(snapshot.data)}`);
  }

  console.log("\nFinal data:", JSON.stringify(result.data));
  console.log("Usage:", result.usage ?? {});
  console.log("Finish reason:", result.finishReason ?? "unknown");
  console.log("\n✅ Streaming check passed.");
} catch (error) {
  if (error instanceof StructuredParseError) {
    console.error("\n❌ Structured parsing failed.");
    console.error("Attempt:", error.attempt);
    console.error("Candidates:", error.candidates.length);
    console.error("Zod issues:", error.zodIssues ?? []);
    console.error("Repair log:", error.repairLog ?? []);
    process.exit(1);
  }

  console.error("\n❌ Streaming example failed:", error);
  process.exit(1);
}
