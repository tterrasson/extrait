/**
 * Sentiment Analysis Example
 *
 * Demonstrates:
 * - Enum types for categorical data
 * - Number validation with ranges (0-1 confidence)
 * - Boolean fields
 * - Simple, fast execution (no streaming)
 * - Multiple data types in one schema
 *
 * Usage: bun run dev sentiment-analysis [text to analyze]
 */

import { z } from "zod";
import { createLLM, prompt, s, StructuredParseError } from "@/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "openai-compatible-legacy"
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
    mode: "strict", // Use strict mode for faster, simpler execution
    selfHeal: false,
    debug: debugEnabled
  },
});

// Schema with enums, numbers with validation, and booleans
const SentimentSchema = s.schema(
  "SentimentAnalysis",
  z.object({
    sentiment: z.enum(["positive", "negative", "neutral"]).describe("Overall sentiment of the text"),
    confidence: s
      .number()
      .min(0)
      .max(1)
      .describe("Confidence score between 0 and 1"),
    emotional: s.boolean().describe("Whether the text is emotional or factual"),
    urgent: s.boolean().describe("Whether the text conveys urgency"),
    keywords: s
      .array(s.string())
      .max(5)
      .describe("Up to 5 key words that indicate the sentiment"),
    category: z
      .enum(["feedback", "complaint", "question", "praise", "neutral"])
      .optional()
      .describe("Type of message"),
  })
);

const userInput = process.argv.slice(3).join(" ").trim();
const textToAnalyze =
  userInput ||
  "I absolutely love this product! It's been a game-changer for my workflow. Thank you so much!";

console.log("🎭 Analyzing sentiment...\n");
console.log("Text:", textToAnalyze);
console.log();

try {
  const result = await llm.structured(
    SentimentSchema,
    prompt`
      Analyze the sentiment of this text and provide structured analysis:

      """
      ${textToAnalyze}
      """

      Be accurate with the confidence score and identify whether it's emotional or factual.
    `,
    {
      // Keep strict mode, but allow lightweight JSON repair for malformed escapes.
      parse: {
        repair: true,
      },
    }
    // No streaming - get results immediately
  );

  console.log("✅ Analysis complete!\n");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log(`Sentiment:  ${result.data.sentiment.toUpperCase()}`);
  console.log(`Confidence: ${(result.data.confidence * 100).toFixed(1)}%`);
  console.log(`Emotional:  ${result.data.emotional ? "Yes" : "No"}`);
  console.log(`Urgent:     ${result.data.urgent ? "Yes" : "No"}`);
  if (result.data.category) {
    console.log(`Category:   ${result.data.category}`);
  }
  console.log(`Keywords:   ${result.data.keywords.join(", ")}`);
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  console.log("📊 Usage:", result.usage ?? {});
} catch (error) {
  if (error instanceof StructuredParseError) {
    console.error("\n❌ Sentiment analysis failed.");
    console.error("Validation issues:", error.zodIssues ?? []);
    if ((error.zodIssues?.length ?? 0) === 0) {
      console.error(
        "No schema issues were reported. The model response was likely not valid JSON.",
      );
    }
    console.error("\nModel text:");
    console.error(error.text);
    if (error.reasoning.length > 0) {
      console.error("\nModel reasoning:");
      console.error(error.reasoning);
    }
    if (error.candidates.length > 0) {
      console.error("\nExtracted JSON candidates:");
      error.candidates.forEach((candidate, index) => {
        console.error(`  [${index + 1}] ${candidate}`);
      });
    }
    if ((error.repairLog?.length ?? 0) > 0) {
      console.error("\nRepair diagnostics:");
      error.repairLog?.forEach((line) => console.error(`  - ${line}`));
    }
    process.exit(1);
  }

  throw error;
}
