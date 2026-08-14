/**
 * Conversation Example
 *
 * Demonstrates:
 * - Building multi-turn conversation history with `conversation()`
 * - Passing multimodal messages (text + images) in conversation turns
 * - Using `prompt()` builder with `LLMMessageContent` for inline image messages
 *
 * Usage: bun run dev conversation [path-to-image]
 * Example: bun run dev conversation photo.png
 */

import { z } from "zod";
import { conversation, createLLM, loadImages, prompt, s, StructuredParseError } from "@/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "openai-compatible-legacy"
  | "anthropic-compatible";
const debugEnabled = process.env.STRUCTURED_DEBUG === "1";

const llm = createLLM({
  provider,
  model: process.env.LLM_MODEL ?? "gpt-4o-mini",
  transport: { baseURL: process.env.LLM_BASE_URL, apiKey: process.env.LLM_API_KEY },
  defaults: { mode: "loose", selfHeal: 1, debug: debugEnabled },
});

const ReplySchema = s.schema(
  "Reply",
  z.object({
    answer: s.string().min(1).describe("The assistant's answer."),
    confidence: s.number().min(0).max(1).describe("Confidence level between 0 and 1."),
  })
);

// --- Example 1: multi-turn conversation from history ---

console.log("=== Example 1: conversation() from history ===\n");

const messages = conversation("You are a knowledgeable science tutor.", [
  { role: "user", text: "What is the speed of light?" },
  { role: "assistant", text: "The speed of light is approximately 299,792 km/s in a vacuum." },
  { role: "user", text: "How long does it take for light to travel from the Sun to Earth?" },
]);

try {
  const result = await llm.structured(ReplySchema, { messages });
  console.log("Answer:", result.data.answer);
  console.log("Confidence:", result.data.confidence);
} catch (error) {
  if (error instanceof StructuredParseError) {
    console.error("Parse failed:", error.zodIssues);
    process.exit(1);
  }
  throw error;
}

// --- Example 2: prompt() builder with inline image content ---
//
// Requires a vision-capable model (e.g. gpt-4o, claude-3-5-sonnet).
// Pass an image path as argument: bun run dev conversation <path-to-image>

console.log("\n=== Example 2: prompt() builder with LLMMessageContent ===\n");

const imagePath = process.argv[3];

if (!imagePath) {
  console.log("Tip: pass an image path to test multimodal — bun run dev conversation <path-to-image>");
} else {
  // loadImages() reads the file and builds the content blocks — no resizing.
  const imageContent = await loadImages(imagePath);

  const multimodalPrompt = prompt()
    .system`You are a vision assistant. Describe images concisely.`
    .user([{ type: "text", text: "What color is dominant in this image?" }, ...imageContent]);

  try {
    const result = await llm.structured(ReplySchema, multimodalPrompt);
    console.log("Answer:", result.data.answer);
    console.log("Confidence:", result.data.confidence);
  } catch (error) {
    if (error instanceof StructuredParseError) {
      console.error("Parse failed:", error.zodIssues);
      process.exit(1);
    }
    throw error;
  }
}
