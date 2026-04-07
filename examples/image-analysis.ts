/**
 * Image Analysis Example
 *
 * Demonstrates:
 * - Building base64 image content blocks with `images()`
 * - Sending multimodal messages to a vision-capable LLM
 * - Structured extraction from image analysis
 *
 * Usage: bun run dev image-analysis <path-to-image> [low|mid|high|raw|<px>]
 * Example: bun run dev image-analysis photo.png high
 */

import { z } from "zod";
import {
  createLLM,
  images,
  resizeImage,
  s,
  StructuredParseError,
  type ImageSize
} from "@/index";

const filePath = process.argv[3];
const rawSize = process.argv[4] ?? "mid";
const sizeArg: ImageSize = /^\d+$/.test(rawSize) ? parseInt(rawSize, 10) : (rawSize as ImageSize);

if (!filePath) {
  console.error("Usage: bun run dev image-analysis <path-to-image> [low|mid|high|raw|<px>]");
  console.error("Example: bun run dev image-analysis photo.png high");
  process.exit(1);
}

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "anthropic-compatible";

const model = process.env.LLM_MODEL ?? "gpt-4o-mini";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;
const debugEnabled = process.env.STRUCTURED_DEBUG === "1";

const llm = createLLM({
  provider,
  model,
  transport: { baseURL, apiKey },
  defaults: { mode: "loose", selfHeal: 1, debug: debugEnabled },
});

const ImageAnalysisSchema = s.schema(
  "ImageAnalysis",
  z.object({
    description: s.string().min(1).describe("What is visible in the image."),
    colors: s.array(s.string()).describe("Dominant colors present."),
    objects: s.array(s.string()).describe("Main objects or subjects detected."),
    mood: s.string().describe("Overall mood or atmosphere of the image."),
  })
);

const imageInput = await resizeImage(filePath, sizeArg);

try {
  const result = await llm.structured(
    ImageAnalysisSchema,
    {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Analyze this image and return structured data." },
            ...images(imageInput),
          ],
        },
      ],
    },
    { request: { signal: AbortSignal.timeout(300_000) } },
  );

  console.log("Image analysis:");
  console.log(JSON.stringify(result.data, null, 2));
  console.log("Usage:", result.usage ?? {});
} catch (error) {
  if (error instanceof StructuredParseError) {
    console.error("Structured parsing failed.");
    console.error("Zod issues:", error.zodIssues ?? []);
    process.exit(1);
  }
  throw error;
}
