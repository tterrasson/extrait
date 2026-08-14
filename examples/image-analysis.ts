/**
 * Image Analysis Example
 *
 * Demonstrates:
 * - Loading an image from a file path with `loadImages()`
 * - Sending multimodal messages to a vision-capable LLM
 * - Structured extraction from image analysis
 *
 * Images are sent as-is: extrait never decodes or resizes them. Resize upstream
 * if you care about token cost — see the commented snippet below.
 *
 * Usage: bun run dev image-analysis <path-to-image-or-url>
 * Example: bun run dev image-analysis photo.png
 */

import { z } from "zod";
import { createLLM, loadImages, s, StructuredParseError } from "@/index";

const filePath = process.argv[3];

if (!filePath) {
  console.error("Usage: bun run dev image-analysis <path-to-image-or-url>");
  console.error("Example: bun run dev image-analysis photo.png");
  process.exit(1);
}

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "openai-compatible-legacy"
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

// Accepts a file path, a data URL, an http(s) URL, a Blob or raw bytes.
const imageContent = await loadImages(filePath);

// To resize before sending, do it yourself with the tool of your choice:
//
//   import sharp from "sharp";
//   const buf = await sharp(filePath)
//     .resize(512, 512, { fit: "inside", withoutEnlargement: true })
//     .toBuffer();
//   const imageContent = images(buf);

try {
  const result = await llm.structured(
    ImageAnalysisSchema,
    {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Analyze this image and return structured data." },
            ...imageContent,
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
