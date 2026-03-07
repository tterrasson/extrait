/**
 * Abort Signal Example (real LLM call)
 *
 * Demonstrates:
 * - Passing AbortSignal through `request.signal`
 * - Starting a structured generation
 * - Stopping the request quickly to validate cancellation behavior
 *
 * Usage: bun run dev abort-signal [abort-after-ms] [topic]
 */

import { z } from "zod";
import { createLLM, prompt, s } from "@/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "anthropic-compatible";
const model = process.env.LLM_MODEL ?? "gpt-5-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;

if (!apiKey) {
  console.error("Missing LLM_API_KEY in environment.");
  console.error("Set it before running: bun run dev abort-signal");
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

const AbortSchema = s.schema(
  "AbortSignalValidation",
  z.object({
    topic: s.string().min(1),
    summary: s.string().min(1),
  }),
);

const abortAfterMs = Number(process.argv[3] ?? "120");
const topic = process.argv.slice(4).join(" ").trim() || "AbortSignal in LLM requests";

if (!Number.isFinite(abortAfterMs) || abortAfterMs < 0) {
  console.error(`Invalid abort delay: "${process.argv[3] ?? ""}"`);
  process.exit(1);
}

const controller = new AbortController();
const timeout = setTimeout(() => {
  console.log(`[abort] stopping request after ${abortAfterMs}ms`);
  controller.abort();
}, Math.floor(abortAfterMs));

try {
  console.log(`Provider: ${provider}`);
  console.log(`Model: ${model}`);
  console.log(`Abort after: ${Math.floor(abortAfterMs)}ms`);
  console.log("Launching generation...");

  const result = await llm.structured(
    AbortSchema,
    prompt`
      Produce JSON in English about "${topic}".
      Make the summary very detailed (at least 12 sentences) before finishing.
    `,
    {
      request: {
        signal: controller.signal,
        maxTokens: 2_000,
      },
      stream: {
        enabled: true,
      },
    },
  );

  console.error("Request completed before abort.");
  console.error("Try a smaller delay, e.g. `bun run dev abort-signal 20`.");
  console.error("Received data:", JSON.stringify(result.data));
  process.exit(1);
} catch (error) {
  if (isAbortError(error)) {
    console.log("Abort caught successfully. Signal cancellation is wired correctly.");
    process.exit(0);
  }

  console.error("Request failed with a non-abort error:", error);
  process.exit(1);
} finally {
  clearTimeout(timeout);
}

function isAbortError(error: unknown): boolean {
  if (typeof error === "string") {
    const value = error.toLowerCase();
    return value.includes("abort") || value.includes("stop");
  }

  if (!error || typeof error !== "object") {
    return false;
  }

  const maybeName = "name" in error ? error.name : undefined;
  const maybeMessage = "message" in error ? error.message : undefined;
  const name = typeof maybeName === "string" ? maybeName : "";
  const message = typeof maybeMessage === "string" ? maybeMessage.toLowerCase() : "";

  return name === "AbortError" || message.includes("abort");
}
