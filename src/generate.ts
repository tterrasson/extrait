import {
  aggregateUsage,
  applyOutdentToOptionalPrompt,
  applyPromptOutdent,
  applyToolTimeout,
  callModel,
  mergeSystemPrompts,
  normalizeDebugConfig,
  normalizeStreamConfig,
  resolvePrompt,
} from "./generate-shared";
import type {
  GenerateAttempt,
  GenerateCallOptions,
  GenerateResult,
  LLMAdapter,
  LLMMessage,
  StructuredPromptBuilder,
} from "./types";

/**
 * Generates free-form text. `prompt` is a string, a `prompt\`...\`` template, a
 * `prompt()` builder, a `{ prompt?, systemPrompt?, messages? }` payload, or a
 * function of the context returning one of those; `options` configure the call.
 */
export async function generate(
  adapter: LLMAdapter,
  prompt: StructuredPromptBuilder,
  options: GenerateCallOptions = {},
): Promise<GenerateResult> {
  assertGeneratePrompt(prompt);
  const useOutdent = options.outdent ?? true;
  const streamConfig = normalizeStreamConfig(options.stream);
  const debugConfig = normalizeDebugConfig(options.debug);
  const resolvedPrompt = applyPromptOutdent(resolvePrompt(prompt, { mode: "loose" }), useOutdent);
  const resolvedSystemPrompt = applyOutdentToOptionalPrompt(options.systemPrompt, useOutdent);
  const preparedPrompt = prepareGeneratePromptPayload(resolvedPrompt, resolvedSystemPrompt);

  const resolvedRequest =
    options.timeout?.tool !== undefined && options.request?.mcpClients !== undefined
      ? {
          ...options.request,
          mcpClients: applyToolTimeout(options.request.mcpClients, options.timeout.tool),
        }
      : options.request;

  const response = await callModel(adapter, {
    prompt: preparedPrompt.prompt,
    messages: preparedPrompt.messages,
    systemPrompt: preparedPrompt.systemPrompt,
    request: resolvedRequest,
    stream: streamConfig,
    observe: options.observe,
    buildEvent: ({ stage, message, details }) => ({
      stage,
      attempt: 1,
      message,
      details,
    }),
    buildSnapshot: (model) => ({
      text: model.text,
      reasoning: model.reasoning,
      ...(model.reasoningBlocks ? { reasoningBlocks: model.reasoningBlocks } : {}),
    }),
    debug: debugConfig,
    debugLabel: "generate",
    attempt: 1,
    selfHeal: false,
    selfHealEnabled: false,
    timeout: options.timeout,
  });

  const attempt: GenerateAttempt = {
    attempt: 1,
    via: response.via,
    text: response.text,
    reasoning: response.reasoning,
    usage: response.usage,
    finishReason: response.finishReason,
    ...(response.logprobs ? { logprobs: response.logprobs } : {}),
    ...(response.reasoningBlocks ? { reasoningBlocks: response.reasoningBlocks } : {}),
  };
  const attempts = [attempt];

  options.observe?.({
    stage: "result",
    attempt: 1,
    message: "Text generation completed.",
    details: {
      via: response.via,
      finishReason: response.finishReason,
    },
  });

  return {
    text: attempt.text,
    reasoning: attempt.reasoning,
    attempts,
    usage: aggregateUsage(attempts),
    finishReason: attempt.finishReason,
    ...(attempt.logprobs ? { logprobs: attempt.logprobs } : {}),
    ...(attempt.reasoningBlocks ? { reasoningBlocks: attempt.reasoningBlocks } : {}),
  };
}

// Keys of the former single-object `generate({ prompt, ...options })` form.
const LEGACY_OPTION_KEYS = ["stream", "request", "debug", "observe", "outdent", "timeout"] as const;

/**
 * `generate()` used to also accept one object mixing the prompt and the
 * options, which could not be told apart from a prompt payload: a payload's
 * `messages` were dropped, and an options object is now a payload whose options
 * would be. Both shapes are rejected with the way out instead.
 */
function assertGeneratePrompt(prompt: StructuredPromptBuilder): void {
  if (!prompt) {
    throw new Error("Missing prompt in generate(prompt, options?) call.");
  }
  if (typeof prompt !== "object" || "resolvePrompt" in prompt) {
    return;
  }

  const record = prompt as Record<string, unknown>;
  const legacyKeys = LEGACY_OPTION_KEYS.filter((key) => key in record);
  if ((record.prompt !== undefined && typeof record.prompt !== "string") || legacyKeys.length > 0) {
    throw new TypeError(
      "generate() takes the prompt and the options as two arguments: " +
        "generate(prompt, options). The single-object form generate({ prompt, ...options }) was removed" +
        (legacyKeys.length > 0 ? ` (found option ${legacyKeys.map((key) => `"${key}"`).join(", ")} in the prompt).` : "."),
    );
  }
}

function prepareGeneratePromptPayload(
  payload: ReturnType<typeof resolvePrompt>,
  systemPrompt: string | undefined,
): { prompt?: string; systemPrompt?: string; messages?: LLMMessage[] } {
  if (Array.isArray(payload.messages) && payload.messages.length > 0) {
    const messages = payload.messages.map((message) => ({ ...message }));
    const mergedSystemPrompt = mergeSystemPrompts(payload.systemPrompt, systemPrompt);
    const systemMessages = mergedSystemPrompt ? [{ role: "system" as const, content: mergedSystemPrompt }] : [];

    return {
      messages: [...systemMessages, ...messages],
    };
  }

  const resolvedPrompt = payload.prompt?.trim();
  if (!resolvedPrompt) {
    throw new Error("Structured prompt payload must include a non-empty prompt or messages.");
  }

  return {
    prompt: resolvedPrompt,
    systemPrompt: mergeSystemPrompts(payload.systemPrompt, systemPrompt),
  };
}
