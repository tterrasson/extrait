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
  GenerateOptions,
  GenerateResult,
  LLMAdapter,
  LLMMessage,
  StructuredPromptBuilder,
} from "./types";

export async function generate(
  adapter: LLMAdapter,
  prompt: StructuredPromptBuilder,
  options?: GenerateCallOptions,
): Promise<GenerateResult>;
export async function generate(
  adapter: LLMAdapter,
  options: GenerateOptions,
): Promise<GenerateResult>;
export async function generate(
  adapter: LLMAdapter,
  promptOrOptions: StructuredPromptBuilder | GenerateOptions,
  callOptions?: GenerateCallOptions,
): Promise<GenerateResult> {
  const normalized = normalizeGenerateInput(promptOrOptions, callOptions);
  const useOutdent = normalized.outdent ?? true;
  const streamConfig = normalizeStreamConfig(normalized.stream);
  const debugConfig = normalizeDebugConfig(normalized.debug);
  const resolvedPrompt = applyPromptOutdent(resolvePrompt(normalized.prompt, { mode: "loose" }), useOutdent);
  const resolvedSystemPrompt = applyOutdentToOptionalPrompt(normalized.systemPrompt, useOutdent);
  const preparedPrompt = prepareGeneratePromptPayload(resolvedPrompt, resolvedSystemPrompt);

  const resolvedRequest =
    normalized.timeout?.tool !== undefined && normalized.request?.mcpClients !== undefined
      ? {
          ...normalized.request,
          mcpClients: applyToolTimeout(normalized.request.mcpClients, normalized.timeout.tool),
        }
      : normalized.request;

  const response = await callModel(adapter, {
    prompt: preparedPrompt.prompt,
    messages: preparedPrompt.messages,
    systemPrompt: preparedPrompt.systemPrompt,
    request: resolvedRequest,
    stream: streamConfig,
    observe: normalized.observe,
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
    timeout: normalized.timeout,
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

  normalized.observe?.({
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

function normalizeGenerateInput(
  promptOrOptions: StructuredPromptBuilder | GenerateOptions,
  callOptions?: GenerateCallOptions,
): GenerateOptions {
  if (isGenerateOptions(promptOrOptions)) {
    return promptOrOptions;
  }

  if (!promptOrOptions) {
    throw new Error("Missing prompt in generate(adapter, prompt, options?) call.");
  }

  return {
    ...callOptions,
    prompt: promptOrOptions,
  };
}

function isGenerateOptions(value: StructuredPromptBuilder | GenerateOptions): value is GenerateOptions {
  return typeof value === "object" && value !== null && "prompt" in value;
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
