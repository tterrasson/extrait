import type {
  LLMAdapter,
  LLMLogprobs,
  LLMMessage,
  LLMRequest,
  LLMResponse,
  LLMStreamCallbacks,
  LLMTokenLogprob,
  LLMToolCall,
  LLMUsage,
  ReasoningBlock,
} from "../types";
import { consumeSSE } from "./stream-utils";
import {
  executeMCPToolCalls,
  hasMCPClients,
  normalizeMaxToolRounds,
  resolveMCPToolset,
  stringifyToolOutput,
  toProviderFunctionTools,
} from "./mcp-runtime";
import {
  cleanUndefined,
  isRecord,
  mergeUsage,
  pickString,
  preferLatestUsage,
  readErrorBody,
  safeJSONParse,
  toFiniteNumber,
} from "./utils";
import {
  embedOpenAI,
  emitOpenAIStreamChunk,
  joinReasoningBlocks,
  normalizeLogprobEntries,
  pickReasoningText,
  pickTextLike,
  pickUsage,
  pushReasoningBlock,
  sendOpenAIJsonRequest,
  sendOpenAIRequest,
  toOpenAIReasoningEffort,
  validateTopLogprobs,
} from "./openai-compatible-common";
import type { OpenAICompatibleAdapterOptions } from "./openai-compatible-common";

export type OpenAICompatibleLegacyAdapterOptions = OpenAICompatibleAdapterOptions;

export function createOpenAICompatibleLegacyAdapter(options: OpenAICompatibleLegacyAdapterOptions): LLMAdapter {
  const fetcher = options.fetcher ?? fetch;
  const path = options.path ?? "/v1/chat/completions";
  const embeddingPath = options.embeddingPath ?? "/v1/embeddings";

  return {
    provider: "openai-compatible-legacy",
    model: options.model,
    complete(request) {
      return hasMCPClients(request.mcpClients)
        ? completeWithChatCompletionsWithMCP(options, fetcher, path, request)
        : completeWithChatCompletionsPassThrough(options, fetcher, path, request);
    },
    stream(request, callbacks = {}) {
      return hasMCPClients(request.mcpClients)
        ? streamWithChatCompletionsWithMCP(options, fetcher, path, request, callbacks)
        : streamWithChatCompletionsPassThrough(options, fetcher, path, request, callbacks);
    },
    embed(request) {
      return embedOpenAI(options, fetcher, embeddingPath, request);
    },
  };
}

async function streamWithChatCompletionsPassThrough(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  callbacks: LLMStreamCallbacks,
): Promise<LLMResponse> {
  const response = await sendOpenAIRequest(
    options,
    fetcher,
    path,
    request,
    buildChatCompletionsBody(options, request, {
      messages: buildMessages(request),
      stream: true,
      stream_options: streamUsageOptions(request),
    }),
  );

  if (!response.ok) {
    const message = await readErrorBody(response);
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  callbacks.onStart?.();
  let text = "";
  let reasoning = "";
  let usage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  let streamTerminated = false;
  const logprobsContent: LLMTokenLogprob[] = [];
  const logprobsRefusal: LLMTokenLogprob[] = [];
  const streamedToolCalls = new Map<number, OpenAIStreamToolCallState>();
  const nativeToolCalls = new NativeToolCallStreamState(requestDeclaresTools(options, request));

  await consumeSSE(response, (data) => {
    if (data === "[DONE]") {
      streamTerminated = true;
      return;
    }

    const json = safeJSONParse(data);
    if (!isRecord(json)) {
      throw new Error("Invalid JSON event in OpenAI Chat Completions stream.");
    }

    throwForChatCompletionsStreamError(json);
    lastPayload = json;

    const rawDelta = pickAssistantDelta(json);
    const reasoningDelta = pickAssistantReasoningDelta(json);
    const chunkUsage = pickUsage(json);
    const chunkFinishReason = pickFinishReason(json);
    const chunkLogprobs = pickChatLogprobs(json);
    appendChatLogprobs(chunkLogprobs, logprobsContent, logprobsRefusal);
    collectOpenAIStreamToolCalls(json, streamedToolCalls);
    const nativeDelta = nativeToolCalls.push(rawDelta);
    const delta = nativeDelta.textDelta;
    const chunkToolCalls = mergeToolCalls(buildOpenAIStreamToolCalls(streamedToolCalls), nativeDelta.toolCalls);

    usage = preferLatestUsage(usage, chunkUsage);
    if (chunkFinishReason) {
      finishReason = chunkFinishReason;
      streamTerminated = true;
    }

    if (delta) {
      text += delta;
      callbacks.onToken?.(delta);
    }

    if (reasoningDelta) {
      reasoning += reasoningDelta;
    }

    emitOpenAIStreamChunk(
      callbacks,
      undefined,
      json,
      delta,
      reasoningDelta,
      chunkUsage,
      chunkFinishReason,
      chunkToolCalls.length > 0 ? chunkToolCalls : undefined,
      chunkLogprobs,
    );
  });

  assertChatCompletionsStreamTerminated(streamTerminated);

  const tail = nativeToolCalls.flush();
  if (tail.textDelta) {
    text += tail.textDelta;
    callbacks.onToken?.(tail.textDelta);
    emitOpenAIStreamChunk(callbacks, undefined, {}, tail.textDelta, "", undefined, undefined);
  }
  const toolCalls = mergeToolCalls(buildOpenAIStreamToolCalls(streamedToolCalls), nativeToolCalls.calls);

  const out = {
    text,
    reasoning: reasoning.length > 0 ? reasoning : undefined,
    raw: lastPayload,
    usage,
    finishReason: finishReason ?? (toolCalls.length > 0 ? "tool_calls" : undefined),
    ...withChatLogprobs(logprobsContent, logprobsRefusal),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
  };
  callbacks.onComplete?.(out);
  return out;
}

function buildChatCompletionsBody(
  options: OpenAICompatibleAdapterOptions,
  request: LLMRequest,
  overrides: Record<string, unknown>,
): Record<string, unknown> {
  const topLogprobs = validateTopLogprobs(request.topLogprobs);
  return cleanUndefined({
    ...options.defaultBody,
    ...request.body,
    model: options.model,
    ...(request.temperature !== undefined ? { temperature: request.temperature } : {}),
    ...(request.reasoningEffort
      ? { reasoning_effort: toOpenAIReasoningEffort(request.reasoningEffort) }
      : {}),
    ...(request.maxTokens !== undefined
      ? { max_tokens: undefined, max_completion_tokens: request.maxTokens }
      : {}),
    ...(topLogprobs !== undefined ? { logprobs: true, top_logprobs: topLogprobs } : {}),
    ...cleanUndefined(overrides),
  });
}

/**
 * Opt into usage reporting for streamed chat completions. OpenAI-compatible
 * servers only emit the `usage` block in streaming mode when
 * `stream_options.include_usage` is set, so without this the final chunk
 * carries no token counts. Any caller-provided `stream_options` win.
 */
function streamUsageOptions(request: LLMRequest): Record<string, unknown> {
  const userOptions = isRecord(request.body?.stream_options) ? request.body.stream_options : undefined;
  return { include_usage: true, ...userOptions };
}

async function completeWithChatCompletionsPassThrough(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
): Promise<LLMResponse> {
  const payload = await sendOpenAIJsonRequest(
    options,
    fetcher,
    path,
    request,
    buildChatCompletionsBody(options, request, {
      messages: buildMessages(request),
      stream: false,
    }),
    "Failed to parse OpenAI-compatible chat completion response",
  );
  const assistantMessage = pickAssistantMessage(payload);
  if (!assistantMessage) {
    throw new Error("No assistant message in OpenAI-compatible response.");
  }

  const toolCalls = pickChatToolCalls(payload);
  const reasoning = pickAssistantReasoning(payload);
  const logprobs = pickChatLogprobs(payload);
  return {
    text: pickAssistantText(payload),
    reasoning: reasoning.length > 0 ? reasoning : undefined,
    raw: payload,
    usage: pickUsage(payload),
    finishReason: pickFinishReason(payload),
    ...(logprobs ? { logprobs } : {}),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
  };
}

async function completeWithChatCompletionsWithMCP(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
): Promise<LLMResponse> {
  const maxToolRounds = normalizeMaxToolRounds(request.maxToolRounds ?? options.defaultMaxToolRounds);

  let messages = buildMessages(request);
  let aggregatedUsage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  const toolCalls: LLMToolCall[] = [];
  const toolExecutions: NonNullable<LLMResponse["toolExecutions"]> = [];
  const reasoningBlocks: ReasoningBlock[] = [];

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const transportTools = toProviderFunctionTools(mcpToolset);

    const payload = await sendOpenAIJsonRequest(
      options,
      fetcher,
      path,
      request,
      buildChatCompletionsBody(options, request, {
        messages,
        tools: transportTools,
        tool_choice: request.toolChoice,
        parallel_tool_calls: request.parallelToolCalls,
      }),
    );
    lastPayload = payload;
    aggregatedUsage = mergeUsage(aggregatedUsage, pickUsage(payload));
    finishReason = pickFinishReason(payload) ?? finishReason;

    const assistantMessage = pickAssistantMessage(payload);
    const calledTools = pickChatToolCalls(payload);
    const roundReasoning = pickAssistantReasoning(payload);
    pushReasoningBlock(reasoningBlocks, round, roundReasoning);

    if (!assistantMessage) {
      throw new Error("No assistant message in OpenAI-compatible response.");
    }

    if (calledTools.length === 0) {
      const reasoning = joinReasoningBlocks(reasoningBlocks) || undefined;
      return {
        text: pickAssistantText(payload),
        reasoning,
        reasoningBlocks: reasoningBlocks.length > 0 ? reasoningBlocks : undefined,
        raw: payload,
        usage: aggregatedUsage,
        finishReason,
        ...withPickedChatLogprobs(payload),
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
      };
    }

    if (round > maxToolRounds) {
      throw new Error(`Tool call loop exceeded maxToolRounds (${maxToolRounds}).`);
    }

    const outputs = await executeMCPToolCalls(calledTools, mcpToolset, {
      round,
      request,
      provider: "openai-compatible-legacy",
      model: options.model,
    });
    toolCalls.push(...outputs.map((entry) => entry.call));
    toolExecutions.push(...outputs.map((entry) => entry.execution));

    const toolMessages = outputs.map((entry) => ({
      role: "tool",
      tool_call_id: entry.call.id,
      content: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
    }));
    messages = [...messages, assistantMessage, ...toolMessages];
  }

  return {
    text: pickAssistantText(lastPayload ?? {}),
    reasoning: joinReasoningBlocks(reasoningBlocks) || undefined,
    reasoningBlocks: reasoningBlocks.length > 0 ? reasoningBlocks : undefined,
    raw: lastPayload,
    usage: aggregatedUsage,
    finishReason,
    ...(lastPayload ? withPickedChatLogprobs(lastPayload) : {}),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
  };
}

async function streamWithChatCompletionsWithMCP(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  callbacks: LLMStreamCallbacks,
): Promise<LLMResponse> {
  const maxToolRounds = normalizeMaxToolRounds(request.maxToolRounds ?? options.defaultMaxToolRounds);

  let messages = buildMessages(request);
  let aggregatedUsage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  const executedToolCalls: LLMToolCall[] = [];
  const toolExecutions: NonNullable<LLMResponse["toolExecutions"]> = [];
  const reasoningBlocks: ReasoningBlock[] = [];

  callbacks.onStart?.();
  let lastRoundText = "";

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const transportTools = toProviderFunctionTools(mcpToolset);

    const response = await sendOpenAIRequest(
      options,
      fetcher,
      path,
      request,
      buildChatCompletionsBody(options, request, {
        messages,
        tools: transportTools,
        tool_choice: request.toolChoice,
        parallel_tool_calls: request.parallelToolCalls,
        stream: true,
        stream_options: streamUsageOptions(request),
      }),
    );

    if (!response.ok) {
      const message = await readErrorBody(response);
      throw new Error(`HTTP ${response.status}: ${message}`);
    }

    let roundText = "";
    let roundReasoning = "";
    let roundUsage: LLMUsage | undefined;
    let roundFinishReason: string | undefined;
    let streamTerminated = false;
    const roundLogprobsContent: LLMTokenLogprob[] = [];
    const roundLogprobsRefusal: LLMTokenLogprob[] = [];
    const streamedToolCalls = new Map<number, OpenAIStreamToolCallState>();
    const nativeToolCalls = new NativeToolCallStreamState(true, `round_${round}`);
    let reasoningFieldName: OpenAIReasoningFieldName | undefined;

    await consumeSSE(response, (data) => {
      if (data === "[DONE]") {
        streamTerminated = true;
        return;
      }

      const json = safeJSONParse(data);
      if (!isRecord(json)) {
        throw new Error("Invalid JSON event in OpenAI Chat Completions stream.");
      }

      throwForChatCompletionsStreamError(json);
      lastPayload = json;

      const rawDelta = pickAssistantDelta(json);
      const reasoningDelta = pickAssistantReasoningDelta(json);
      const chunkUsage = pickUsage(json);
      const chunkFinishReason = pickFinishReason(json);
      const chunkLogprobs = pickChatLogprobs(json);
      appendChatLogprobs(chunkLogprobs, roundLogprobsContent, roundLogprobsRefusal);
      const nativeDelta = nativeToolCalls.push(rawDelta);
      const delta = nativeDelta.textDelta;

      collectOpenAIStreamToolCalls(json, streamedToolCalls);
      roundUsage = preferLatestUsage(roundUsage, chunkUsage);
      if (chunkFinishReason) {
        roundFinishReason = chunkFinishReason;
        streamTerminated = true;
      }

      if (delta) {
        roundText += delta;
        callbacks.onToken?.(delta);
      }

      if (reasoningDelta) {
        roundReasoning += reasoningDelta;
        reasoningFieldName ??= pickAssistantReasoningDeltaFieldName(json);
      }

      // Surface the accumulated tool-call snapshot on every chunk that carries a
      // tool-call delta, so consumers can stream partial arguments as they build
      // up (matching the text/reasoning deltas) instead of only seeing the full
      // call once the round completes. Standard OpenAI `tool_calls` deltas are
      // accumulated in `streamedToolCalls`; native `<tool_call>` markup deltas are
      // merged on top. Downstream dedups identical snapshots via the stream
      // fingerprint, so re-emitting the same snapshot is harmless.
      const streamedSnapshot = buildOpenAIStreamToolCalls(streamedToolCalls);
      const chunkToolCalls =
        nativeDelta.toolCalls.length > 0
          ? mergeToolCalls(streamedSnapshot, nativeDelta.toolCalls)
          : streamedSnapshot.length > 0
            ? streamedSnapshot
            : undefined;
      emitOpenAIStreamChunk(
        callbacks,
        round,
        json,
        delta,
        reasoningDelta,
        chunkUsage,
        chunkFinishReason,
        chunkToolCalls,
        chunkLogprobs,
      );
    });

    assertChatCompletionsStreamTerminated(streamTerminated);

    const tail = nativeToolCalls.flush();
    if (tail.textDelta) {
      roundText += tail.textDelta;
      callbacks.onToken?.(tail.textDelta);
      emitOpenAIStreamChunk(callbacks, round, {}, tail.textDelta, "", undefined, undefined);
    }
    aggregatedUsage = mergeUsage(aggregatedUsage, roundUsage);
    if (roundFinishReason) {
      finishReason = roundFinishReason;
    }

    const calledTools = mergeToolCalls(buildOpenAIStreamToolCalls(streamedToolCalls), nativeToolCalls.calls);
    pushReasoningBlock(reasoningBlocks, round, roundReasoning);
    request.onTurnTransition?.({
      turnIndex: round,
      kind: "reasoningComplete",
      reasoningText: roundReasoning,
    });

    if (calledTools.length === 0) {
      const out: LLMResponse = {
        text: roundText,
        reasoning: joinReasoningBlocks(reasoningBlocks) || undefined,
        reasoningBlocks: reasoningBlocks.length > 0 ? reasoningBlocks : undefined,
        raw: lastPayload,
        usage: aggregatedUsage,
        finishReason,
        ...withChatLogprobs(roundLogprobsContent, roundLogprobsRefusal),
        toolCalls: executedToolCalls.length > 0 ? executedToolCalls : undefined,
        toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
      };
      request.onTurnTransition?.({ turnIndex: round, kind: "streamEnd" });
      callbacks.onComplete?.(out);
      return out;
    }

    if (round > maxToolRounds) {
      throw new Error(`Tool call loop exceeded maxToolRounds (${maxToolRounds}).`);
    }

    request.onTurnTransition?.({
      turnIndex: round,
      kind: "toolCallsEmit",
      toolCalls: calledTools,
    });
    callbacks.onChunk?.({
      textDelta: "",
      turnIndex: round,
      toolCalls: calledTools,
      finishReason: roundFinishReason,
    });

    const outputs = await executeMCPToolCalls(calledTools, mcpToolset, {
      round,
      request,
      provider: "openai-compatible-legacy",
      model: options.model,
    });
    executedToolCalls.push(...outputs.map((entry) => entry.call));
    toolExecutions.push(...outputs.map((entry) => entry.execution));
    request.onTurnTransition?.({ turnIndex: round, kind: "toolResultsReceived" });

    lastRoundText = roundText;

    const assistantMessage = buildOpenAIAssistantToolMessage(roundText, calledTools, {
      reasoning: roundReasoning,
      reasoningFieldName,
    });
    const toolMessages = outputs.map((entry) => ({
      role: "tool",
      tool_call_id: entry.call.id,
      content: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
    }));
    messages = [...messages, assistantMessage, ...toolMessages];
  }

  const out: LLMResponse = {
    text: lastRoundText,
    reasoning: joinReasoningBlocks(reasoningBlocks) || undefined,
    reasoningBlocks: reasoningBlocks.length > 0 ? reasoningBlocks : undefined,
    raw: lastPayload,
    usage: aggregatedUsage,
    finishReason,
    toolCalls: executedToolCalls.length > 0 ? executedToolCalls : undefined,
    toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
  };
  request.onTurnTransition?.({ turnIndex: maxToolRounds + 1, kind: "streamEnd" });
  callbacks.onComplete?.(out);
  return out;
}

function buildMessages(request: LLMRequest): Array<Record<string, unknown>> {
  if (Array.isArray(request.messages) && request.messages.length > 0) {
    return request.messages.map((message) => toOpenAIMessage(message));
  }

  if (typeof request.prompt !== "string" || request.prompt.trim().length === 0) {
    throw new Error("LLMRequest must include a prompt or messages.");
  }

  const messages: Array<Record<string, unknown>> = [];
  if (request.systemPrompt) {
    messages.push({ role: "system", content: request.systemPrompt });
  }
  messages.push({ role: "user", content: request.prompt });
  return messages;
}

function toOpenAIMessage(message: LLMMessage): Record<string, unknown> {
  return { ...message };
}

function pickChatToolCalls(payload: Record<string, unknown>): LLMToolCall[] {
  const message = pickAssistantMessage(payload);
  if (!message) {
    return [];
  }

  const toolCalls = message.tool_calls;
  if (!Array.isArray(toolCalls)) {
    return [];
  }

  return toolCalls.map((entry) => {
    if (!isRecord(entry)) {
      return { id: "", type: "function" };
    }

    const fn = isRecord(entry.function) ? entry.function : undefined;
    return {
      id: pickString(entry.id) ?? "",
      type: pickString(entry.type) ?? "function",
      name: pickString(fn?.name),
      arguments: fn?.arguments,
    };
  });
}

/**
 * Reads `choices[0].logprobs` from a non-streaming chat-completion payload,
 * normalizing it to {@link LLMLogprobs}. Returns undefined when the response
 * carries no logprobs (the model was not asked for them), so callers can leave
 * the field off entirely.
 */
function pickChatLogprobs(payload: Record<string, unknown>): LLMLogprobs | undefined {
  const first = firstChoice(payload);
  if (!first || !isRecord(first.logprobs)) {
    return undefined;
  }
  const content = normalizeLogprobEntries(first.logprobs.content);
  const refusal = normalizeLogprobEntries(first.logprobs.refusal);
  if (!content && !refusal) {
    return undefined;
  }
  return {
    ...(content ? { content } : {}),
    ...(refusal ? { refusal } : {}),
  };
}

function withPickedChatLogprobs(payload: Record<string, unknown>): { logprobs?: LLMLogprobs } {
  const logprobs = pickChatLogprobs(payload);
  return logprobs ? { logprobs } : {};
}

/**
 * Appends a streaming chunk's content/refusal logprobs to the running
 * accumulators. Concatenating chunks reconstructs the full token sequences.
 */
function appendChatLogprobs(
  logprobs: LLMLogprobs | undefined,
  contentTarget: LLMTokenLogprob[],
  refusalTarget: LLMTokenLogprob[],
): void {
  if (!logprobs) {
    return;
  }
  if (logprobs.content) {
    contentTarget.push(...logprobs.content);
  }
  if (logprobs.refusal) {
    refusalTarget.push(...logprobs.refusal);
  }
}

function withChatLogprobs(
  content: LLMTokenLogprob[],
  refusal: LLMTokenLogprob[],
): { logprobs?: LLMLogprobs } {
  if (content.length === 0 && refusal.length === 0) {
    return {};
  }
  return {
    logprobs: {
      ...(content.length > 0 ? { content } : {}),
      ...(refusal.length > 0 ? { refusal } : {}),
    },
  };
}

function firstChoice(payload: Record<string, unknown>): Record<string, unknown> | undefined {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return undefined;
  }
  return isRecord(choices[0]) ? choices[0] : undefined;
}

function pickAssistantMessage(payload: Record<string, unknown>): Record<string, unknown> | undefined {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return undefined;
  }

  const first = choices[0];
  if (!isRecord(first)) {
    return undefined;
  }

  const message = first.message;
  if (!isRecord(message)) {
    return undefined;
  }

  return message;
}

type OpenAIReasoningFieldName = "reasoning" | "reasoning_content";

function pickAssistantDelta(payload: Record<string, unknown>): string {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return "";
  }

  const first = choices[0];
  if (!isRecord(first)) {
    return "";
  }

  const delta = first.delta;
  if (!isRecord(delta)) {
    return "";
  }

  const content = pickTextLike(delta.content);
  return content.length > 0 ? content : pickTextLike(delta.refusal);
}

function throwForChatCompletionsStreamError(payload: Record<string, unknown>): void {
  const error = isRecord(payload.error) ? payload.error : undefined;
  if (!error && pickString(payload.type) !== "error") {
    return;
  }

  throw new Error(pickString(error?.message) ?? pickString(payload.message) ?? "Chat Completions stream failed.");
}

function assertChatCompletionsStreamTerminated(terminated: boolean): void {
  if (!terminated) {
    throw new Error("OpenAI Chat Completions stream ended before a terminal event.");
  }
}

function pickAssistantReasoning(payload: Record<string, unknown>): string {
  const message = pickAssistantMessage(payload);
  if (!message) {
    return "";
  }

  return pickReasoningText(message);
}

function pickAssistantReasoningDelta(payload: Record<string, unknown>): string {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return "";
  }

  const first = choices[0];
  if (!isRecord(first)) {
    return "";
  }

  const delta = first.delta;
  if (!isRecord(delta)) {
    return "";
  }

  return pickReasoningText(delta);
}

function pickAssistantReasoningDeltaFieldName(payload: Record<string, unknown>): OpenAIReasoningFieldName | undefined {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return undefined;
  }

  const first = choices[0];
  if (!isRecord(first)) {
    return undefined;
  }

  const delta = first.delta;
  if (!isRecord(delta)) {
    return undefined;
  }

  if (hasTextLikeValue(delta.reasoning)) {
    return "reasoning";
  }

  if (hasTextLikeValue(delta.reasoning_content)) {
    return "reasoning_content";
  }

  return undefined;
}

interface OpenAIStreamToolCallState {
  index: number;
  id?: string;
  type?: string;
  name?: string;
  argumentsText: string;
}

interface ToolCallStreamDelta {
  textDelta: string;
  toolCalls: LLMToolCall[];
}

const NATIVE_TOOL_CALL_OPEN = "<tool_call";

const NATIVE_TOOL_CALL_CLOSE = "</tool_call>";

class NativeToolCallStreamState {
  readonly calls: LLMToolCall[] = [];
  private pending = "";

  // Native `<tool_call>` markup is only intercepted when the request actually
  // declares tools; otherwise the state is a transparent pass-through so prose
  // that happens to contain the literal markup is never swallowed.
  constructor(
    private readonly enabled = true,
    private readonly idNamespace?: string,
  ) {}

  push(delta: string): ToolCallStreamDelta {
    if (!delta || !this.enabled) {
      return { textDelta: delta, toolCalls: [] };
    }

    this.pending += delta;
    return this.drain(false);
  }

  flush(): ToolCallStreamDelta {
    return this.enabled ? this.drain(true) : { textDelta: "", toolCalls: [] };
  }

  private drain(flush: boolean): ToolCallStreamDelta {
    let textDelta = "";
    const toolCalls: LLMToolCall[] = [];

    while (this.pending.length > 0) {
      const openIndex = this.pending.indexOf(NATIVE_TOOL_CALL_OPEN);
      if (openIndex < 0) {
        const keep = flush ? 0 : nativeToolCallPrefixSuffixLength(this.pending);
        const emitLength = this.pending.length - keep;
        if (emitLength > 0) {
          textDelta += this.pending.slice(0, emitLength);
          this.pending = this.pending.slice(emitLength);
        }
        break;
      }

      if (openIndex > 0) {
        textDelta += this.pending.slice(0, openIndex);
        this.pending = this.pending.slice(openIndex);
        continue;
      }

      const closeIndex = this.pending.indexOf(NATIVE_TOOL_CALL_CLOSE);
      if (closeIndex < 0) {
        if (flush) {
          textDelta += this.pending;
          this.pending = "";
        }
        break;
      }

      const blockEnd = closeIndex + NATIVE_TOOL_CALL_CLOSE.length;
      const call = parseNativeToolCallBlock(
        this.pending.slice(0, blockEnd),
        this.fallbackId(this.calls.length),
      );
      if (call) {
        this.calls.push(call);
        toolCalls.push(call);
      }
      this.pending = this.pending.slice(blockEnd);
    }

    return { textDelta, toolCalls };
  }

  private fallbackId(index: number): string {
    return this.idNamespace
      ? `call_native_${this.idNamespace}_${index}`
      : `call_native_${index}`;
  }
}

function requestDeclaresTools(options: OpenAICompatibleAdapterOptions, request: LLMRequest): boolean {
  const hasTools = (body: Record<string, unknown> | undefined): boolean =>
    Array.isArray(body?.tools) && body.tools.length > 0;
  return hasTools(request.body) || hasTools(options.defaultBody);
}

const NATIVE_FUNCTION_PATTERN = /<function=([^>\s]+)\s*>([\s\S]*?)<\/function>/;

const NATIVE_PARAMETER_PATTERN = /<parameter=([^>\s]+)\s*>([\s\S]*?)<\/parameter>/g;

function parseNativeToolCallBlock(block: string, fallbackId: string): LLMToolCall | undefined {
  const inner = extractNativeToolCallInner(block);
  if (inner === undefined) {
    return undefined;
  }

  // Two shapes appear inside <tool_call>…</tool_call>: a JSON object (Hermes/Qwen
  // style) or a nested <function=…><parameter=…> tree (Llama style).
  return parseNativeJsonToolCall(inner, fallbackId) ?? parseNativeXmlToolCall(inner, fallbackId);
}

function extractNativeToolCallInner(block: string): string | undefined {
  const openEnd = block.indexOf(">");
  const closeStart = block.lastIndexOf(NATIVE_TOOL_CALL_CLOSE);
  if (openEnd < 0 || closeStart < 0 || closeStart <= openEnd) {
    return undefined;
  }

  return block.slice(openEnd + 1, closeStart).trim();
}

function parseNativeXmlToolCall(inner: string, fallbackId: string): LLMToolCall | undefined {
  const functionMatch = NATIVE_FUNCTION_PATTERN.exec(inner);
  const functionName = functionMatch?.[1];
  const functionBody = functionMatch?.[2];
  if (!functionName || functionBody === undefined) {
    return undefined;
  }

  const args: Record<string, unknown> = {};
  for (const [, key, rawValue] of functionBody.matchAll(NATIVE_PARAMETER_PATTERN)) {
    if (key && rawValue !== undefined) {
      args[key] = coerceNativeParameterValue(rawValue.trim());
    }
  }

  return {
    id: fallbackId,
    type: "function",
    name: functionName,
    arguments: JSON.stringify(args),
  };
}

/**
 * Parameter values arrive as untyped text. Decode them as JSON so numbers,
 * booleans, and nested objects keep their type, falling back to the raw string
 * when the value is not valid JSON.
 */
function coerceNativeParameterValue(value: string): unknown {
  if (value.length === 0) {
    return "";
  }

  const parsed = safeJSONParse(value);
  // safeJSONParse returns null both for the literal `null` and on parse failure;
  // only the literal should survive as null, everything else stays a string.
  if (parsed === null) {
    return value === "null" ? null : value;
  }

  return parsed;
}

function nativeToolCallPrefixSuffixLength(value: string): number {
  const max = Math.min(value.length, NATIVE_TOOL_CALL_OPEN.length - 1);
  for (let length = max; length > 0; length -= 1) {
    if (NATIVE_TOOL_CALL_OPEN.startsWith(value.slice(-length))) {
      return length;
    }
  }
  return 0;
}

function parseNativeJsonToolCall(inner: string, fallbackId: string): LLMToolCall | undefined {
  const parsed = safeJSONParse(inner);
  if (!isRecord(parsed)) {
    return undefined;
  }

  const name = pickString(parsed.name) ?? pickString(parsed.function);
  if (!name) {
    return undefined;
  }

  const rawArguments = parsed.arguments ?? parsed.parameters ?? {};
  const args = typeof rawArguments === "string" ? rawArguments : JSON.stringify(rawArguments);
  return {
    id: pickString(parsed.id) ?? fallbackId,
    type: "function",
    name,
    arguments: args,
  };
}

function mergeToolCalls(...groups: LLMToolCall[][]): LLMToolCall[] {
  const merged: LLMToolCall[] = [];
  const seen = new Set<string>();
  for (const group of groups) {
    for (const call of group) {
      const key = call.id || `${call.name ?? ""}:${String(call.arguments ?? "")}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      merged.push(call);
    }
  }
  return merged;
}

function collectOpenAIStreamToolCalls(
  payload: Record<string, unknown>,
  state: Map<number, OpenAIStreamToolCallState>,
): void {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return;
  }

  for (const choice of choices) {
    if (!isRecord(choice)) {
      continue;
    }

    const delta = isRecord(choice.delta) ? choice.delta : undefined;
    const message = isRecord(choice.message) ? choice.message : undefined;
    const toolCalls = Array.isArray(delta?.tool_calls)
      ? delta.tool_calls
      : Array.isArray(message?.tool_calls)
        ? message.tool_calls
        : Array.isArray(choice.tool_calls)
          ? choice.tool_calls
          : undefined;
    if (!toolCalls) {
      continue;
    }

    for (const rawToolCall of toolCalls) {
      if (!isRecord(rawToolCall)) {
        continue;
      }

      const index = toFiniteNumber(rawToolCall.index);
      const toolIndex = index !== undefined ? Math.floor(index) : state.size;
      const existing = state.get(toolIndex) ?? {
        index: toolIndex,
        argumentsText: "",
      };

      const id = pickString(rawToolCall.id);
      if (id) {
        existing.id = id;
      }

      const type = pickString(rawToolCall.type);
      if (type) {
        existing.type = type;
      }

      const functionCall = isRecord(rawToolCall.function) ? rawToolCall.function : undefined;
      const name = pickString(functionCall?.name);
      if (name) {
        existing.name = `${existing.name ?? ""}${name}`;
      }

      const argumentsDelta = pickString(functionCall?.arguments);
      if (argumentsDelta) {
        if (message?.tool_calls === toolCalls || choice.tool_calls === toolCalls) {
          existing.argumentsText = argumentsDelta;
        } else {
          existing.argumentsText += argumentsDelta;
        }
      }

      state.set(toolIndex, existing);
    }
  }
}

function buildOpenAIStreamToolCalls(state: Map<number, OpenAIStreamToolCallState>): LLMToolCall[] {
  return [...state.values()]
    .sort((a, b) => a.index - b.index)
    .map((entry) => ({
      id: entry.id ?? "",
      type: entry.type ?? "function",
      name: entry.name,
      arguments: entry.argumentsText.length > 0 ? entry.argumentsText : {},
    }));
}

function buildOpenAIAssistantToolMessage(
  text: string,
  toolCalls: LLMToolCall[],
  reasoning?: {
    reasoning?: string;
    reasoningFieldName?: OpenAIReasoningFieldName;
  },
): Record<string, unknown> {
  const message: Record<string, unknown> = {
    role: "assistant",
    content: text,
    tool_calls: toolCalls.map((call) => ({
      id: call.id,
      type: "function",
      function: {
        name: call.name,
        arguments: typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments ?? {}),
      },
    })),
  };

  if (reasoning?.reasoning && reasoning.reasoning.length > 0) {
    message[reasoning.reasoningFieldName ?? "reasoning"] = reasoning.reasoning;
  }

  return message;
}

function pickAssistantText(payload: Record<string, unknown>): string {
  const message = pickAssistantMessage(payload);
  if (message) {
    const text = pickTextLike(message.content);
    if (text.length > 0) {
      return text;
    }

    const refusal = pickTextLike(message.refusal);
    if (refusal.length > 0) {
      return refusal;
    }
  }

  const choices = payload.choices;
  if (Array.isArray(choices) && choices.length > 0 && isRecord(choices[0])) {
    const legacyText = choices[0].text;
    if (typeof legacyText === "string") {
      return legacyText;
    }
  }

  return "";
}

function hasTextLikeValue(value: unknown): boolean {
  return pickTextLike(value).length > 0;
}

function pickFinishReason(payload: Record<string, unknown>): string | undefined {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0 || !isRecord(choices[0])) {
    return undefined;
  }

  const reason = choices[0].finish_reason;
  return typeof reason === "string" && reason.length > 0 ? reason : undefined;
}
