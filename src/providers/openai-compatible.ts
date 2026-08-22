import type {
  EmbeddingRequest,
  EmbeddingResult,
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
} from "./utils";
import {
  embedOpenAI,
  emitOpenAIStreamChunk,
  joinReasoningBlocks,
  normalizeLogprobEntries,
  pickReasoningText,
  pickTextLike,
  pickTextLikePart,
  pickUsage,
  pushReasoningBlock,
  sendOpenAIJsonRequest,
  sendOpenAIRequest,
  validateTopLogprobs,
} from "./openai-compatible-common";
import { toOpenAIReasoningEffort } from "./reasoning-effort";
import type { OpenAICompatibleAdapterOptions } from "./openai-compatible-common";

interface OpenAIResponsesMCPState {
  input: ReturnType<typeof buildResponsesInput>;
  previousResponseId: string | undefined;
  aggregatedUsage: LLMUsage | undefined;
  finishReason: string | undefined;
  lastPayload: Record<string, unknown> | undefined;
  executedToolCalls: LLMToolCall[];
  toolExecutions: NonNullable<LLMResponse["toolExecutions"]>;
  reasoningBlocks: ReasoningBlock[];
}

export function createOpenAICompatibleAdapter(options: OpenAICompatibleAdapterOptions): LLMAdapter {
  const fetcher = options.fetcher ?? fetch;
  const path = options.path ?? "/v1/responses";
  const embeddingPath = options.embeddingPath ?? "/v1/embeddings";

  return {
    provider: "openai-compatible",
    model: options.model,

    async complete(request: LLMRequest): Promise<LLMResponse> {
      return hasMCPClients(request.mcpClients)
        ? completeWithResponsesAPIWithMCP(options, fetcher, path, request)
        : completeWithResponsesAPIPassThrough(options, fetcher, path, request);
    },

    async stream(request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
      return hasMCPClients(request.mcpClients)
        ? streamWithResponsesAPIWithMCP(options, fetcher, path, request, callbacks)
        : streamWithResponsesAPIPassThrough(options, fetcher, path, request, callbacks);
    },

    async embed(request: EmbeddingRequest): Promise<EmbeddingResult> {
      return embedOpenAI(options, fetcher, embeddingPath, request);
    },
  };
}

function buildResponsesBody(
  options: OpenAICompatibleAdapterOptions,
  request: LLMRequest,
  overrides: Record<string, unknown>,
): Record<string, unknown> {
  const topLogprobs = validateTopLogprobs(request.topLogprobs);
  const configuredReasoning = request.body?.reasoning ?? options.defaultBody?.reasoning;
  const bodyReasoning = isRecord(configuredReasoning) ? configuredReasoning : undefined;
  const effort = toOpenAIReasoningEffort(request.reasoningEffort);
  const configuredInclude = request.body?.include ?? options.defaultBody?.include;
  return cleanUndefined({
    ...options.defaultBody,
    ...request.body,
    model: options.model,
    ...(request.temperature !== undefined ? { temperature: request.temperature } : {}),
    ...(request.systemPrompt !== undefined ? { instructions: request.systemPrompt } : {}),
    ...(effort ? { reasoning: withResponsesReasoningSummary({ ...bodyReasoning, effort }) } : {}),
    ...(request.maxTokens !== undefined ? { max_output_tokens: request.maxTokens } : {}),
    ...(topLogprobs !== undefined
      ? { top_logprobs: topLogprobs, include: withResponsesLogprobsInclude(configuredInclude) }
      : {}),
    ...cleanUndefined(overrides),
  });
}

/**
 * The Responses API returns reasoning items only for a request that asks for a
 * summary, so requesting an effort implies `summary: "auto"`: without it the
 * thinking the effort was enabled for never reaches the response at all. The
 * default keys off the presence of `summary`, not its value, so a caller keeps
 * every way out: another summary level, `null` to ask for none, or `undefined`
 * to drop the field from the payload for an endpoint that rejects it.
 */
function withResponsesReasoningSummary(reasoning: Record<string, unknown>): Record<string, unknown> {
  // Effort `none` produces no reasoning at all, so there is nothing to summarize
  // and asking for one is a contradiction some endpoints reject.
  if (reasoning.effort === "none" || "summary" in reasoning) {
    return reasoning;
  }
  return { ...reasoning, summary: "auto" };
}

const RESPONSES_LOGPROBS_INCLUDE = "message.output_text.logprobs";

/**
 * The Responses API only returns logprobs when the response payload explicitly
 * includes them, so requesting `topLogprobs` implies the `include` entry.
 */
function withResponsesLogprobsInclude(configured: unknown): unknown[] {
  const base = Array.isArray(configured) ? configured : [];
  return base.includes(RESPONSES_LOGPROBS_INCLUDE) ? base : [...base, RESPONSES_LOGPROBS_INCLUDE];
}

function createResponsesMCPState(request: LLMRequest): OpenAIResponsesMCPState {
  return {
    input: buildResponsesInput(request),
    previousResponseId: pickString(
      isRecord(request.body) ? (request.body.previous_response_id as unknown) : undefined,
    ),
    aggregatedUsage: undefined,
    finishReason: undefined,
    lastPayload: undefined,
    executedToolCalls: [],
    toolExecutions: [],
    reasoningBlocks: [],
  };
}

function buildResponsesMCPResult(
  state: OpenAIResponsesMCPState,
  text: string,
  raw: Record<string, unknown> | undefined,
  streamedLogprobs?: LLMTokenLogprob[],
): LLMResponse {
  return {
    text,
    reasoning: joinReasoningBlocks(state.reasoningBlocks) || undefined,
    reasoningBlocks: state.reasoningBlocks.length > 0 ? state.reasoningBlocks : undefined,
    raw,
    usage: state.aggregatedUsage,
    finishReason: state.finishReason,
    ...(streamedLogprobs && streamedLogprobs.length > 0
      ? { logprobs: { content: streamedLogprobs } }
      : withPickedResponsesLogprobs(raw)),
    toolCalls: state.executedToolCalls.length > 0 ? state.executedToolCalls : undefined,
    toolExecutions: state.toolExecutions.length > 0 ? state.toolExecutions : undefined,
  };
}

function buildResponsesMCPRoundBody(
  options: OpenAICompatibleAdapterOptions,
  request: LLMRequest,
  state: OpenAIResponsesMCPState,
  transportTools: Array<Record<string, unknown>> | undefined,
  extraOverrides?: Record<string, unknown>,
): Record<string, unknown> {
  return buildResponsesBody(options, request, {
    input: state.input,
    previous_response_id: state.previousResponseId,
    tools: transportTools,
    tool_choice: toResponsesToolChoice(request.toolChoice),
    parallel_tool_calls: request.parallelToolCalls,
    ...extraOverrides,
  });
}

async function completeWithResponsesAPIPassThrough(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
): Promise<LLMResponse> {
  const body = isRecord(request.body) ? request.body : undefined;
  const payload = await sendOpenAIJsonRequest(
    options,
    fetcher,
    path,
    request,
    buildResponsesBody(options, request, {
      input: buildResponsesInput(request),
      previous_response_id: pickString(body?.previous_response_id),
    }),
  );
  throwForResponsesPayloadError(payload);
  const toolCalls = pickResponsesToolCalls(payload);
  const reasoning = pickResponsesReasoning(payload);
  return {
    text: pickResponsesText(payload),
    reasoning: reasoning || undefined,
    raw: payload,
    usage: pickUsage(payload),
    finishReason: pickResponsesFinishReason(payload),
    ...withPickedResponsesLogprobs(payload),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
  };
}

async function completeWithResponsesAPIWithMCP(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
): Promise<LLMResponse> {
  const maxToolRounds = normalizeMaxToolRounds(request.maxToolRounds ?? options.defaultMaxToolRounds);
  const state = createResponsesMCPState(request);

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const transportTools = toResponsesTools(toProviderFunctionTools(mcpToolset));

    const payload = await sendOpenAIJsonRequest(
      options,
      fetcher,
      path,
      request,
      buildResponsesMCPRoundBody(options, request, state, transportTools),
    );
    throwForResponsesPayloadError(payload);
    state.lastPayload = payload;
    state.aggregatedUsage = mergeUsage(state.aggregatedUsage, pickUsage(payload));
    state.finishReason = pickResponsesFinishReason(payload) ?? state.finishReason;
    pushReasoningBlock(state.reasoningBlocks, round, pickResponsesReasoning(payload));

    const providerToolCalls = pickResponsesToolCalls(payload);
    const functionCalls = providerToolCalls.filter(
      (toolCall): toolCall is LLMToolCall & { id: string; name: string } =>
        toolCall.type === "function" && typeof toolCall.id === "string" && typeof toolCall.name === "string",
    );

    if (functionCalls.length === 0) {
      const text = pickResponsesText(payload);
      return buildResponsesMCPResult(state, text, payload);
    }

    if (round > maxToolRounds) {
      throw new Error(`Tool call loop exceeded maxToolRounds (${maxToolRounds}).`);
    }

    const outputs = await executeMCPToolCalls(functionCalls, mcpToolset, {
      round,
      request,
      provider: "openai-compatible",
      model: options.model,
    });
    state.executedToolCalls.push(...outputs.map((entry) => entry.call));
    state.toolExecutions.push(...outputs.map((entry) => entry.execution));

    state.input = [
      ...normalizeResponsesInput(state.input),
      ...pickResponsesRoundOutputItems(payload, functionCalls),
      ...outputs.map((entry) => ({
        type: "function_call_output",
        call_id: entry.call.id,
        output: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
      })),
    ];
  }

  return buildResponsesMCPResult(
    state,
    pickResponsesText(state.lastPayload ?? {}),
    state.lastPayload,
  );
}

async function streamWithResponsesAPIPassThrough(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  callbacks: LLMStreamCallbacks,
): Promise<LLMResponse> {
  const body = isRecord(request.body) ? request.body : undefined;
  const response = await sendOpenAIRequest(
    options,
    fetcher,
    path,
    request,
    buildResponsesBody(options, request, {
      input: buildResponsesInput(request),
      previous_response_id: pickString(body?.previous_response_id),
      stream: true,
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
  const streamedToolCalls = new Map<string, OpenAIResponsesStreamToolCallState>();
  const logprobsContent: LLMTokenLogprob[] = [];

  await consumeSSE(response, (data) => {
    if (data === "[DONE]") {
      streamTerminated = true;
      return;
    }

    const json = safeJSONParse(data);
    if (!isRecord(json)) {
      throw new Error("Invalid JSON event in OpenAI Responses stream.");
    }

    const roundPayload = pickResponsesStreamPayload(json);
    if (roundPayload) {
      lastPayload = roundPayload;
    }

    const delta = pickResponsesStreamTextDelta(json);
    const reasoningDelta = pickResponsesStreamReasoningDelta(json);
    const chunkUsage = pickResponsesStreamUsage(json);
    const chunkFinishReason = pickResponsesStreamFinishReason(json);
    const chunkLogprobs = pickResponsesStreamLogprobs(json);
    appendResponsesLogprobs(chunkLogprobs, logprobsContent);
    throwForResponsesStreamError(json);
    streamTerminated ||= isResponsesTerminalEvent(json);
    collectResponsesStreamToolCalls(json, streamedToolCalls);
    const chunkToolCalls = buildResponsesStreamToolCalls(streamedToolCalls);

    usage = preferLatestUsage(usage, chunkUsage);
    if (chunkFinishReason) {
      finishReason = chunkFinishReason;
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

  assertResponsesStreamTerminated(streamTerminated);

  const finalPayload = lastPayload ?? {};
  const finalReasoning = reasoning || pickResponsesReasoning(finalPayload);
  const toolCalls = buildResponsesStreamToolCalls(streamedToolCalls);
  const out: LLMResponse = {
    text: text.length > 0 ? text : pickResponsesText(finalPayload),
    reasoning: finalReasoning || undefined,
    raw: finalPayload,
    usage: preferLatestUsage(usage, pickUsage(finalPayload)),
    finishReason: finishReason ?? pickResponsesFinishReason(finalPayload),
    ...(logprobsContent.length > 0
      ? { logprobs: { content: logprobsContent } }
      : withPickedResponsesLogprobs(finalPayload)),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
  };
  callbacks.onComplete?.(out);
  return out;
}

async function streamWithResponsesAPIWithMCP(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  callbacks: LLMStreamCallbacks,
): Promise<LLMResponse> {
  const maxToolRounds = normalizeMaxToolRounds(request.maxToolRounds ?? options.defaultMaxToolRounds);
  const state = createResponsesMCPState(request);

  callbacks.onStart?.();

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const transportTools = toResponsesTools(toProviderFunctionTools(mcpToolset));

    const response = await sendOpenAIRequest(
      options,
      fetcher,
      path,
      request,
      buildResponsesMCPRoundBody(options, request, state, transportTools, { stream: true }),
    );

    if (!response.ok) {
      const message = await readErrorBody(response);
      throw new Error(`HTTP ${response.status}: ${message}`);
    }

    let roundText = "";
    let roundReasoning = "";
    let roundUsage: LLMUsage | undefined;
    let roundFinishReason: string | undefined;
    let roundPayload: Record<string, unknown> | undefined;
    let streamTerminated = false;
    const roundLogprobs: LLMTokenLogprob[] = [];
    const streamedToolCalls = new Map<string, OpenAIResponsesStreamToolCallState>();

    await consumeSSE(response, (data) => {
      if (data === "[DONE]") {
        streamTerminated = true;
        return;
      }

      const json = safeJSONParse(data);
      if (!isRecord(json)) {
        throw new Error("Invalid JSON event in OpenAI Responses stream.");
      }

      const payload = pickResponsesStreamPayload(json);
      if (payload) {
        roundPayload = payload;
        state.lastPayload = payload;
      }

      const delta = pickResponsesStreamTextDelta(json);
      const reasoningDelta = pickResponsesStreamReasoningDelta(json);
      const chunkUsage = pickResponsesStreamUsage(json);
      const chunkFinishReason = pickResponsesStreamFinishReason(json);
      const chunkLogprobs = pickResponsesStreamLogprobs(json);
      appendResponsesLogprobs(chunkLogprobs, roundLogprobs);
      throwForResponsesStreamError(json);
      streamTerminated ||= isResponsesTerminalEvent(json);

      collectResponsesStreamToolCalls(json, streamedToolCalls);
      roundUsage = preferLatestUsage(roundUsage, chunkUsage);
      if (chunkFinishReason) {
        roundFinishReason = chunkFinishReason;
      }

      if (delta) {
        roundText += delta;
        callbacks.onToken?.(delta);
      }

      if (reasoningDelta) {
        roundReasoning += reasoningDelta;
      }

      // Stream the accumulated tool-call snapshot per chunk (same contract as
      // openai-compatible-legacy.ts), so partial arguments surface as they
      // build up.
      const chunkToolCalls = buildResponsesStreamToolCalls(streamedToolCalls);
      emitOpenAIStreamChunk(
        callbacks,
        round,
        json,
        delta,
        reasoningDelta,
        chunkUsage,
        chunkFinishReason,
        chunkToolCalls.length > 0 ? chunkToolCalls : undefined,
        chunkLogprobs,
      );
    });

    assertResponsesStreamTerminated(streamTerminated);

    const resolvedRoundUsage = preferLatestUsage(roundUsage, roundPayload ? pickUsage(roundPayload) : undefined);
    state.aggregatedUsage = mergeUsage(state.aggregatedUsage, resolvedRoundUsage);
    if (roundFinishReason) {
      state.finishReason = roundFinishReason;
    } else if (roundPayload) {
      state.finishReason = pickResponsesFinishReason(roundPayload) ?? state.finishReason;
    }

    const payloadToolCalls = roundPayload ? pickResponsesToolCalls(roundPayload) : [];
    if (roundPayload && roundReasoning.length === 0) {
      roundReasoning = pickResponsesReasoning(roundPayload);
    }
    const streamedCalls = buildResponsesStreamToolCalls(streamedToolCalls);
    const providerToolCalls = payloadToolCalls.length > 0 ? payloadToolCalls : streamedCalls;
    const functionCalls = providerToolCalls.filter(
      (toolCall): toolCall is LLMToolCall & { id: string; name: string } =>
        toolCall.type === "function" && typeof toolCall.id === "string" && typeof toolCall.name === "string",
    );
    pushReasoningBlock(state.reasoningBlocks, round, roundReasoning);
    request.onTurnTransition?.({
      turnIndex: round,
      kind: "reasoningComplete",
      reasoningText: roundReasoning,
    });

    if (functionCalls.length === 0) {
      const finalText = roundText.length > 0
        ? roundText
        : (roundPayload ? pickResponsesText(roundPayload) : "");
      const out = buildResponsesMCPResult(state, finalText, roundPayload ?? state.lastPayload, roundLogprobs);
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
      toolCalls: functionCalls,
    });
    callbacks.onChunk?.({
      textDelta: "",
      turnIndex: round,
      toolCalls: functionCalls,
      finishReason: roundFinishReason,
    });

    const outputs = await executeMCPToolCalls(functionCalls, mcpToolset, {
      round,
      request,
      provider: "openai-compatible",
      model: options.model,
    });
    state.executedToolCalls.push(...outputs.map((entry) => entry.call));
    state.toolExecutions.push(...outputs.map((entry) => entry.execution));
    request.onTurnTransition?.({ turnIndex: round, kind: "toolResultsReceived" });

    state.input = [
      ...normalizeResponsesInput(state.input),
      ...pickResponsesRoundOutputItems(roundPayload, functionCalls),
      ...outputs.map((entry) => ({
        type: "function_call_output",
        call_id: entry.call.id,
        output: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
      })),
    ];
  }

  const out = buildResponsesMCPResult(
    state,
    pickResponsesText(state.lastPayload ?? {}),
    state.lastPayload,
  );
  request.onTurnTransition?.({ turnIndex: maxToolRounds + 1, kind: "streamEnd" });
  callbacks.onComplete?.(out);
  return out;
}

function buildResponsesInput(request: LLMRequest): unknown {
  if (isRecord(request.body) && "input" in request.body) {
    return request.body.input;
  }

  if (Array.isArray(request.messages) && request.messages.length > 0) {
    return request.messages.flatMap((message) => toResponsesItems(message));
  }

  if (typeof request.prompt !== "string" || request.prompt.trim().length === 0) {
    throw new Error("LLMRequest must include a prompt or messages.");
  }
  return request.prompt;
}

/**
 * Maps one Chat Completions-shaped {@link LLMMessage} to Responses API input
 * items. `role: "tool"` messages and assistant `tool_calls` have no message
 * equivalent in the Responses API — they must become `function_call_output`
 * and `function_call` items, so one message can expand to several items.
 */
function toResponsesItems(message: LLMMessage): Array<Record<string, unknown>> {
  if (message.role === "tool") {
    return [{
      type: "function_call_output",
      call_id: pickString(message.tool_call_id) ?? "",
      output: typeof message.content === "string" ? message.content : JSON.stringify(message.content ?? null),
    }];
  }

  if (message.role === "assistant" && Array.isArray(message.tool_calls)) {
    const items: Array<Record<string, unknown>> = [];
    if (
      (typeof message.content === "string" && message.content.length > 0)
      || (Array.isArray(message.content) && message.content.length > 0)
    ) {
      items.push(toResponsesMessage({ role: "assistant", content: message.content }));
    }
    for (const toolCall of message.tool_calls) {
      if (!isRecord(toolCall) || !isRecord(toolCall.function)) {
        continue;
      }
      items.push({
        type: "function_call",
        call_id: pickString(toolCall.id) ?? "",
        name: pickString(toolCall.function.name),
        arguments: typeof toolCall.function.arguments === "string"
          ? toolCall.function.arguments
          : JSON.stringify(toolCall.function.arguments ?? {}),
      });
    }
    return items;
  }

  return [toResponsesMessage(message)];
}

function toResponsesMessage(message: LLMMessage): Record<string, unknown> {
  if (!Array.isArray(message.content)) {
    return { ...message };
  }

  return {
    ...message,
    content: message.content.map((part) => {
      if (part.type === "image_url") {
        return { type: "input_image", image_url: part.image_url.url };
      }
      return {
        type: "input_text",
        text: part.text,
      };
    }),
  };
}

function toResponsesTools(tools: Array<Record<string, unknown>> | undefined): Array<Record<string, unknown>> | undefined {
  if (!Array.isArray(tools) || tools.length === 0) {
    return undefined;
  }

  return tools.map((tool) => {
    if (tool.type === "function" && isRecord(tool.function)) {
      const functionTool = tool.function;
      return {
        type: "function",
        name: functionTool.name,
        description: functionTool.description,
        parameters: functionTool.parameters,
        strict: functionTool.strict,
      };
    }

    return { ...tool };
  });
}

function toResponsesToolChoice(choice: LLMRequest["toolChoice"]): LLMRequest["toolChoice"] {
  if (!isRecord(choice) || choice.type !== "function" || !isRecord(choice.function)) {
    return choice;
  }
  return {
    type: "function",
    name: choice.function.name,
  };
}

function pickResponsesLogprobs(payload: Record<string, unknown>): LLMLogprobs | undefined {
  const content: LLMTokenLogprob[] = [];
  const output = payload.output;
  if (Array.isArray(output)) {
    for (const item of output) {
      if (!isRecord(item)) continue;
      const parts = Array.isArray(item.content) ? item.content : [];
      for (const part of parts) {
        if (!isRecord(part)) continue;
        const entries = normalizeLogprobEntries(part.logprobs);
        if (entries) content.push(...entries);
      }
    }
  }
  return content.length > 0 ? { content } : undefined;
}

function withPickedResponsesLogprobs(
  payload: Record<string, unknown> | undefined,
): { logprobs?: LLMLogprobs } {
  if (!payload) return {};
  const logprobs = pickResponsesLogprobs(payload);
  return logprobs ? { logprobs } : {};
}

function pickResponsesStreamLogprobs(payload: Record<string, unknown>): LLMLogprobs | undefined {
  const eventType = pickString(payload.type) ?? "";
  if (!eventType.includes("output_text")) return undefined;
  const content = normalizeLogprobEntries(payload.logprobs);
  return content ? { content } : undefined;
}

function appendResponsesLogprobs(logprobs: LLMLogprobs | undefined, target: LLMTokenLogprob[]): void {
  if (logprobs?.content) target.push(...logprobs.content);
}

function normalizeResponsesInput(input: unknown): unknown[] {
  if (Array.isArray(input)) return [...input];
  if (input === undefined || input === null) return [];
  // A bare string is valid as the whole `input`, but not as an item of the
  // input array, so it becomes an explicit user message before tool-round
  // items are appended after it.
  if (typeof input === "string") return [{ role: "user", content: input }];
  return [input];
}

function pickResponsesOutputItems(payload: Record<string, unknown> | undefined): unknown[] {
  return Array.isArray(payload?.output) ? payload.output : [];
}

function pickResponsesRoundOutputItems(
  payload: Record<string, unknown> | undefined,
  functionCalls: LLMToolCall[],
): unknown[] {
  const output = pickResponsesOutputItems(payload);
  if (output.length > 0) return output;
  return functionCalls.map((call) => ({
    type: "function_call",
    call_id: call.id,
    name: call.name,
    arguments: typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments ?? {}),
  }));
}

function throwForResponsesStreamError(payload: Record<string, unknown>): void {
  const eventType = pickString(payload.type);
  if (eventType !== "error" && eventType !== "response.failed") return;
  const error = isRecord(payload.error)
    ? payload.error
    : isRecord(payload.response) && isRecord(payload.response.error)
      ? payload.response.error
      : undefined;
  const message = pickString(error?.message) ?? pickString(payload.message) ?? "Responses stream failed.";
  throw new Error(message);
}

function isResponsesTerminalEvent(payload: Record<string, unknown>): boolean {
  const eventType = pickString(payload.type);
  return eventType === "response.completed" || eventType === "response.incomplete";
}

function assertResponsesStreamTerminated(terminated: boolean): void {
  if (!terminated) {
    throw new Error("OpenAI Responses stream ended before a terminal event.");
  }
}

function throwForResponsesPayloadError(payload: Record<string, unknown>): void {
  if (pickString(payload.status) !== "failed") return;
  const error = isRecord(payload.error) ? payload.error : undefined;
  throw new Error(pickString(error?.message) ?? "Responses request failed.");
}

function pickResponsesToolCalls(payload: Record<string, unknown>): LLMToolCall[] {
  const output = payload.output;
  if (!Array.isArray(output)) {
    return [];
  }

  const calls: LLMToolCall[] = [];

  for (const item of output) {
    if (!isRecord(item)) {
      continue;
    }

    const type = pickString(item.type);
    if (type === "function_call") {
      calls.push({
        id: pickString(item.call_id) ?? pickString(item.id) ?? "",
        type: "function",
        name: pickString(item.name),
        arguments: item.arguments,
      });
      continue;
    }

    if (type?.includes("mcp") || type?.includes("tool")) {
      calls.push({
        id: pickString(item.id) ?? pickString(item.call_id) ?? "",
        type,
        name: pickString(item.name),
        arguments: item.arguments,
      });
    }
  }

  return calls;
}

interface OpenAIResponsesStreamToolCallState {
  key: string;
  id?: string;
  type?: string;
  name?: string;
  argumentsText: string;
}

function pickResponsesStreamPayload(payload: Record<string, unknown>): Record<string, unknown> | undefined {
  if (isRecord(payload.response)) {
    return payload.response;
  }

  if ("output" in payload || "output_text" in payload || "status" in payload || "id" in payload) {
    return payload;
  }

  return undefined;
}

function pickResponsesStreamTextDelta(payload: Record<string, unknown>): string {
  const eventType = pickString(payload.type) ?? "";
  if (!eventType.includes("output_text.delta") && !eventType.includes("refusal.delta")) {
    return "";
  }

  const direct = pickString(payload.delta);
  if (direct) {
    return direct;
  }

  if (isRecord(payload.delta)) {
    return pickString(payload.delta.text) ?? pickString(payload.delta.output_text) ?? "";
  }

  return "";
}

function pickResponsesStreamReasoningDelta(payload: Record<string, unknown>): string {
  const eventType = pickString(payload.type) ?? "";
  if (!eventType.includes("reasoning") && !eventType.includes("thinking")) {
    return "";
  }

  const direct = pickString(payload.delta);
  if (direct) {
    return direct;
  }

  if (isRecord(payload.delta)) {
    return pickReasoningText(payload.delta)
      || pickString(payload.delta.text)
      || pickString(payload.delta.summary_text)
      || "";
  }

  return "";
}

function pickResponsesStreamUsage(payload: Record<string, unknown>): LLMUsage | undefined {
  const direct = pickUsage(payload);
  if (direct) {
    return direct;
  }

  if (isRecord(payload.response)) {
    return pickUsage(payload.response);
  }

  return undefined;
}

function pickResponsesStreamFinishReason(payload: Record<string, unknown>): string | undefined {
  const eventType = pickString(payload.type);
  if (eventType === "response.completed") {
    return "completed";
  }
  if (eventType === "response.failed") {
    return "failed";
  }
  if (eventType === "response.incomplete") {
    return "incomplete";
  }

  const status = pickString(payload.status)
    ?? (isRecord(payload.response) ? pickString(payload.response.status) : undefined);
  // Lifecycle events such as `response.created` carry a non-terminal status;
  // surfacing it as a finishReason would make consumers treat the very first
  // chunk as the end of the stream.
  return status && status !== "in_progress" && status !== "queued" ? status : undefined;
}

function collectResponsesStreamToolCalls(
  payload: Record<string, unknown>,
  state: Map<string, OpenAIResponsesStreamToolCallState>,
): void {
  if (isRecord(payload.response)) {
    collectResponsesStreamToolCallsFromOutput(payload.response.output, state);
  }
  collectResponsesStreamToolCallsFromOutput(payload.output, state);

  if (isRecord(payload.item)) {
    const itemKey = pickString(payload.item_id) ?? pickString(payload.call_id);
    collectResponsesStreamToolCallsFromItem(payload.item, state, itemKey);
  }

  if (isRecord(payload.output_item)) {
    const itemKey = pickString(payload.item_id) ?? pickString(payload.call_id);
    collectResponsesStreamToolCallsFromItem(payload.output_item, state, itemKey);
  }

  const eventType = pickString(payload.type) ?? "";
  if (eventType.includes("function_call_arguments.delta")) {
    const key = pickString(payload.item_id) ?? pickString(payload.call_id) ?? "function_call";
    const existing = state.get(key) ?? {
      key,
      argumentsText: "",
    };
    const delta = pickString(payload.delta)
      ?? (isRecord(payload.delta) ? pickString(payload.delta.text) ?? pickString(payload.delta.arguments) : undefined)
      ?? pickString(payload.arguments_delta);
    if (delta) {
      existing.argumentsText += delta;
    }
    state.set(key, existing);
  }

  if (eventType.includes("function_call_arguments.done")) {
    const key = pickString(payload.item_id) ?? pickString(payload.call_id) ?? "function_call";
    const existing = state.get(key) ?? { key, argumentsText: "" };
    const argumentsText = pickString(payload.arguments);
    if (argumentsText) existing.argumentsText = argumentsText;
    state.set(key, existing);
  }
}

function collectResponsesStreamToolCallsFromOutput(
  output: unknown,
  state: Map<string, OpenAIResponsesStreamToolCallState>,
): void {
  if (!Array.isArray(output)) {
    return;
  }

  for (const item of output) {
    if (!isRecord(item)) {
      continue;
    }
    collectResponsesStreamToolCallsFromItem(item, state);
  }
}

function collectResponsesStreamToolCallsFromItem(
  item: Record<string, unknown>,
  state: Map<string, OpenAIResponsesStreamToolCallState>,
  forcedKey?: string,
): void {
  const type = pickString(item.type);
  if (type !== "function_call" && !type?.includes("tool") && !type?.includes("mcp")) {
    return;
  }

  const key = forcedKey ?? pickString(item.id) ?? pickString(item.call_id) ?? `call_${state.size}`;
  const existing = state.get(key) ?? {
    key,
    argumentsText: "",
  };

  const callId = pickString(item.call_id) ?? pickString(item.id);
  if (callId) {
    existing.id = callId;
  }

  if (type) {
    existing.type = type;
  }

  const name = pickString(item.name);
  if (name) {
    existing.name = name;
  }

  const argumentsText = pickString(item.arguments);
  if (argumentsText && argumentsText.length >= existing.argumentsText.length) {
    existing.argumentsText = argumentsText;
  }

  state.set(key, existing);
}

function buildResponsesStreamToolCalls(state: Map<string, OpenAIResponsesStreamToolCallState>): LLMToolCall[] {
  return [...state.values()].map((entry) => ({
    id: entry.id ?? entry.key,
    type: entry.type === "function_call" ? "function" : (entry.type ?? "function"),
    name: entry.name,
    arguments: entry.argumentsText.length > 0 ? entry.argumentsText : {},
  }));
}

function pickResponsesText(payload: Record<string, unknown>): string {
  const outputText = payload.output_text;
  if (typeof outputText === "string") {
    return outputText;
  }

  const output = payload.output;
  if (!Array.isArray(output)) {
    return "";
  }

  return output
    .map((item) => {
      if (!isRecord(item)) {
        return "";
      }

      if (typeof item.text === "string") {
        return item.text;
      }

      if (typeof item.refusal === "string") {
        return item.refusal;
      }

      const content = item.content;
      if (!Array.isArray(content)) {
        return "";
      }

      return content
        .map((part) => {
          if (!isRecord(part)) {
            return "";
          }

          if (typeof part.text === "string") {
            return part.text;
          }

          if (typeof part.output_text === "string") {
            return part.output_text;
          }

          if (typeof part.refusal === "string") {
            return part.refusal;
          }

          return "";
        })
        .join("");
    })
    .join("");
}

function pickResponsesReasoning(payload: Record<string, unknown>): string {
  const direct = pickReasoningText(payload);
  if (direct) {
    return direct;
  }

  const output = payload.output;
  if (!Array.isArray(output)) {
    return "";
  }

  return output
    .map((item) => {
      if (!isRecord(item)) {
        return "";
      }

      const itemReasoning = pickReasoningText(item);
      if (itemReasoning) {
        return itemReasoning;
      }

      const itemType = pickString(item.type) ?? "";
      if (itemType.includes("reasoning") && Array.isArray(item.summary)) {
        return item.summary.map((part) => pickTextLikePart(part)).join("");
      }
      if ((itemType.includes("reasoning") || itemType.includes("thinking")) && Array.isArray(item.content)) {
        return item.content.map((part) => isRecord(part) ? pickTextLike(part) : "").join("");
      }

      return "";
    })
    .join("");
}

function pickResponsesFinishReason(payload: Record<string, unknown>): string | undefined {
  const reason = payload.status;
  if (typeof reason === "string" && reason.length > 0) {
    return reason;
  }

  return undefined;
}

export type { OpenAICompatibleAdapterOptions } from "./openai-compatible-common";
