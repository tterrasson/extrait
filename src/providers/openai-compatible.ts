import type {
  EmbeddingRequest,
  EmbeddingResult,
  HTTPHeaders,
  LLMAdapter,
  LLMMessage,
  LLMRequest,
  LLMResponse,
  LLMStreamCallbacks,
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
  buildURL,
  cleanUndefined,
  isRecord,
  mergeUsage,
  pickString,
  preferLatestUsage,
  safeJSONParse,
  toFiniteNumber,
} from "./utils";

export interface OpenAICompatibleAdapterOptions {
  baseURL: string;
  model: string;
  apiKey?: string;
  path?: string;
  responsesPath?: string;
  embeddingPath?: string;
  defaultMaxToolRounds?: number;
  headers?: HTTPHeaders;
  defaultBody?: Record<string, unknown>;
  fetcher?: typeof fetch;
}

type OpenAIReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh";

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
  const path = options.path ?? "/v1/chat/completions";
  const responsesPath = options.responsesPath ?? "/v1/responses";
  const embeddingPath = options.embeddingPath ?? "/v1/embeddings";

  return {
    provider: "openai-compatible",
    model: options.model,

    async complete(request: LLMRequest): Promise<LLMResponse> {
      return completeOpenAIRequest(options, fetcher, path, responsesPath, request);
    },

    async stream(request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
      const usesResponses = shouldUseResponsesAPI(options, request);
      const usesMCP = hasMCPClients(request.mcpClients);
      if (usesResponses) {
        if (usesMCP) {
          return streamWithResponsesAPIWithMCP(options, fetcher, responsesPath, request, callbacks);
        }
        return streamWithResponsesAPIPassThrough(options, fetcher, responsesPath, request, callbacks);
      }
      if (usesMCP) {
        return streamWithChatCompletionsWithMCP(options, fetcher, path, request, callbacks);
      }

      return streamWithChatCompletionsPassThrough(options, fetcher, path, request, callbacks);
    },

    async embed(request: EmbeddingRequest): Promise<EmbeddingResult> {
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
    const message = await response.text();
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  callbacks.onStart?.();
  let text = "";
  let reasoning = "";
  let usage: LLMUsage | undefined;
  let finishReason: string | undefined;
  const streamedToolCalls = new Map<number, OpenAIStreamToolCallState>();
  const nativeToolCalls = new NativeToolCallStreamState(requestDeclaresTools(options, request));

  await consumeSSE(response, (data) => {
    if (data === "[DONE]") {
      return;
    }

    const json = safeJSONParse(data);
    if (!isRecord(json)) {
      return;
    }

    const rawDelta = pickAssistantDelta(json);
    const reasoningDelta = pickAssistantReasoningDelta(json);
    const chunkUsage = pickUsage(json);
    const chunkFinishReason = pickFinishReason(json);
    collectOpenAIStreamToolCalls(json, streamedToolCalls);
    const nativeDelta = nativeToolCalls.push(rawDelta);
    const delta = nativeDelta.textDelta;
    const chunkToolCalls = mergeToolCalls(buildOpenAIStreamToolCalls(streamedToolCalls), nativeDelta.toolCalls);

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
    );
  });

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
    usage,
    finishReason: finishReason ?? (toolCalls.length > 0 ? "tool_calls" : undefined),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
  };
  callbacks.onComplete?.(out);
  return out;
}

async function embedOpenAI(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: EmbeddingRequest,
): Promise<EmbeddingResult> {
  const body = cleanUndefined({
    ...options.defaultBody,
    ...request.body,
    model: request.model ?? options.model,
    input: request.input,
    dimensions: request.dimensions,
    encoding_format: "float",
  });

  const response = await fetcher(buildURL(options.baseURL, path), {
    method: "POST",
    headers: buildHeaders(options),
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  const json = (await response.json()) as Record<string, unknown>;
  const data = json.data;
  if (!Array.isArray(data)) {
    throw new Error("Unexpected embedding response: missing data array");
  }

  return {
    embeddings: data.map((d: unknown) => (
      isRecord(d) && Array.isArray(d.embedding) ? (d.embedding as number[]) : []
    )),
    model: pickString(json.model) ?? (body.model as string),
    usage: pickUsage(json),
    raw: json,
  };
}

async function completeOpenAIRequest(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  chatPath: string,
  responsesPath: string,
  request: LLMRequest,
): Promise<LLMResponse> {
  const usesResponses = shouldUseResponsesAPI(options, request);
  const usesMCP = hasMCPClients(request.mcpClients);

  if (usesResponses) {
    if (usesMCP) {
      return completeWithResponsesAPIWithMCP(options, fetcher, responsesPath, request);
    }
    return completeWithResponsesAPIPassThrough(options, fetcher, responsesPath, request);
  }

  if (usesMCP) {
    return completeWithChatCompletionsWithMCP(options, fetcher, chatPath, request);
  }

  return completeWithChatCompletionsPassThrough(options, fetcher, chatPath, request);
}

function buildChatCompletionsBody(
  options: OpenAICompatibleAdapterOptions,
  request: LLMRequest,
  overrides: Record<string, unknown>,
): Record<string, unknown> {
  return buildOpenAIRequestBody(options, request, "max_tokens", overrides);
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

function buildResponsesBody(
  options: OpenAICompatibleAdapterOptions,
  request: LLMRequest,
  overrides: Record<string, unknown>,
): Record<string, unknown> {
  return buildOpenAIRequestBody(options, request, "max_output_tokens", overrides);
}

function buildOpenAIRequestBody(
  options: OpenAICompatibleAdapterOptions,
  request: LLMRequest,
  maxTokenKey: "max_tokens" | "max_output_tokens",
  overrides: Record<string, unknown>,
): Record<string, unknown> {
  return cleanUndefined({
    ...options.defaultBody,
    ...request.body,
    model: options.model,
    temperature: request.temperature,
    reasoning_effort: toOpenAIReasoningEffort(request.reasoningEffort),
    [maxTokenKey]: request.maxTokens,
    ...overrides,
  });
}

function sendOpenAIRequest(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  body: Record<string, unknown>,
): Promise<Response> {
  return fetcher(buildURL(options.baseURL, path), {
    method: "POST",
    headers: buildHeaders(options),
    body: JSON.stringify(body),
    signal: request.signal,
  });
}

async function sendOpenAIJsonRequest(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const response = await sendOpenAIRequest(options, fetcher, path, request, body);

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  return (await response.json()) as Record<string, unknown>;
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
): LLMResponse {
  return {
    text,
    reasoning: joinReasoningBlocks(state.reasoningBlocks) || undefined,
    reasoningBlocks: state.reasoningBlocks.length > 0 ? state.reasoningBlocks : undefined,
    raw,
    usage: state.aggregatedUsage,
    finishReason: state.finishReason,
    toolCalls: state.executedToolCalls.length > 0 ? state.executedToolCalls : undefined,
    toolExecutions: state.toolExecutions.length > 0 ? state.toolExecutions : undefined,
  };
}

function emitOpenAIStreamChunk(
  callbacks: LLMStreamCallbacks,
  round: number | undefined,
  raw: Record<string, unknown>,
  delta: string,
  reasoningDelta: string,
  usage: LLMUsage | undefined,
  finishReason: string | undefined,
  toolCalls?: LLMToolCall[],
): void {
  if (delta || reasoningDelta || usage || finishReason || toolCalls) {
    callbacks.onChunk?.({
      textDelta: delta,
      reasoningDelta: reasoningDelta || undefined,
      ...(round !== undefined ? { turnIndex: round } : {}),
      raw,
      usage,
      finishReason,
      toolCalls,
    });
  }
}

async function completeWithChatCompletionsPassThrough(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
): Promise<LLMResponse> {
  const response = await sendOpenAIRequest(
    options,
    fetcher,
    path,
    request,
    buildChatCompletionsBody(options, request, {
      messages: buildMessages(request),
      stream: false,
    }),
  );

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  const payload = await parseOpenAICompatibleJSONResponse(
    response,
    "Failed to parse OpenAI-compatible chat completion response",
  );
  const assistantMessage = pickAssistantMessage(payload);
  if (!assistantMessage) {
    throw new Error("No assistant message in OpenAI-compatible response.");
  }

  const toolCalls = pickChatToolCalls(payload);
  const reasoning = pickAssistantReasoning(payload);
  return {
    text: pickAssistantText(payload),
    reasoning: reasoning.length > 0 ? reasoning : undefined,
    raw: payload,
    usage: pickUsage(payload),
    finishReason: pickFinishReason(payload),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
  };
}

async function parseOpenAICompatibleJSONResponse(
  response: Response,
  context: string,
): Promise<Record<string, unknown>> {
  const rawBody = await response.text();

  try {
    return JSON.parse(rawBody) as Record<string, unknown>;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `${context} (HTTP ${response.status}): ${message}. Raw body: ${formatResponseBodyForError(rawBody)}`,
    );
  }
}

function formatResponseBodyForError(rawBody: string, maxLength = 2_000): string {
  const normalized = rawBody.trim();
  if (normalized.length === 0) {
    return "[empty body]";
  }

  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength)}...[truncated ${normalized.length - maxLength} chars]`;
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
    finishReason = pickFinishReason(payload);

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
      provider: "openai-compatible",
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
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
  };
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
  const toolCalls = pickResponsesToolCalls(payload);
  return {
    text: pickResponsesText(payload) || pickAssistantText(payload),
    raw: payload,
    usage: pickUsage(payload),
    finishReason: pickResponsesFinishReason(payload) ?? pickFinishReason(payload),
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
      buildResponsesBody(options, request, {
        input: state.input,
        previous_response_id: state.previousResponseId,
        tools: transportTools,
        tool_choice: request.toolChoice,
        parallel_tool_calls: request.parallelToolCalls,
      }),
    );
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
      const text = pickResponsesText(payload) || pickAssistantText(payload);
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

    state.input = outputs.map((entry) => ({
      type: "function_call_output",
      call_id: entry.call.id,
      output: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
    }));
    state.previousResponseId = pickString(payload.id);
  }

  return buildResponsesMCPResult(
    state,
    pickResponsesText(state.lastPayload ?? {}) || pickAssistantText(state.lastPayload ?? {}),
    state.lastPayload,
  );
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
      const message = await response.text();
      throw new Error(`HTTP ${response.status}: ${message}`);
    }

    let roundText = "";
    let roundReasoning = "";
    let roundUsage: LLMUsage | undefined;
    let roundFinishReason: string | undefined;
    const streamedToolCalls = new Map<number, OpenAIStreamToolCallState>();
    const nativeToolCalls = new NativeToolCallStreamState();
    let reasoningFieldName: OpenAIReasoningFieldName | undefined;

    await consumeSSE(response, (data) => {
      if (data === "[DONE]") {
        return;
      }

      const json = safeJSONParse(data);
      if (!isRecord(json)) {
        return;
      }

      lastPayload = json;

      const rawDelta = pickAssistantDelta(json);
      const reasoningDelta = pickAssistantReasoningDelta(json);
      const chunkUsage = pickUsage(json);
      const chunkFinishReason = pickFinishReason(json);
      const nativeDelta = nativeToolCalls.push(rawDelta);
      const delta = nativeDelta.textDelta;

      collectOpenAIStreamToolCalls(json, streamedToolCalls);
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
      emitOpenAIStreamChunk(callbacks, round, json, delta, reasoningDelta, chunkUsage, chunkFinishReason, chunkToolCalls);
    });

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
      provider: "openai-compatible",
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
    const message = await response.text();
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  callbacks.onStart?.();

  let text = "";
  let usage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  const streamedToolCalls = new Map<string, OpenAIResponsesStreamToolCallState>();

  await consumeSSE(response, (data) => {
    if (data === "[DONE]") {
      return;
    }

    const json = safeJSONParse(data);
    if (!isRecord(json)) {
      return;
    }

    const roundPayload = pickResponsesStreamPayload(json);
    if (roundPayload) {
      lastPayload = roundPayload;
    }

    const delta = pickResponsesStreamTextDelta(json);
    const chunkUsage = pickResponsesStreamUsage(json);
    const chunkFinishReason = pickResponsesStreamFinishReason(json);
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

    emitOpenAIStreamChunk(
      callbacks,
      undefined,
      json,
      delta,
      "",
      chunkUsage,
      chunkFinishReason,
      chunkToolCalls.length > 0 ? chunkToolCalls : undefined,
    );
  });

  const finalPayload = lastPayload ?? {};
  const toolCalls = buildResponsesStreamToolCalls(streamedToolCalls);
  const out: LLMResponse = {
    text: text.length > 0 ? text : (pickResponsesText(finalPayload) || pickAssistantText(finalPayload)),
    raw: finalPayload,
    usage: preferLatestUsage(usage, pickUsage(finalPayload)),
    finishReason: finishReason ?? pickResponsesFinishReason(finalPayload) ?? pickFinishReason(finalPayload),
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
      buildResponsesBody(options, request, {
        input: state.input,
        previous_response_id: state.previousResponseId,
        tools: transportTools,
        tool_choice: request.toolChoice,
        parallel_tool_calls: request.parallelToolCalls,
        stream: true,
      }),
    );

    if (!response.ok) {
      const message = await response.text();
      throw new Error(`HTTP ${response.status}: ${message}`);
    }

    let roundText = "";
    let roundReasoning = "";
    let roundUsage: LLMUsage | undefined;
    let roundFinishReason: string | undefined;
    let roundPayload: Record<string, unknown> | undefined;
    const streamedToolCalls = new Map<string, OpenAIResponsesStreamToolCallState>();

    await consumeSSE(response, (data) => {
      if (data === "[DONE]") {
        return;
      }

      const json = safeJSONParse(data);
      if (!isRecord(json)) {
        return;
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

      // Stream the accumulated tool-call snapshot per chunk (see the chat
      // completions variant), so partial arguments surface as they build up.
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
      );
    });

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
        : (roundPayload ? (pickResponsesText(roundPayload) || pickAssistantText(roundPayload)) : "");
      const out = buildResponsesMCPResult(state, finalText, roundPayload ?? state.lastPayload);
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

    state.input = outputs.map((entry) => ({
      type: "function_call_output",
      call_id: entry.call.id,
      output: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
    }));
    state.previousResponseId = pickString(roundPayload?.id);
  }

  const out = buildResponsesMCPResult(
    state,
    pickResponsesText(state.lastPayload ?? {}) || pickAssistantText(state.lastPayload ?? {}),
    state.lastPayload,
  );
  request.onTurnTransition?.({ turnIndex: maxToolRounds + 1, kind: "streamEnd" });
  callbacks.onComplete?.(out);
  return out;
}

function shouldUseResponsesAPI(options: OpenAICompatibleAdapterOptions, request: LLMRequest): boolean {
  if (options.path?.includes("/responses")) {
    return true;
  }

  const body = request.body;
  if (!isRecord(body)) {
    return false;
  }

  return "input" in body || "previous_response_id" in body;
}

function buildHeaders(options: OpenAICompatibleAdapterOptions): HTTPHeaders {
  return {
    "content-type": "application/json",
    ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}),
    ...options.headers,
  };
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

function buildResponsesInput(request: LLMRequest): unknown {
  if (isRecord(request.body) && "input" in request.body) {
    return request.body.input;
  }

  if (Array.isArray(request.messages) && request.messages.length > 0) {
    return request.messages.map((message) => toOpenAIMessage(message));
  }

  return buildMessages(request);
}

function toOpenAIMessage(message: LLMMessage): Record<string, unknown> {
  return { ...message };
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

function toOpenAIReasoningEffort(value: LLMRequest["reasoningEffort"]): OpenAIReasoningEffort | undefined {
  if (!value) {
    return undefined;
  }

  return value === "max" ? "xhigh" : value;
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

  return pickTextLike(delta.content);
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

interface OpenAIResponsesStreamToolCallState {
  key: string;
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
  constructor(private readonly enabled = true) {}

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
          this.pending = "";
        }
        break;
      }

      const blockEnd = closeIndex + NATIVE_TOOL_CALL_CLOSE.length;
      const call = parseNativeToolCallBlock(this.pending.slice(0, blockEnd), this.calls.length);
      if (call) {
        this.calls.push(call);
        toolCalls.push(call);
      }
      this.pending = this.pending.slice(blockEnd);
    }

    return { textDelta, toolCalls };
  }
}

function requestDeclaresTools(options: OpenAICompatibleAdapterOptions, request: LLMRequest): boolean {
  const hasTools = (body: Record<string, unknown> | undefined): boolean =>
    Array.isArray(body?.tools) && body.tools.length > 0;
  return hasTools(request.body) || hasTools(options.defaultBody);
}

const NATIVE_FUNCTION_PATTERN = /<function=([^>\s]+)\s*>([\s\S]*?)<\/function>/;
const NATIVE_PARAMETER_PATTERN = /<parameter=([^>\s]+)\s*>([\s\S]*?)<\/parameter>/g;

function parseNativeToolCallBlock(block: string, index: number): LLMToolCall | undefined {
  const inner = extractNativeToolCallInner(block);
  if (inner === undefined) {
    return undefined;
  }

  // Two shapes appear inside <tool_call>…</tool_call>: a JSON object (Hermes/Qwen
  // style) or a nested <function=…><parameter=…> tree (Llama style).
  return parseNativeJsonToolCall(inner, index) ?? parseNativeXmlToolCall(inner, index);
}

function extractNativeToolCallInner(block: string): string | undefined {
  const openEnd = block.indexOf(">");
  const closeStart = block.lastIndexOf(NATIVE_TOOL_CALL_CLOSE);
  if (openEnd < 0 || closeStart < 0 || closeStart <= openEnd) {
    return undefined;
  }

  return block.slice(openEnd + 1, closeStart).trim();
}

function parseNativeXmlToolCall(inner: string, index: number): LLMToolCall | undefined {
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
    id: `call_native_${index}`,
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

function parseNativeJsonToolCall(inner: string, index: number): LLMToolCall | undefined {
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
    id: pickString(parsed.id) ?? `call_native_${index}`,
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
  if (!eventType.includes("output_text.delta")) {
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

  const directStatus = pickString(payload.status);
  if (directStatus) {
    return directStatus;
  }

  if (isRecord(payload.response)) {
    return pickString(payload.response.status);
  }

  return undefined;
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

  const key = forcedKey ?? pickString(item.call_id) ?? pickString(item.id) ?? `call_${state.size}`;
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
      if ((itemType.includes("reasoning") || itemType.includes("thinking")) && Array.isArray(item.content)) {
        return item.content.map((part) => isRecord(part) ? pickTextLike(part) : "").join("");
      }

      return "";
    })
    .join("");
}

function pickAssistantText(payload: Record<string, unknown>): string {
  const message = pickAssistantMessage(payload);
  if (message) {
    const text = pickTextLike(message.content);
    if (text.length > 0) {
      return text;
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

function pickReasoningText(value: Record<string, unknown>): string {
  return pickTextLike(value.reasoning) || pickTextLike(value.reasoning_content);
}

function pushReasoningBlock(blocks: ReasoningBlock[], turnIndex: number, text: string | undefined): void {
  const clean = text?.replace(/<\/?think\s*>/gi, "").trim();
  if (!clean) {
    return;
  }

  blocks.push({ turnIndex, text: clean });
}

function joinReasoningBlocks(blocks: ReasoningBlock[]): string {
  return blocks.map((block) => block.text).filter(Boolean).join("\n\n");
}

function pickTextLike(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  if (Array.isArray(value)) {
    return value.map((part) => pickTextLikePart(part)).join("");
  }

  if (!isRecord(value)) {
    return "";
  }

  return pickTextLikePart(value);
}

function pickTextLikePart(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  if (!isRecord(value)) {
    return "";
  }

  return pickString(value.text)
    ?? pickString(value.output_text)
    ?? pickString(value.reasoning)
    ?? pickString(value.reasoning_content)
    ?? (Array.isArray(value.content) ? value.content.map((part) => pickTextLikePart(part)).join("") : "");
}

function hasTextLikeValue(value: unknown): boolean {
  return pickTextLike(value).length > 0;
}

function pickUsage(payload: Record<string, unknown>): LLMUsage | undefined {
  const usage = payload.usage;
  if (!isRecord(usage)) {
    return undefined;
  }

  const promptTokens = toFiniteNumber(usage.prompt_tokens ?? usage.input_tokens);
  const completionTokens = toFiniteNumber(usage.completion_tokens ?? usage.output_tokens);
  const totalTokens = toFiniteNumber(usage.total_tokens);

  return {
    inputTokens: promptTokens,
    outputTokens: completionTokens,
    totalTokens,
  };
}

function pickFinishReason(payload: Record<string, unknown>): string | undefined {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0 || !isRecord(choices[0])) {
    return undefined;
  }

  const reason = choices[0].finish_reason;
  return typeof reason === "string" && reason.length > 0 ? reason : undefined;
}

function pickResponsesFinishReason(payload: Record<string, unknown>): string | undefined {
  const reason = payload.status;
  if (typeof reason === "string" && reason.length > 0) {
    return reason;
  }

  return undefined;
}
