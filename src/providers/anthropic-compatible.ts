import type {
  EmbeddingResult,
  HTTPHeaders,
  LLMAdapter,
  LLMMessage,
  LLMRequest,
  LLMResponse,
  LLMStreamCallbacks,
  LLMStreamChunk,
  LLMToolCall,
  LLMToolCallRef,
  LLMUsage,
  ReasoningBlock,
} from "../types";
import { toAnthropicReasoningEffort } from "./reasoning-effort";
import { consumeSSE } from "./stream-utils";
import {
  executeMCPToolCalls,
  hasMCPClients,
  normalizeMaxToolRounds,
  parseToolArguments,
  resolveMCPToolset,
  stringifyToolOutput,
  toProviderFunctionTools,
} from "./mcp-runtime";
import {
  buildURL,
  cleanUndefined,
  isRecord,
  joinReasoningBlocks,
  mergeUsage,
  pickString,
  preferLatestUsage,
  pushReasoningBlock,
  readErrorBody,
  safeJSONParse,
  toFiniteNumber,
} from "./utils";

export interface AnthropicCompatibleAdapterOptions {
  baseURL: string;
  model: string;
  apiKey?: string;
  path?: string;
  headers?: HTTPHeaders;
  version?: string;
  defaultMaxTokens?: number;
  defaultMaxToolRounds?: number;
  defaultBody?: Record<string, unknown>;
  fetcher?: typeof fetch;
}

export const DEFAULT_ANTHROPIC_MAX_TOKENS = 1024;
export const DEFAULT_ANTHROPIC_VERSION = "2023-06-01";

export function createAnthropicCompatibleAdapter(options: AnthropicCompatibleAdapterOptions): LLMAdapter {
  const fetcher = options.fetcher ?? fetch;
  const path = options.path ?? "/v1/messages";

  return {
    provider: "anthropic-compatible",
    model: options.model,

    async complete(request: LLMRequest): Promise<LLMResponse> {
      if (hasMCPClients(request.mcpClients)) {
        return completeWithMCPToolLoop(options, fetcher, path, request);
      }

      return completePassThrough(options, fetcher, path, request);
    },

    async stream(request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
      if (hasMCPClients(request.mcpClients)) {
        return streamWithMCPToolLoop(options, fetcher, path, request, callbacks);
      }

      return streamPassThrough(options, fetcher, path, request, callbacks);
    },

    async embed(): Promise<EmbeddingResult> {
      throw new Error(
        "Anthropic does not provide a native embedding API. " +
          "Use the openai-compatible provider with Voyage AI (https://api.voyageai.com) — " +
          "Anthropic's recommended embedding solution, which uses the same request format.",
      );
    },
  };
}

async function streamPassThrough(
  options: AnthropicCompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  callbacks: LLMStreamCallbacks,
): Promise<LLMResponse> {
  const input = resolveAnthropicInput(request);
  const response = await sendAnthropicMessage(options, fetcher, path, request, {
    system: input.systemPrompt,
    messages: input.messages,
    stream: true,
  });

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
  const streamedToolCalls = new Map<number, AnthropicStreamToolCallState>();

  await consumeSSE(response, (data) => {
    if (data === "[DONE]") {
      streamTerminated = true;
      return;
    }

    const json = safeJSONParse(data);
    if (!isRecord(json)) {
      throw new Error("Invalid JSON event in Anthropic-compatible stream.");
    }

    throwForAnthropicStreamError(json);
    streamTerminated ||= isAnthropicTerminalEvent(json);
    lastPayload = json;

    const delta = pickAnthropicDelta(json);
    const reasoningDelta = pickAnthropicReasoningDelta(json);
    const chunkUsage = pickUsage(json);
    const chunkFinishReason = pickFinishReason(json);

    collectAnthropicStreamToolCalls(json, streamedToolCalls);
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

    // Surface the accumulated tool-call snapshot (id/name from `tool_use` blocks,
    // arguments from `input_json_delta` fragments) on every chunk once a tool
    // call is in flight, so callers stream partial arguments instead of only
    // seeing the completed call. Downstream dedups identical snapshots.
    const streamedSnapshot = buildAnthropicStreamToolCalls(streamedToolCalls);
    const chunkToolCalls = streamedSnapshot.length > 0 ? streamedSnapshot : undefined;

    if (delta || reasoningDelta || chunkUsage || chunkFinishReason || chunkToolCalls) {
      callbacks.onChunk?.({
        textDelta: delta,
        reasoningDelta: reasoningDelta || undefined,
        raw: json,
        usage: chunkUsage,
        finishReason: chunkFinishReason,
        toolCalls: chunkToolCalls,
      });
    }
  });

  assertAnthropicStreamTerminated(streamTerminated || finishReason !== undefined);

  const toolCalls = buildAnthropicStreamToolCalls(streamedToolCalls);
  const out: LLMResponse = {
    text,
    reasoning: reasoning.length > 0 ? reasoning : undefined,
    raw: lastPayload,
    usage,
    finishReason: finishReason ?? (toolCalls.length > 0 ? "tool_use" : undefined),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
  };
  callbacks.onComplete?.(out);
  return out;
}

function sendAnthropicMessage(
  options: AnthropicCompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  body: Record<string, unknown>,
): Promise<Response> {
  return fetcher(buildURL(options.baseURL, path), {
    method: "POST",
    headers: buildHeaders(options),
    body: JSON.stringify(
      buildAnthropicRequestBody(options, request, {
        ...options.defaultBody,
        ...request.body,
        model: options.model,
        temperature: request.temperature,
        ...body,
      }),
    ),
    signal: request.signal,
  });
}

async function completePassThrough(
  options: AnthropicCompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
): Promise<LLMResponse> {
  const input = resolveAnthropicInput(request);
  const response = await sendAnthropicMessage(options, fetcher, path, request, {
    system: input.systemPrompt,
    messages: input.messages,
    stream: false,
  });

  if (!response.ok) {
    const message = await readErrorBody(response);
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  const data = (await response.json()) as Record<string, unknown>;
  const text = extractAnthropicText(data);
  const reasoning = extractAnthropicReasoning(data);
  const toolCalls = pickAnthropicToolCalls(data);

  if (!text && !reasoning && toolCalls.length === 0) {
    throw new Error("No assistant text in Anthropic-compatible response.");
  }

  return {
    text,
    reasoning: reasoning.length > 0 ? reasoning : undefined,
    raw: data,
    usage: pickUsage(data),
    finishReason: pickFinishReason(data),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
  };
}

async function completeWithMCPToolLoop(
  options: AnthropicCompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
): Promise<LLMResponse> {
  const maxToolRounds = normalizeMaxToolRounds(request.maxToolRounds ?? options.defaultMaxToolRounds);

  const input = resolveAnthropicInput(request);
  let messages: Array<Record<string, unknown>> = input.messages;
  let aggregatedUsage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  const toolCalls: LLMToolCall[] = [];
  const toolExecutions: NonNullable<LLMResponse["toolExecutions"]> = [];
  const reasoningBlocks: ReasoningBlock[] = [];

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const tools = toAnthropicTools(toProviderFunctionTools(mcpToolset));

    const response = await sendAnthropicMessage(options, fetcher, path, request, {
      system: input.systemPrompt,
      messages,
      tools,
      tool_choice: toAnthropicToolChoice(request.toolChoice, request.parallelToolCalls),
      stream: false,
    });

    if (!response.ok) {
      const message = await readErrorBody(response);
      throw new Error(`HTTP ${response.status}: ${message}`);
    }

    const payload = (await response.json()) as Record<string, unknown>;
    lastPayload = payload;
    aggregatedUsage = mergeUsage(aggregatedUsage, pickUsage(payload));
    finishReason = pickFinishReason(payload) ?? finishReason;

    const content = Array.isArray(payload.content) ? payload.content : [];
    const calledTools = pickAnthropicToolCalls(payload).filter((call) => call.type === "function");
    pushReasoningBlock(reasoningBlocks, round, extractAnthropicReasoning(payload));

    if (calledTools.length === 0) {
      return {
        text: extractAnthropicText(payload),
        reasoning: joinReasoningBlocks(reasoningBlocks) || undefined,
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

    const toolResultContent: Array<Record<string, unknown>> = [];
    const outputs = await executeMCPToolCalls(calledTools, mcpToolset, {
      round,
      request,
      provider: "anthropic-compatible",
      model: options.model,
    });
    toolCalls.push(...outputs.map((entry) => entry.call));
    toolExecutions.push(...outputs.map((entry) => entry.execution));

    for (const entry of outputs) {
      toolResultContent.push({
        type: "tool_result",
        tool_use_id: entry.call.id,
        ...(entry.call.error ? { is_error: true } : {}),
        content: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
      });
    }

    messages = [
      ...messages,
      { role: "assistant", content },
      { role: "user", content: toolResultContent },
    ];
  }

  return {
    text: extractAnthropicText(lastPayload ?? {}),
    reasoning: joinReasoningBlocks(reasoningBlocks) || undefined,
    reasoningBlocks: reasoningBlocks.length > 0 ? reasoningBlocks : undefined,
    raw: lastPayload,
    usage: aggregatedUsage,
    finishReason,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
  };
}

async function streamWithMCPToolLoop(
  options: AnthropicCompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  callbacks: LLMStreamCallbacks,
): Promise<LLMResponse> {
  const maxToolRounds = normalizeMaxToolRounds(request.maxToolRounds ?? options.defaultMaxToolRounds);

  const input = resolveAnthropicInput(request);
  let messages: Array<Record<string, unknown>> = input.messages;
  let aggregatedUsage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  const toolCalls: LLMToolCall[] = [];
  const toolExecutions: NonNullable<LLMResponse["toolExecutions"]> = [];
  const reasoningBlocks: ReasoningBlock[] = [];

  callbacks.onStart?.();

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const tools = toAnthropicTools(toProviderFunctionTools(mcpToolset));

    const response = await sendAnthropicMessage(options, fetcher, path, request, {
      system: input.systemPrompt,
      messages,
      tools,
      tool_choice: toAnthropicToolChoice(request.toolChoice, request.parallelToolCalls),
      stream: true,
    });

    if (!response.ok) {
      const message = await readErrorBody(response);
      throw new Error(`HTTP ${response.status}: ${message}`);
    }

    let roundText = "";
    let roundReasoning = "";
    let roundUsage: LLMUsage | undefined;
    let roundFinishReason: string | undefined;
    let streamTerminated = false;
    const streamedToolCalls = new Map<number, AnthropicStreamToolCallState>();
    const streamedThinkingBlocks = new Map<number, AnthropicStreamThinkingBlockState>();

    await consumeSSE(response, (data) => {
      if (data === "[DONE]") {
        streamTerminated = true;
        return;
      }

      const json = safeJSONParse(data);
      if (!isRecord(json)) {
        throw new Error("Invalid JSON event in Anthropic-compatible stream.");
      }

      throwForAnthropicStreamError(json);
      streamTerminated ||= isAnthropicTerminalEvent(json);
      lastPayload = json;

      const delta = pickAnthropicDelta(json);
      const reasoningDelta = pickAnthropicReasoningDelta(json);
      const chunkUsage = pickUsage(json);
      const chunkFinishReason = pickFinishReason(json);

      collectAnthropicStreamToolCalls(json, streamedToolCalls);
      collectAnthropicStreamThinkingBlocks(json, streamedThinkingBlocks);
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

      // Stream the accumulated tool-call snapshot per chunk (see streamPassThrough)
      // so partial arguments surface as they build up, rather than only via the
      // single round-end emit below.
      const streamedSnapshot = buildAnthropicStreamToolCalls(streamedToolCalls);
      const chunkToolCalls = streamedSnapshot.length > 0 ? streamedSnapshot : undefined;

      if (delta || reasoningDelta || chunkUsage || chunkFinishReason || chunkToolCalls) {
        const chunk: LLMStreamChunk = {
          textDelta: delta,
          reasoningDelta: reasoningDelta || undefined,
          turnIndex: round,
          raw: json,
          usage: chunkUsage,
          finishReason: chunkFinishReason,
          toolCalls: chunkToolCalls,
        };
        callbacks.onChunk?.(chunk);
      }
    });

    assertAnthropicStreamTerminated(streamTerminated || roundFinishReason !== undefined);

    aggregatedUsage = mergeUsage(aggregatedUsage, roundUsage);
    if (roundFinishReason) {
      finishReason = roundFinishReason;
    }

    const calledTools = buildAnthropicStreamToolCalls(streamedToolCalls);
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
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
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

    const toolResultContent: Array<Record<string, unknown>> = [];
    const outputs = await executeMCPToolCalls(calledTools, mcpToolset, {
      round,
      request,
      provider: "anthropic-compatible",
      model: options.model,
    });
    toolCalls.push(...outputs.map((entry) => entry.call));
    toolExecutions.push(...outputs.map((entry) => entry.execution));
    request.onTurnTransition?.({ turnIndex: round, kind: "toolResultsReceived" });

    for (const entry of outputs) {
      toolResultContent.push({
        type: "tool_result",
        tool_use_id: entry.call.id,
        ...(entry.call.error ? { is_error: true } : {}),
        content: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
      });
    }

    messages = [
      ...messages,
      {
        role: "assistant",
        content: buildAnthropicAssistantToolContent(roundText, calledTools, streamedThinkingBlocks),
      },
      { role: "user", content: toolResultContent },
    ];
  }

  const out: LLMResponse = {
    text: "",
    reasoning: joinReasoningBlocks(reasoningBlocks) || undefined,
    reasoningBlocks: reasoningBlocks.length > 0 ? reasoningBlocks : undefined,
    raw: lastPayload,
    usage: aggregatedUsage,
    finishReason,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
  };
  request.onTurnTransition?.({ turnIndex: maxToolRounds + 1, kind: "streamEnd" });
  callbacks.onComplete?.(out);
  return out;
}

function buildHeaders(options: AnthropicCompatibleAdapterOptions): HTTPHeaders {
  return {
    "content-type": "application/json",
    ...(options.apiKey ? { "x-api-key": options.apiKey } : {}),
    "anthropic-version": options.version ?? DEFAULT_ANTHROPIC_VERSION,
    ...options.headers,
  };
}

function buildAnthropicRequestBody(
  options: AnthropicCompatibleAdapterOptions,
  request: LLMRequest,
  body: Record<string, unknown>,
): Record<string, unknown> {
  const bodyOutputConfig = isRecord(body.output_config) ? body.output_config : undefined;
  const bodyThinking = body.thinking;
  const hasExplicitThinking = Object.prototype.hasOwnProperty.call(body, "thinking");
  const reasoningEffort = request.reasoningEffort;
  const anthropicEffort = toAnthropicReasoningEffort(reasoningEffort);
  const outputConfigWithoutEffort = omitRecordKey(bodyOutputConfig, "effort");
  // An explicit `thinking` in the body always wins: it is the escape hatch for
  // an endpoint whose thinking shape we do not model.
  const thinking = hasExplicitThinking
    ? bodyThinking
    : reasoningEffort === "none"
      ? { type: "disabled" }
      : anthropicEffort
        ? { type: "adaptive" }
        : undefined;

  return cleanUndefined({
    ...body,
    // Precedence: per-request maxTokens > max_tokens already present in the
    // body (request.body over defaultBody via spread) > adapter default.
    max_tokens: resolveMaxTokens(request.maxTokens, toFiniteNumber(body.max_tokens) ?? options.defaultMaxTokens),
    output_config: anthropicEffort
      ? cleanUndefined({
          ...bodyOutputConfig,
          effort: anthropicEffort,
        })
      : reasoningEffort === "none"
        ? outputConfigWithoutEffort
        : bodyOutputConfig,
    thinking,
  });
}

function omitRecordKey(
  value: Record<string, unknown> | undefined,
  key: string,
): Record<string, unknown> | undefined {
  if (!value) {
    return undefined;
  }

  const result = Object.fromEntries(Object.entries(value).filter(([entryKey]) => entryKey !== key));
  return Object.keys(result).length > 0 ? result : undefined;
}

function resolveAnthropicInput(
  request: LLMRequest,
): { systemPrompt?: string; messages: Array<Record<string, unknown>> } {
  if (Array.isArray(request.messages) && request.messages.length > 0) {
    return toAnthropicInput(request.messages);
  }

  if (typeof request.prompt !== "string" || request.prompt.trim().length === 0) {
    throw new Error("LLMRequest must include a prompt or messages.");
  }

  return {
    systemPrompt: request.systemPrompt,
    messages: [{ role: "user", content: request.prompt }],
  };
}

function toAnthropicInput(
  messages: LLMMessage[],
): { systemPrompt?: string; messages: Array<Record<string, unknown>> } {
  const systemParts: string[] = [];
  const normalizedMessages: Array<Record<string, unknown>> = [];
  let sawNonSystem = false;

  for (const message of messages) {
    if (message.role === "system") {
      if (sawNonSystem) {
        throw new Error('Anthropic-compatible messages only support "system" turns at the beginning.');
      }
      systemParts.push(stringifyAnthropicSystemContent(message.content));
      continue;
    }

    sawNonSystem = true;

    if (message.role === "assistant" && Array.isArray(message.tool_calls)) {
      const parts: unknown[] = [];
      parts.push(...toAnthropicMessageContent(message.content));
      for (const tc of message.tool_calls as LLMToolCallRef[]) {
        const input = parseToolArguments(tc.function.arguments);
        parts.push({ type: "tool_use", id: tc.id, name: tc.function.name, input: isRecord(input) ? input : {} });
      }
      normalizedMessages.push({ role: "assistant", content: parts });
      continue;
    }

    if (message.role === "tool") {
      normalizedMessages.push({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: message.tool_call_id,
          content: Array.isArray(message.content)
            ? toAnthropicMessageContent(message.content)
            : message.content,
        }],
      });
      continue;
    }

    normalizedMessages.push({
      role: message.role,
      content: Array.isArray(message.content)
        ? toAnthropicMessageContent(message.content)
        : message.content,
    });
  }

  if (normalizedMessages.length === 0) {
    throw new Error("Anthropic-compatible requests require at least one non-system message.");
  }

  return {
    systemPrompt: systemParts.length > 0 ? systemParts.join("\n\n") : undefined,
    messages: normalizedMessages,
  };
}

function toAnthropicMessageContent(content: LLMMessage["content"]): Array<Record<string, unknown>> {
  if (typeof content === "string") {
    return content.length > 0 ? [{ type: "text", text: content }] : [];
  }

  return content.map((part) => {
    if (part.type === "text") {
      return { type: "text", text: part.text };
    }

    return {
      type: "image",
      source: toAnthropicImageSource(part.image_url.url),
    };
  });
}

function toAnthropicImageSource(url: string): Record<string, unknown> {
  // Scheme and the `;base64` marker are case-insensitive.
  const dataURL = /^data:([^;,]+);base64,(.*)$/is.exec(url);
  if (dataURL) {
    return {
      type: "base64",
      media_type: dataURL[1],
      data: dataURL[2],
    };
  }

  return { type: "url", url };
}

function stringifyAnthropicSystemContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (content === null || content === undefined) {
    return "";
  }

  try {
    return JSON.stringify(content, null, 2) ?? "";
  } catch {
    return String(content);
  }
}

function resolveMaxTokens(value: number | undefined, fallback: number | undefined): number {
  const requested = toFiniteNumber(value);
  if (requested !== undefined && requested > 0) {
    return Math.floor(requested);
  }

  const configured = toFiniteNumber(fallback);
  if (configured !== undefined && configured > 0) {
    return Math.floor(configured);
  }

  return DEFAULT_ANTHROPIC_MAX_TOKENS;
}

function extractAnthropicText(payload: Record<string, unknown>): string {
  const content = payload.content;
  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((part) => {
      if (!isRecord(part) || part.type !== "text") {
        return "";
      }

      const text = part.text;
      return typeof text === "string" ? text : "";
    })
    .join("");
}

function extractAnthropicReasoning(payload: Record<string, unknown>): string {
  const content = payload.content;
  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((part) => {
      if (!isRecord(part)) {
        return "";
      }

      const type = pickString(part.type) ?? "";
      if (type !== "thinking" && type !== "reasoning") {
        return "";
      }

      return pickString(part.thinking) ?? pickString(part.text) ?? pickString(part.reasoning) ?? "";
    })
    .join("");
}

function pickAnthropicToolCalls(payload: Record<string, unknown>): LLMToolCall[] {
  const content = payload.content;
  if (!Array.isArray(content)) {
    return [];
  }

  const calls: LLMToolCall[] = [];
  for (const part of content) {
    if (!isRecord(part) || part.type !== "tool_use") {
      continue;
    }

    calls.push({
      id: pickString(part.id) ?? "",
      type: "function",
      name: pickString(part.name),
      arguments: part.input,
    });
  }

  return calls;
}

function pickAnthropicDelta(payload: Record<string, unknown>): string {
  const deltaObject = payload.delta;
  if (isRecord(deltaObject) && typeof deltaObject.text === "string") {
    return deltaObject.text;
  }

  const contentBlock = payload.content_block;
  if (isRecord(contentBlock) && typeof contentBlock.text === "string") {
    return contentBlock.text;
  }

  return "";
}

function pickAnthropicReasoningDelta(payload: Record<string, unknown>): string {
  const deltaObject = payload.delta;
  if (isRecord(deltaObject)) {
    const type = pickString(deltaObject.type) ?? "";
    if (type === "thinking_delta" || type === "reasoning_delta") {
      return pickString(deltaObject.thinking) ?? pickString(deltaObject.text) ?? "";
    }
  }

  const contentBlock = payload.content_block;
  if (isRecord(contentBlock)) {
    const type = pickString(contentBlock.type) ?? "";
    if (type === "thinking" || type === "reasoning") {
      return pickString(contentBlock.thinking) ?? pickString(contentBlock.text) ?? "";
    }
  }

  return "";
}

interface AnthropicStreamToolCallState {
  index: number;
  id?: string;
  name?: string;
  input?: unknown;
  argumentsText: string;
}

interface AnthropicStreamThinkingBlockState {
  index: number;
  block: Record<string, unknown>;
}

function collectAnthropicStreamThinkingBlocks(
  payload: Record<string, unknown>,
  state: Map<number, AnthropicStreamThinkingBlockState>,
): void {
  const eventType = pickString(payload.type);
  const index = pickContentBlockIndex(payload.index);

  if (eventType === "content_block_start" && isRecord(payload.content_block)) {
    const type = pickString(payload.content_block.type);
    if (type === "thinking" || type === "redacted_thinking") {
      state.set(index, { index, block: { ...payload.content_block } });
    }
    return;
  }

  if (eventType !== "content_block_delta" || !isRecord(payload.delta)) {
    return;
  }

  const deltaType = pickString(payload.delta.type);
  if (deltaType !== "thinking_delta" && deltaType !== "signature_delta") {
    return;
  }

  const existing = state.get(index) ?? {
    index,
    block: { type: "thinking", thinking: "" },
  };

  if (deltaType === "thinking_delta") {
    const thinking = pickString(payload.delta.thinking) ?? "";
    existing.block.thinking = `${pickString(existing.block.thinking) ?? ""}${thinking}`;
  } else {
    const signature = pickString(payload.delta.signature) ?? "";
    existing.block.signature = `${pickString(existing.block.signature) ?? ""}${signature}`;
  }

  state.set(index, existing);
}

function collectAnthropicStreamToolCalls(
  payload: Record<string, unknown>,
  state: Map<number, AnthropicStreamToolCallState>,
): void {
  const eventType = pickString(payload.type);
  if (!eventType) {
    return;
  }

  if (eventType === "content_block_start" && isRecord(payload.content_block)) {
    const block = payload.content_block;
    if (pickString(block.type) !== "tool_use") {
      return;
    }

    const index = pickContentBlockIndex(payload.index);
    const existing = state.get(index) ?? {
      index,
      argumentsText: "",
    };

    const id = pickString(block.id);
    if (id) {
      existing.id = id;
    }

    const name = pickString(block.name);
    if (name) {
      existing.name = name;
    }

    if ("input" in block) {
      existing.input = block.input;
    }

    state.set(index, existing);
    return;
  }

  if (eventType === "content_block_delta" && isRecord(payload.delta)) {
    const delta = payload.delta;
    if (pickString(delta.type) !== "input_json_delta") {
      return;
    }

    const index = pickContentBlockIndex(payload.index);
    const existing = state.get(index) ?? {
      index,
      argumentsText: "",
    };

    const partial = pickString(delta.partial_json);
    if (partial) {
      existing.argumentsText += partial;
    }

    state.set(index, existing);
  }
}

function pickContentBlockIndex(value: unknown): number {
  const numeric = toFiniteNumber(value);
  if (numeric !== undefined) {
    return Math.floor(numeric);
  }

  return 0;
}

function buildAnthropicStreamToolCalls(state: Map<number, AnthropicStreamToolCallState>): LLMToolCall[] {
  return [...state.values()]
    .sort((a, b) => a.index - b.index)
    .map((entry) => ({
      id: entry.id ?? "",
      type: "function",
      name: entry.name,
      arguments: entry.argumentsText.length > 0 ? entry.argumentsText : entry.input ?? {},
    }));
}

function buildAnthropicAssistantToolContent(
  text: string,
  toolCalls: LLMToolCall[],
  thinkingBlocks: Map<number, AnthropicStreamThinkingBlockState>,
): Array<Record<string, unknown>> {
  const content = [...thinkingBlocks.values()]
    .sort((a, b) => a.index - b.index)
    .map((entry) => ({ ...entry.block }));
  if (text.length > 0) {
    content.push({ type: "text", text });
  }

  for (const call of toolCalls) {
    const parsedArguments = typeof call.arguments === "string" ? parseToolArguments(call.arguments) : call.arguments;
    content.push({
      type: "tool_use",
      id: call.id,
      name: call.name,
      input: isRecord(parsedArguments) ? parsedArguments : {},
    });
  }

  return content;
}

function pickUsage(payload: Record<string, unknown>): LLMUsage | undefined {
  const fromUsage = extractUsageObject(payload.usage);
  if (fromUsage) {
    return fromUsage;
  }

  if (isRecord(payload.message)) {
    const nested = extractUsageObject(payload.message.usage);
    if (nested) {
      return nested;
    }
  }

  if (isRecord(payload.delta)) {
    const nested = extractUsageObject(payload.delta.usage);
    if (nested) {
      return nested;
    }
  }

  return undefined;
}

function pickFinishReason(payload: Record<string, unknown>): string | undefined {
  const direct = payload.stop_reason;
  if (typeof direct === "string" && direct.length > 0) {
    return direct;
  }

  if (isRecord(payload.delta)) {
    const reason = payload.delta.stop_reason;
    if (typeof reason === "string" && reason.length > 0) {
      return reason;
    }
  }

  if (isRecord(payload.message)) {
    const reason = payload.message.stop_reason;
    if (typeof reason === "string" && reason.length > 0) {
      return reason;
    }
  }

  return undefined;
}

function extractUsageObject(value: unknown): LLMUsage | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  return {
    inputTokens: toFiniteNumber(value.input_tokens ?? value.prompt_tokens),
    outputTokens: toFiniteNumber(value.output_tokens ?? value.completion_tokens),
    totalTokens: toFiniteNumber(value.total_tokens),
  };
}

function toAnthropicTools(tools: Array<Record<string, unknown>> | undefined): Array<Record<string, unknown>> | undefined {
  if (!Array.isArray(tools) || tools.length === 0) {
    return undefined;
  }

  return tools
    .filter((tool) => tool.type === "function" && isRecord(tool.function))
    .map((tool) => {
      const functionTool = tool.function as Record<string, unknown>;
      return {
        name: functionTool.name,
        description: functionTool.description,
        input_schema: functionTool.parameters ?? { type: "object", properties: {} },
      };
    });
}

function toAnthropicToolChoice(
  value: LLMRequest["toolChoice"],
  parallelToolCalls?: boolean,
): unknown {
  let choice: Record<string, unknown> | undefined;

  if (value === "required") {
    choice = { type: "any" };
  } else if (value === "auto" || value === "none") {
    choice = { type: value };
  }

  if (isRecord(value) && value.type === "function") {
    const maybeFn = value.function;
    if (isRecord(maybeFn)) {
      const name = pickString(maybeFn.name);
      if (name) {
        choice = { type: "tool", name };
      }
    }
  }

  if (!choice && parallelToolCalls !== undefined) {
    choice = { type: "auto" };
  }

  return choice
    ? {
        ...choice,
        ...(parallelToolCalls !== undefined && choice.type !== "none"
          ? { disable_parallel_tool_use: !parallelToolCalls }
          : {}),
      }
    : undefined;
}

function throwForAnthropicStreamError(payload: Record<string, unknown>): void {
  if (pickString(payload.type) !== "error") {
    return;
  }

  const error = isRecord(payload.error) ? payload.error : undefined;
  throw new Error(pickString(error?.message) ?? pickString(payload.message) ?? "Anthropic stream failed.");
}

function isAnthropicTerminalEvent(payload: Record<string, unknown>): boolean {
  return pickString(payload.type) === "message_stop";
}

function assertAnthropicStreamTerminated(terminated: boolean): void {
  if (!terminated) {
    throw new Error("Anthropic-compatible stream ended before a terminal event.");
  }
}
