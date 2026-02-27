import type {
  HTTPHeaders,
  LLMAdapter,
  LLMRequest,
  LLMResponse,
  LLMStreamCallbacks,
  LLMStreamChunk,
  LLMToolCall,
  LLMUsage,
} from "../types";
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
import { buildURL, cleanUndefined, isRecord, mergeUsage, toFiniteNumber, pickString, safeJSONParse } from "./utils";

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

      const response = await fetcher(buildURL(options.baseURL, path), {
        method: "POST",
        headers: buildHeaders(options),
        body: JSON.stringify(
          cleanUndefined({
            ...options.defaultBody,
            ...request.body,
            model: options.model,
            system: request.systemPrompt,
            messages: [{ role: "user", content: request.prompt }],
            temperature: request.temperature,
            max_tokens: resolveMaxTokens(request.maxTokens, options.defaultMaxTokens),
            stream: true,
          }),
        ),
        signal: request.signal,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(`HTTP ${response.status}: ${message}`);
      }

      callbacks.onStart?.();
      let text = "";
      let usage: LLMUsage | undefined;
      let finishReason: string | undefined;

      await consumeSSE(response, (data) => {
        if (data === "[DONE]") {
          return;
        }

        const json = safeJSONParse(data);
        if (!isRecord(json)) {
          return;
        }

        const delta = pickAnthropicDelta(json);
        const chunkUsage = pickUsage(json);
        const chunkFinishReason = pickFinishReason(json);

        usage = mergeUsage(usage, chunkUsage);
        if (chunkFinishReason) {
          finishReason = chunkFinishReason;
        }

        if (delta) {
          text += delta;
          callbacks.onToken?.(delta);
        }

        if (delta || chunkUsage || chunkFinishReason) {
          const chunk: LLMStreamChunk = {
            textDelta: delta,
            raw: json,
            usage: chunkUsage,
            finishReason: chunkFinishReason,
          };
          callbacks.onChunk?.(chunk);
        }
      });

      const out = { text, usage, finishReason };
      callbacks.onComplete?.(out);
      return out;
    },
  };
}

async function completePassThrough(
  options: AnthropicCompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
): Promise<LLMResponse> {
  const response = await fetcher(buildURL(options.baseURL, path), {
    method: "POST",
    headers: buildHeaders(options),
    body: JSON.stringify(
      cleanUndefined({
        ...options.defaultBody,
        ...request.body,
        model: options.model,
        system: request.systemPrompt,
        messages: [{ role: "user", content: request.prompt }],
        temperature: request.temperature,
        max_tokens: resolveMaxTokens(request.maxTokens, options.defaultMaxTokens),
        stream: false,
      }),
    ),
    signal: request.signal,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  const data = (await response.json()) as Record<string, unknown>;
  const text = extractAnthropicText(data);
  const toolCalls = pickAnthropicToolCalls(data);

  if (!text && toolCalls.length === 0) {
    throw new Error("No assistant text in Anthropic-compatible response.");
  }

  return {
    text,
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

  let messages: Array<Record<string, unknown>> = [{ role: "user", content: request.prompt }];
  let aggregatedUsage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  const toolCalls: LLMToolCall[] = [];
  const toolExecutions: NonNullable<LLMResponse["toolExecutions"]> = [];

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const tools = toAnthropicTools(toProviderFunctionTools(mcpToolset));

    const response = await fetcher(buildURL(options.baseURL, path), {
      method: "POST",
      headers: buildHeaders(options),
      body: JSON.stringify(
        cleanUndefined({
          ...options.defaultBody,
          ...request.body,
          model: options.model,
          system: request.systemPrompt,
          messages,
          temperature: request.temperature,
          max_tokens: resolveMaxTokens(request.maxTokens, options.defaultMaxTokens),
          tools,
          tool_choice: toAnthropicToolChoice(request.toolChoice),
          stream: false,
        }),
      ),
      signal: request.signal,
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(`HTTP ${response.status}: ${message}`);
    }

    const payload = (await response.json()) as Record<string, unknown>;
    lastPayload = payload;
    aggregatedUsage = mergeUsage(aggregatedUsage, pickUsage(payload));
    finishReason = pickFinishReason(payload);

    const content = Array.isArray(payload.content) ? payload.content : [];
    const calledTools = pickAnthropicToolCalls(payload).filter((call) => call.type === "function");

    if (calledTools.length === 0) {
      return {
        text: extractAnthropicText(payload),
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

  let messages: Array<Record<string, unknown>> = [{ role: "user", content: request.prompt }];
  let aggregatedUsage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  const toolCalls: LLMToolCall[] = [];
  const toolExecutions: NonNullable<LLMResponse["toolExecutions"]> = [];

  callbacks.onStart?.();

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const tools = toAnthropicTools(toProviderFunctionTools(mcpToolset));

    const response = await fetcher(buildURL(options.baseURL, path), {
      method: "POST",
      headers: buildHeaders(options),
      body: JSON.stringify(
        cleanUndefined({
          ...options.defaultBody,
          ...request.body,
          model: options.model,
          system: request.systemPrompt,
          messages,
          temperature: request.temperature,
          max_tokens: resolveMaxTokens(request.maxTokens, options.defaultMaxTokens),
          tools,
          tool_choice: toAnthropicToolChoice(request.toolChoice),
          stream: true,
        }),
      ),
      signal: request.signal,
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(`HTTP ${response.status}: ${message}`);
    }

    let roundText = "";
    let roundUsage: LLMUsage | undefined;
    let roundFinishReason: string | undefined;
    const streamedToolCalls = new Map<number, AnthropicStreamToolCallState>();

    await consumeSSE(response, (data) => {
      if (data === "[DONE]") {
        return;
      }

      const json = safeJSONParse(data);
      if (!isRecord(json)) {
        return;
      }

      lastPayload = json;

      const delta = pickAnthropicDelta(json);
      const chunkUsage = pickUsage(json);
      const chunkFinishReason = pickFinishReason(json);

      collectAnthropicStreamToolCalls(json, streamedToolCalls);
      roundUsage = mergeUsage(roundUsage, chunkUsage);
      if (chunkFinishReason) {
        roundFinishReason = chunkFinishReason;
      }

      if (delta) {
        roundText += delta;
        callbacks.onToken?.(delta);
      }

      if (delta || chunkUsage || chunkFinishReason) {
        const chunk: LLMStreamChunk = {
          textDelta: delta,
          raw: json,
          usage: chunkUsage,
          finishReason: chunkFinishReason,
        };
        callbacks.onChunk?.(chunk);
      }
    });

    aggregatedUsage = mergeUsage(aggregatedUsage, roundUsage);
    if (roundFinishReason) {
      finishReason = roundFinishReason;
    }

    const calledTools = buildAnthropicStreamToolCalls(streamedToolCalls);
    if (calledTools.length === 0) {
      const out: LLMResponse = {
        text: roundText,
        raw: lastPayload,
        usage: aggregatedUsage,
        finishReason,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
      };
      callbacks.onComplete?.(out);
      return out;
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
      { role: "assistant", content: buildAnthropicAssistantToolContent(roundText, calledTools) },
      { role: "user", content: toolResultContent },
    ];
  }

  const out: LLMResponse = {
    text: "",
    raw: lastPayload,
    usage: aggregatedUsage,
    finishReason,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
  };
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

interface AnthropicStreamToolCallState {
  index: number;
  id?: string;
  name?: string;
  input?: unknown;
  argumentsText: string;
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

function buildAnthropicAssistantToolContent(text: string, toolCalls: LLMToolCall[]): Array<Record<string, unknown>> {
  const content: Array<Record<string, unknown>> = [];
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

function toAnthropicToolChoice(value: LLMRequest["toolChoice"]): unknown {
  if (value === undefined) {
    return undefined;
  }

  if (value === "required") {
    return { type: "any" };
  }

  if (isRecord(value) && value.type === "function") {
    const maybeFn = value.function;
    if (isRecord(maybeFn)) {
      const name = pickString(maybeFn.name);
      if (name) {
        return { type: "tool", name };
      }
    }
  }

  return value;
}
