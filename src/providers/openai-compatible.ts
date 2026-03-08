import type {
  EmbeddingRequest,
  EmbeddingResult,
  HTTPHeaders,
  LLMAdapter,
  LLMMessage,
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
  resolveMCPToolset,
  stringifyToolOutput,
  toProviderFunctionTools,
} from "./mcp-runtime";
import { buildURL, cleanUndefined, isRecord, mergeUsage, pickString, safeJSONParse, toFiniteNumber } from "./utils";

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

      const response = await fetcher(buildURL(options.baseURL, path), {
        method: "POST",
        headers: buildHeaders(options),
        body: JSON.stringify(
          cleanUndefined({
            ...options.defaultBody,
            ...request.body,
            model: options.model,
            messages: buildMessages(request),
            temperature: request.temperature,
            max_tokens: request.maxTokens,
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

        const delta = pickAssistantDelta(json);
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

    async embed(request: EmbeddingRequest): Promise<EmbeddingResult> {
      const body = cleanUndefined({
        ...options.defaultBody,
        ...request.body,
        model: request.model ?? options.model,
        input: request.input,
        dimensions: request.dimensions,
        encoding_format: "float",
      });

      const response = await fetcher(buildURL(options.baseURL, embeddingPath), {
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
        embeddings: data.map((d: unknown) => (isRecord(d) && Array.isArray(d.embedding) ? (d.embedding as number[]) : [])),
        model: pickString(json.model) ?? (body.model as string),
        usage: pickUsage(json),
        raw: json,
      };
    },
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

async function completeWithChatCompletionsPassThrough(
  options: OpenAICompatibleAdapterOptions,
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
        messages: buildMessages(request),
        temperature: request.temperature,
        max_tokens: request.maxTokens,
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
  const assistantMessage = pickAssistantMessage(payload);
  if (!assistantMessage) {
    throw new Error("No assistant message in OpenAI-compatible response.");
  }

  const toolCalls = pickChatToolCalls(payload);
  return {
    text: pickAssistantText(payload),
    raw: payload,
    usage: pickUsage(payload),
    finishReason: pickFinishReason(payload),
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

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const transportTools = toProviderFunctionTools(mcpToolset);

    const response = await fetcher(buildURL(options.baseURL, path), {
      method: "POST",
      headers: buildHeaders(options),
      body: JSON.stringify(
        cleanUndefined({
          ...options.defaultBody,
          ...request.body,
          model: options.model,
          messages,
          temperature: request.temperature,
          max_tokens: request.maxTokens,
          tools: transportTools,
          tool_choice: request.toolChoice,
          parallel_tool_calls: request.parallelToolCalls,
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

    const assistantMessage = pickAssistantMessage(payload);
    const calledTools = pickChatToolCalls(payload);

    if (!assistantMessage) {
      throw new Error("No assistant message in OpenAI-compatible response.");
    }

    if (calledTools.length === 0) {
      return {
        text: pickAssistantText(payload),
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
  const response = await fetcher(buildURL(options.baseURL, path), {
    method: "POST",
    headers: buildHeaders(options),
    body: JSON.stringify(
      cleanUndefined({
        ...options.defaultBody,
        ...request.body,
        model: options.model,
        input: buildResponsesInput(request),
        previous_response_id: pickString(body?.previous_response_id),
        temperature: request.temperature,
        max_output_tokens: request.maxTokens,
      }),
    ),
    signal: request.signal,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  const payload = (await response.json()) as Record<string, unknown>;
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

  let input = buildResponsesInput(request);
  let previousResponseId = pickString(
    isRecord(request.body) ? (request.body.previous_response_id as unknown) : undefined,
  );
  let aggregatedUsage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  const executedToolCalls: LLMToolCall[] = [];
  const toolExecutions: NonNullable<LLMResponse["toolExecutions"]> = [];

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const transportTools = toResponsesTools(toProviderFunctionTools(mcpToolset));

    const response = await fetcher(buildURL(options.baseURL, path), {
      method: "POST",
      headers: buildHeaders(options),
      body: JSON.stringify(
        cleanUndefined({
          ...options.defaultBody,
          ...request.body,
          model: options.model,
          input,
          previous_response_id: previousResponseId,
          temperature: request.temperature,
          max_output_tokens: request.maxTokens,
          tools: transportTools,
          tool_choice: request.toolChoice,
          parallel_tool_calls: request.parallelToolCalls,
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
    finishReason = pickResponsesFinishReason(payload) ?? finishReason;

    const providerToolCalls = pickResponsesToolCalls(payload);
    const functionCalls = providerToolCalls.filter(
      (toolCall): toolCall is LLMToolCall & { id: string; name: string } =>
        toolCall.type === "function" && typeof toolCall.id === "string" && typeof toolCall.name === "string",
    );

    if (functionCalls.length === 0) {
      const text = pickResponsesText(payload) || pickAssistantText(payload);
      return {
        text,
        raw: payload,
        usage: aggregatedUsage,
        finishReason,
        toolCalls: executedToolCalls.length > 0 ? executedToolCalls : undefined,
        toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
      };
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
    executedToolCalls.push(...outputs.map((entry) => entry.call));
    toolExecutions.push(...outputs.map((entry) => entry.execution));

    input = outputs.map((entry) => ({
      type: "function_call_output",
      call_id: entry.call.id,
      output: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
    }));
    previousResponseId = pickString(payload.id);
  }

  return {
    text: pickResponsesText(lastPayload ?? {}) || pickAssistantText(lastPayload ?? {}),
    raw: lastPayload,
    usage: aggregatedUsage,
    finishReason,
    toolCalls: executedToolCalls.length > 0 ? executedToolCalls : undefined,
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

  callbacks.onStart?.();

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const transportTools = toProviderFunctionTools(mcpToolset);

    const response = await fetcher(buildURL(options.baseURL, path), {
      method: "POST",
      headers: buildHeaders(options),
      body: JSON.stringify(
        cleanUndefined({
          ...options.defaultBody,
          ...request.body,
          model: options.model,
          messages,
          temperature: request.temperature,
          max_tokens: request.maxTokens,
          tools: transportTools,
          tool_choice: request.toolChoice,
          parallel_tool_calls: request.parallelToolCalls,
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
    const streamedToolCalls = new Map<number, OpenAIStreamToolCallState>();

    await consumeSSE(response, (data) => {
      if (data === "[DONE]") {
        return;
      }

      const json = safeJSONParse(data);
      if (!isRecord(json)) {
        return;
      }

      lastPayload = json;

      const delta = pickAssistantDelta(json);
      const chunkUsage = pickUsage(json);
      const chunkFinishReason = pickFinishReason(json);

      collectOpenAIStreamToolCalls(json, streamedToolCalls);
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

    const calledTools = buildOpenAIStreamToolCalls(streamedToolCalls);
    if (calledTools.length === 0) {
      const out: LLMResponse = {
        text: roundText,
        raw: lastPayload,
        usage: aggregatedUsage,
        finishReason,
        toolCalls: executedToolCalls.length > 0 ? executedToolCalls : undefined,
        toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
      };
      callbacks.onComplete?.(out);
      return out;
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
    executedToolCalls.push(...outputs.map((entry) => entry.call));
    toolExecutions.push(...outputs.map((entry) => entry.execution));

    const assistantMessage = buildOpenAIAssistantToolMessage(roundText, calledTools);
    const toolMessages = outputs.map((entry) => ({
      role: "tool",
      tool_call_id: entry.call.id,
      content: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
    }));
    messages = [...messages, assistantMessage, ...toolMessages];
  }

  const out: LLMResponse = {
    text: "",
    raw: lastPayload,
    usage: aggregatedUsage,
    finishReason,
    toolCalls: executedToolCalls.length > 0 ? executedToolCalls : undefined,
    toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
  };
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
  const response = await fetcher(buildURL(options.baseURL, path), {
    method: "POST",
    headers: buildHeaders(options),
    body: JSON.stringify(
      cleanUndefined({
        ...options.defaultBody,
        ...request.body,
        model: options.model,
        input: buildResponsesInput(request),
        previous_response_id: pickString(body?.previous_response_id),
        temperature: request.temperature,
        max_output_tokens: request.maxTokens,
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
  let lastPayload: Record<string, unknown> | undefined;

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

  const finalPayload = lastPayload ?? {};
  const out: LLMResponse = {
    text: text.length > 0 ? text : (pickResponsesText(finalPayload) || pickAssistantText(finalPayload)),
    raw: finalPayload,
    usage: mergeUsage(usage, pickUsage(finalPayload)),
    finishReason: finishReason ?? pickResponsesFinishReason(finalPayload) ?? pickFinishReason(finalPayload),
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

  let input = buildResponsesInput(request);
  let previousResponseId = pickString(
    isRecord(request.body) ? (request.body.previous_response_id as unknown) : undefined,
  );
  let aggregatedUsage: LLMUsage | undefined;
  let finishReason: string | undefined;
  let lastPayload: Record<string, unknown> | undefined;
  const executedToolCalls: LLMToolCall[] = [];
  const toolExecutions: NonNullable<LLMResponse["toolExecutions"]> = [];

  callbacks.onStart?.();

  for (let round = 1; round <= maxToolRounds + 1; round += 1) {
    const mcpToolset = await resolveMCPToolset(request.mcpClients);
    const transportTools = toResponsesTools(toProviderFunctionTools(mcpToolset));

    const response = await fetcher(buildURL(options.baseURL, path), {
      method: "POST",
      headers: buildHeaders(options),
      body: JSON.stringify(
        cleanUndefined({
          ...options.defaultBody,
          ...request.body,
          model: options.model,
          input,
          previous_response_id: previousResponseId,
          temperature: request.temperature,
          max_output_tokens: request.maxTokens,
          tools: transportTools,
          tool_choice: request.toolChoice,
          parallel_tool_calls: request.parallelToolCalls,
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
        lastPayload = payload;
      }

      const delta = pickResponsesStreamTextDelta(json);
      const chunkUsage = pickResponsesStreamUsage(json);
      const chunkFinishReason = pickResponsesStreamFinishReason(json);

      collectResponsesStreamToolCalls(json, streamedToolCalls);
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
    const payloadUsage = roundPayload ? pickUsage(roundPayload) : undefined;
    aggregatedUsage = mergeUsage(aggregatedUsage, payloadUsage);
    if (roundFinishReason) {
      finishReason = roundFinishReason;
    } else if (roundPayload) {
      finishReason = pickResponsesFinishReason(roundPayload) ?? finishReason;
    }

    const payloadToolCalls = roundPayload ? pickResponsesToolCalls(roundPayload) : [];
    const streamedCalls = buildResponsesStreamToolCalls(streamedToolCalls);
    const providerToolCalls = payloadToolCalls.length > 0 ? payloadToolCalls : streamedCalls;
    const functionCalls = providerToolCalls.filter(
      (toolCall): toolCall is LLMToolCall & { id: string; name: string } =>
        toolCall.type === "function" && typeof toolCall.id === "string" && typeof toolCall.name === "string",
    );

    if (functionCalls.length === 0) {
      const finalText = roundText.length > 0
        ? roundText
        : (roundPayload ? (pickResponsesText(roundPayload) || pickAssistantText(roundPayload)) : "");
      const out: LLMResponse = {
        text: finalText,
        raw: roundPayload ?? lastPayload,
        usage: aggregatedUsage,
        finishReason,
        toolCalls: executedToolCalls.length > 0 ? executedToolCalls : undefined,
        toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
      };
      callbacks.onComplete?.(out);
      return out;
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
    executedToolCalls.push(...outputs.map((entry) => entry.call));
    toolExecutions.push(...outputs.map((entry) => entry.execution));

    input = outputs.map((entry) => ({
      type: "function_call_output",
      call_id: entry.call.id,
      output: stringifyToolOutput(entry.call.error ? { error: entry.call.error } : entry.call.output),
    }));
    previousResponseId = pickString(roundPayload?.id);
  }

  const out: LLMResponse = {
    text: pickResponsesText(lastPayload ?? {}) || pickAssistantText(lastPayload ?? {}),
    raw: lastPayload,
    usage: aggregatedUsage,
    finishReason,
    toolCalls: executedToolCalls.length > 0 ? executedToolCalls : undefined,
    toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
  };
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

  const content = delta.content;
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (!isRecord(part)) {
          return "";
        }
        const text = part.text;
        return typeof text === "string" ? text : "";
      })
      .join("");
  }

  return "";
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

function collectOpenAIStreamToolCalls(
  payload: Record<string, unknown>,
  state: Map<number, OpenAIStreamToolCallState>,
): void {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0 || !isRecord(choices[0])) {
    return;
  }

  const delta = choices[0].delta;
  if (!isRecord(delta) || !Array.isArray(delta.tool_calls)) {
    return;
  }

  for (const rawToolCall of delta.tool_calls) {
    if (!isRecord(rawToolCall)) {
      continue;
    }

    const index = toFiniteNumber(rawToolCall.index);
    const toolIndex = index !== undefined ? Math.floor(index) : 0;
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
      existing.argumentsText += argumentsDelta;
    }

    state.set(toolIndex, existing);
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

function buildOpenAIAssistantToolMessage(text: string, toolCalls: LLMToolCall[]): Record<string, unknown> {
  return {
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

function pickAssistantText(payload: Record<string, unknown>): string {
  const message = pickAssistantMessage(payload);
  if (message) {
    const content = message.content;

    if (typeof content === "string") {
      return content;
    }

    if (Array.isArray(content)) {
      return content
        .map((part) => {
          if (typeof part === "string") {
            return part;
          }

          if (!isRecord(part)) {
            return "";
          }

          const text = part.text;
          return typeof text === "string" ? text : "";
        })
        .join("");
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
