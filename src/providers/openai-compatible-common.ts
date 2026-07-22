import type {
  EmbeddingRequest,
  EmbeddingResult,
  HTTPHeaders,
  LLMLogprobs,
  LLMRequest,
  LLMStreamCallbacks,
  LLMTokenLogprob,
  LLMTopLogprob,
  LLMToolCall,
  LLMUsage,
} from "../types";
import {
  buildURL,
  cleanUndefined,
  isRecord,
  pickString,
  toFiniteNumber,
} from "./utils";

export interface OpenAICompatibleAdapterOptions {
  baseURL: string;
  model: string;
  apiKey?: string;
  path?: string;
  embeddingPath?: string;
  defaultMaxToolRounds?: number;
  headers?: HTTPHeaders;
  defaultBody?: Record<string, unknown>;
  fetcher?: typeof fetch;
}

export type OpenAIReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

export async function embedOpenAI(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: EmbeddingRequest,
): Promise<EmbeddingResult> {
  // `encoding_format` is a default: callers targeting servers that reject it
  // (e.g. Voyage AI only accepts `base64` or none) can override or null it via
  // `defaultBody` / `request.body`.
  const body = cleanUndefined({
    encoding_format: "float",
    ...options.defaultBody,
    ...request.body,
    model: request.model ?? options.model,
    input: request.input,
    dimensions: request.dimensions,
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

export function validateTopLogprobs(value: number | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!Number.isInteger(value) || value < 0 || value > 20) {
    throw new RangeError("topLogprobs must be an integer between 0 and 20.");
  }
  return value;
}

export function sendOpenAIRequest(
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

export async function sendOpenAIJsonRequest(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: LLMRequest,
  body: Record<string, unknown>,
  parseErrorContext = "Failed to parse OpenAI-compatible JSON response",
): Promise<Record<string, unknown>> {
  const response = await sendOpenAIRequest(options, fetcher, path, request, body);

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  return parseOpenAICompatibleJSONResponse(response, parseErrorContext);
}

export function emitOpenAIStreamChunk(
  callbacks: LLMStreamCallbacks,
  round: number | undefined,
  raw: Record<string, unknown>,
  delta: string,
  reasoningDelta: string,
  usage: LLMUsage | undefined,
  finishReason: string | undefined,
  toolCalls?: LLMToolCall[],
  logprobs?: LLMLogprobs,
): void {
  if (delta || reasoningDelta || usage || finishReason || toolCalls || logprobs) {
    callbacks.onChunk?.({
      textDelta: delta,
      reasoningDelta: reasoningDelta || undefined,
      ...(round !== undefined ? { turnIndex: round } : {}),
      raw,
      usage,
      finishReason,
      toolCalls,
      logprobs,
    });
  }
}

export async function parseOpenAICompatibleJSONResponse(
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

export function formatResponseBodyForError(rawBody: string, maxLength = 2_000): string {
  const normalized = rawBody.trim();
  if (normalized.length === 0) {
    return "[empty body]";
  }

  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength)}...[truncated ${normalized.length - maxLength} chars]`;
}

export function buildHeaders(options: OpenAICompatibleAdapterOptions): HTTPHeaders {
  return {
    "content-type": "application/json",
    ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}),
    ...options.headers,
  };
}

export function toOpenAIReasoningEffort(value: LLMRequest["reasoningEffort"]): OpenAIReasoningEffort | undefined {
  return value;
}

export function normalizeLogprobEntries(value: unknown): LLMTokenLogprob[] | undefined {
  if (!Array.isArray(value) || value.length === 0) {
    return undefined;
  }
  const entries: LLMTokenLogprob[] = [];
  for (const item of value) {
    const base = normalizeLogprobBase(item);
    if (!base || !isRecord(item)) {
      continue;
    }
    const top = Array.isArray(item.top_logprobs)
      ? item.top_logprobs
          .map((alt) => normalizeLogprobBase(alt))
          .filter((alt): alt is LLMTopLogprob => alt !== undefined)
      : undefined;
    entries.push({ ...base, ...(top && top.length > 0 ? { top_logprobs: top } : {}) });
  }
  return entries.length > 0 ? entries : undefined;
}

export function normalizeLogprobBase(value: unknown): LLMTopLogprob | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const token = pickString(value.token);
  const logprob = toFiniteNumber(value.logprob);
  if (token === undefined || logprob === undefined) {
    return undefined;
  }
  return {
    token,
    logprob,
    ...normalizeLogprobBytes(value.bytes),
  };
}

export function normalizeLogprobBytes(value: unknown): { bytes?: number[] | null } {
  if (value === null) {
    return { bytes: null };
  }
  if (
    Array.isArray(value) &&
    value.every((byte) => Number.isInteger(byte) && byte >= 0 && byte <= 255)
  ) {
    return { bytes: value as number[] };
  }
  return {};
}

export function pickReasoningText(value: Record<string, unknown>): string {
  return pickTextLike(value.reasoning) || pickTextLike(value.reasoning_content);
}

export { joinReasoningBlocks, pushReasoningBlock } from "./utils";

export function pickTextLike(value: unknown): string {
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

export function pickTextLikePart(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  if (!isRecord(value)) {
    return "";
  }

  return pickString(value.text)
    ?? pickString(value.output_text)
    ?? pickString(value.refusal)
    ?? pickString(value.reasoning)
    ?? pickString(value.reasoning_content)
    ?? (Array.isArray(value.content) ? value.content.map((part) => pickTextLikePart(part)).join("") : "");
}

export function pickUsage(payload: Record<string, unknown>): LLMUsage | undefined {
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
