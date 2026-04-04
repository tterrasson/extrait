import { sanitizeThink } from "./think";
import { color, dim, title } from "./utils/debug-colors";
import type {
  LLMAdapter,
  LLMMessage,
  LLMRequest,
  LLMUsage,
  MCPToolClient,
  StructuredDebugOptions,
  StructuredPromptBuilder,
  StructuredPromptContext,
  StructuredPromptPayload,
  StructuredPromptResolver,
  StructuredPromptValue,
  StructuredTimeoutOptions,
  ThinkBlock,
} from "./types";
import { createOutdent } from "./outdent";
import { preferLatestUsage as preferLatestStreamUsage } from "./providers/utils";

const sharedOutdent = createOutdent({
  trimLeadingNewline: true,
  trimTrailingNewline: true,
  newline: "\n",
});

const RE_THINK_TAGS = /<\/?think\s*>/gi;

export type PromptRequestOptions = Omit<LLMRequest, "prompt" | "systemPrompt" | "messages">;

export interface StreamDelta {
  text: string;
  reasoning: string;
}

export interface NormalizedStreamConfig<TSnapshot> {
  enabled: boolean;
  onData?: (event: {
    delta: StreamDelta;
    snapshot: TSnapshot;
    done: boolean;
    usage?: LLMUsage;
    finishReason?: string;
  }) => void;
  to?: "stdout";
}

export interface NormalizedDebugConfig {
  enabled: boolean;
  colors: boolean;
  verbose: boolean;
  logger: (line: string) => void;
}

export interface NormalizedModelOutput {
  text: string;
  reasoning: string;
  thinkBlocks: ThinkBlock[];
  parseSource: string;
}

export interface ModelCallOptions<TSnapshot, TTraceEvent> {
  prompt?: string;
  messages?: LLMMessage[];
  systemPrompt?: string;
  request?: PromptRequestOptions;
  stream: NormalizedStreamConfig<TSnapshot>;
  observe?: (event: TTraceEvent) => void;
  buildEvent: (input: {
    stage: "llm.request" | "llm.response" | "llm.stream.delta" | "llm.stream.data";
    message: string;
    details?: unknown;
  }) => TTraceEvent;
  buildSnapshot: (input: NormalizedModelOutput) => TSnapshot;
  debug: NormalizedDebugConfig;
  debugLabel: string;
  attempt: number;
  selfHeal: boolean;
  selfHealEnabled: boolean;
  timeout?: StructuredTimeoutOptions;
}

export interface ModelCallResult {
  text: string;
  reasoning: string;
  thinkBlocks: ThinkBlock[];
  parseSource: string;
  via: "complete" | "stream";
  usage?: LLMUsage;
  finishReason?: string;
}

export function resolvePrompt(
  prompt: StructuredPromptBuilder,
  context: StructuredPromptContext,
): StructuredPromptPayload {
  const resolved = typeof prompt === "function" ? prompt(context) : prompt;
  return normalizePromptValue(resolved, context);
}

export function normalizePromptValue(
  value: StructuredPromptValue,
  _context: StructuredPromptContext,
): StructuredPromptPayload {
  if (typeof value === "string") {
    return {
      prompt: value,
    };
  }

  if (isPromptResolver(value)) {
    return normalizePromptPayload(value.resolvePrompt(_context));
  }

  return normalizePromptPayload(value);
}

export function normalizePromptPayload(value: StructuredPromptPayload): StructuredPromptPayload {
  const prompt = typeof value.prompt === "string" ? value.prompt : undefined;
  const messages = Array.isArray(value.messages) ? value.messages.filter(isLLMMessage) : undefined;

  if ((!prompt || prompt.trim().length === 0) && (!messages || messages.length === 0)) {
    throw new Error("Structured prompt payload must include a non-empty prompt or messages.");
  }

  return {
    prompt,
    systemPrompt: typeof value.systemPrompt === "string" ? value.systemPrompt : undefined,
    messages: messages && messages.length > 0 ? messages.map((message) => ({ ...message })) : undefined,
  };
}

export function applyPromptOutdent(payload: StructuredPromptPayload, enabled: boolean): StructuredPromptPayload {
  if (!enabled) {
    return payload;
  }

  return {
    prompt: typeof payload.prompt === "string" ? sharedOutdent.string(payload.prompt) : undefined,
    systemPrompt: applyOutdentToOptionalPrompt(payload.systemPrompt, enabled),
    messages: payload.messages?.map((message) => ({
      ...message,
      content: typeof message.content === "string" ? sharedOutdent.string(message.content) : message.content,
    })),
  };
}

export function applyOutdentToOptionalPrompt(value: string | undefined, enabled: boolean): string | undefined {
  if (!enabled || typeof value !== "string") {
    return value;
  }

  return sharedOutdent.string(value);
}

export function mergeSystemPrompts(primary?: string, secondary?: string): string | undefined {
  const prompts = [primary, secondary]
    .map((value) => value?.trim())
    .filter((value): value is string => Boolean(value));

  if (prompts.length === 0) {
    return undefined;
  }

  return prompts.join("\n\n");
}

export function normalizeStreamConfig<TSnapshot>(
  option:
    | boolean
    | {
        enabled?: boolean;
        onData?: NormalizedStreamConfig<TSnapshot>["onData"];
        to?: "stdout";
      }
    | undefined,
): NormalizedStreamConfig<TSnapshot> {
  if (typeof option === "boolean") {
    return {
      enabled: option,
    };
  }

  if (!option) {
    return {
      enabled: false,
    };
  }

  return {
    enabled: option.enabled ?? true,
    onData: option.onData,
    to: option.to,
  };
}

export function normalizeDebugConfig(
  option: StructuredDebugOptions | boolean | undefined,
): NormalizedDebugConfig {
  if (typeof option === "boolean") {
    return {
      enabled: option,
      colors: true,
      verbose: false,
      logger: (line: string) => console.log(line),
    };
  }

  if (!option) {
    return {
      enabled: false,
      colors: true,
      verbose: false,
      logger: (line: string) => console.log(line),
    };
  }

  return {
    enabled: option.enabled ?? true,
    colors: option.colors ?? true,
    verbose: option.verbose ?? false,
    logger: option.logger ?? ((line: string) => console.log(line)),
  };
}

export function withToolTimeout(client: MCPToolClient, toolTimeoutMs: number): MCPToolClient {
  return {
    id: client.id,
    listTools: client.listTools.bind(client),
    close: client.close?.bind(client),
    async callTool(params) {
      let timeoutId: ReturnType<typeof setTimeout> | undefined;
      const timeoutPromise = new Promise<never>((_, reject) => {
        timeoutId = setTimeout(
          () => reject(new Error(`Tool call timed out after ${toolTimeoutMs}ms`)),
          toolTimeoutMs,
        );
      });
      try {
        return await Promise.race([client.callTool(params), timeoutPromise]);
      } finally {
        clearTimeout(timeoutId);
      }
    },
  };
}

export function applyToolTimeout(clients: MCPToolClient[], toolTimeoutMs: number): MCPToolClient[] {
  return clients.map((client) => withToolTimeout(client, toolTimeoutMs));
}

export async function callModel<TSnapshot, TTraceEvent>(
  adapter: LLMAdapter,
  options: ModelCallOptions<TSnapshot, TTraceEvent>,
): Promise<ModelCallResult> {
  const requestSignal =
    options.request?.signal ??
    (options.timeout?.request !== undefined
      ? AbortSignal.timeout(options.timeout.request)
      : undefined);

  const requestPayload: LLMRequest = {
    prompt: options.prompt,
    messages: options.messages,
    systemPrompt: options.systemPrompt,
    temperature: options.request?.temperature,
    maxTokens: options.request?.maxTokens,
    mcpClients: options.request?.mcpClients,
    toolChoice: options.request?.toolChoice,
    parallelToolCalls: options.request?.parallelToolCalls,
    maxToolRounds: options.request?.maxToolRounds,
    onToolExecution: options.request?.onToolExecution,
    transformToolOutput: options.request?.transformToolOutput,
    transformToolArguments: options.request?.transformToolArguments,
    transformToolCallParams: options.request?.transformToolCallParams,
    unknownToolError: options.request?.unknownToolError,
    toolDebug: options.request?.toolDebug,
    body: options.request?.body,
    signal: requestSignal,
  };

  emitDebugRequest(options.debug, {
    label: options.debugLabel,
    provider: adapter.provider,
    model: adapter.model,
    attempt: options.attempt,
    selfHealAttempt: options.selfHeal,
    selfHealEnabled: options.selfHealEnabled,
    stream: options.stream.enabled && !!adapter.stream,
    requestPayload,
  });

  options.observe?.(
    options.buildEvent({
      stage: "llm.request",
      message: "Sending LLM request.",
      details: {
        provider: adapter.provider,
        model: adapter.model,
        stream: options.stream.enabled && !!adapter.stream,
      },
    }),
  );

  if (options.stream.enabled && adapter.stream) {
    let latestUsage: LLMUsage | undefined;
    let latestFinishReason: string | undefined;
    let streamedProviderText = "";
    let streamedDedicatedReasoning = "";
    let lastSnapshotFingerprint: string | undefined;
    let previousSnapshotText = "";
    let previousSnapshotReasoning = "";

    const emitStreamingData = (
      done: boolean,
      usage?: LLMUsage,
      finishReason?: string,
    ): void => {
      const normalized = normalizeModelOutput(streamedProviderText, streamedDedicatedReasoning);
      const snapshot = options.buildSnapshot(normalized);
      const fingerprint = toStreamDataFingerprint(snapshot);
      if (!done && fingerprint === lastSnapshotFingerprint) {
        return;
      }

      const delta = {
        text: normalized.text.startsWith(previousSnapshotText)
          ? normalized.text.slice(previousSnapshotText.length)
          : "",
        reasoning: normalized.reasoning.startsWith(previousSnapshotReasoning)
          ? normalized.reasoning.slice(previousSnapshotReasoning.length)
          : "",
      };

      lastSnapshotFingerprint = fingerprint;
      previousSnapshotText = normalized.text;
      previousSnapshotReasoning = normalized.reasoning;
      options.stream.onData?.({
        delta,
        snapshot,
        done,
        usage,
        finishReason,
      });

      if (options.stream.to === "stdout" && delta.text) {
        process.stdout.write(delta.text);
      }

      options.observe?.(
        options.buildEvent({
          stage: "llm.stream.data",
          message: done ? "Streaming response completed." : "Streaming response updated.",
          details: {
            done,
            finishReason,
          },
        }),
      );
    };

    const handleTextDelta = (delta: string): void => {
      if (!delta) {
        return;
      }

      streamedProviderText += delta;

      options.observe?.(
        options.buildEvent({
          stage: "llm.stream.delta",
          message: "Received stream delta.",
          details: {
            chars: delta.length,
          },
        }),
      );

      emitStreamingData(false);
    };

    const handleReasoningDelta = (delta: string): void => {
      if (!delta) {
        return;
      }

      streamedDedicatedReasoning += delta;
      emitStreamingData(false);
    };

    const response = await adapter.stream(requestPayload, {
      onChunk: (chunk) => {
        if (chunk.textDelta) {
          handleTextDelta(chunk.textDelta);
        }

        if (chunk.reasoningDelta) {
          handleReasoningDelta(chunk.reasoningDelta);
        }

        if (chunk.usage) {
          latestUsage = preferLatestStreamUsage(latestUsage, chunk.usage);
        }

        if (chunk.finishReason) {
          latestFinishReason = chunk.finishReason;
        }
      },
    });

    streamedProviderText =
      typeof response.text === "string" ? response.text : streamedProviderText;
    streamedDedicatedReasoning =
      typeof response.reasoning === "string" ? response.reasoning : streamedDedicatedReasoning;
    const finalNormalized = normalizeModelOutput(streamedProviderText, streamedDedicatedReasoning);
    const usage = preferLatestStreamUsage(latestUsage, response.usage);
    const finishReason = response.finishReason ?? latestFinishReason;
    emitStreamingData(true, usage, finishReason);

    options.observe?.(
      options.buildEvent({
        stage: "llm.response",
        message: "Streaming response completed.",
        details: {
          via: "stream",
          chars: finalNormalized.parseSource.length,
          finishReason,
        },
      }),
    );

    emitDebugResponse(options.debug, {
      label: options.debugLabel,
      attempt: options.attempt,
      selfHealAttempt: options.selfHeal,
      selfHealEnabled: options.selfHealEnabled,
      via: "stream",
      text: finalNormalized.text,
      reasoning: finalNormalized.reasoning,
      parseSource: finalNormalized.parseSource,
      usage,
      finishReason,
    });

    return {
      text: finalNormalized.text,
      reasoning: finalNormalized.reasoning,
      thinkBlocks: finalNormalized.thinkBlocks,
      parseSource: finalNormalized.parseSource,
      via: "stream",
      usage,
      finishReason,
    };
  }

  const response = await adapter.complete(requestPayload);
  const normalized = normalizeModelOutput(response.text, response.reasoning);

  options.observe?.(
    options.buildEvent({
      stage: "llm.response",
      message: "Completion response received.",
      details: {
        via: "complete",
        chars: normalized.parseSource.length,
        finishReason: response.finishReason,
      },
    }),
  );

  emitDebugResponse(options.debug, {
    label: options.debugLabel,
    attempt: options.attempt,
    selfHealAttempt: options.selfHeal,
    selfHealEnabled: options.selfHealEnabled,
    via: "complete",
    text: normalized.text,
    reasoning: normalized.reasoning,
    parseSource: normalized.parseSource,
    usage: response.usage,
    finishReason: response.finishReason,
  });

  return {
    text: normalized.text,
    reasoning: normalized.reasoning,
    thinkBlocks: normalized.thinkBlocks,
    parseSource: normalized.parseSource,
    via: "complete",
    usage: response.usage,
    finishReason: response.finishReason,
  };
}

export function normalizeModelOutput(text: string, dedicatedReasoning?: string): NormalizedModelOutput {
  const sanitized = sanitizeThink(text);
  const visibleText = stripThinkBlocks(text, sanitized.thinkBlocks);
  const reasoning = joinReasoningSegments([
    dedicatedReasoning,
    ...sanitized.thinkBlocks.map((block) => block.content),
  ]);

  return {
    text: visibleText,
    reasoning,
    thinkBlocks: sanitized.thinkBlocks,
    parseSource: composeParseSource(visibleText, reasoning),
  };
}

export function composeParseSource(text: string, reasoning?: string): string {
  if (typeof reasoning !== "string" || reasoning.length === 0) {
    return text;
  }

  const sanitized = reasoning.replace(RE_THINK_TAGS, "");
  if (sanitized.length === 0) {
    return text;
  }

  return `<think>${sanitized}</think>${text}`;
}

export function aggregateUsage<T extends { usage?: LLMUsage }>(attempts: T[]): LLMUsage | undefined {
  let usage: LLMUsage | undefined;

  for (const attempt of attempts) {
    usage = mergeUsage(usage, attempt.usage);
  }

  return usage;
}

export function mergeUsage(base: LLMUsage | undefined, next: LLMUsage | undefined): LLMUsage | undefined {
  if (!base && !next) {
    return undefined;
  }

  return {
    inputTokens: (base?.inputTokens ?? 0) + (next?.inputTokens ?? 0),
    outputTokens: (base?.outputTokens ?? 0) + (next?.outputTokens ?? 0),
    totalTokens: (base?.totalTokens ?? 0) + (next?.totalTokens ?? 0),
    cost: (base?.cost ?? 0) + (next?.cost ?? 0),
  };
}

function isPromptResolver(value: StructuredPromptValue): value is StructuredPromptResolver {
  return (
    typeof value === "object" &&
    value !== null &&
    "resolvePrompt" in value &&
    typeof value.resolvePrompt === "function"
  );
}

function isLLMMessage(value: unknown): value is LLMMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const candidate = value as Partial<LLMMessage>;
  if (
    candidate.role !== "system" &&
    candidate.role !== "user" &&
    candidate.role !== "assistant" &&
    candidate.role !== "tool"
  ) {
    return false;
  }

  return "content" in candidate;
}

function joinReasoningSegments(parts: Array<string | undefined>): string {
  return parts
    .map((value) => value?.trim())
    .filter((value): value is string => Boolean(value))
    .join("\n\n");
}

function stripThinkBlocks(text: string, thinkBlocks: ThinkBlock[]): string {
  if (thinkBlocks.length === 0) {
    return text;
  }

  let output = "";
  let cursor = 0;

  for (const block of thinkBlocks) {
    output += text.slice(cursor, block.start);
    cursor = block.end;
  }

  output += text.slice(cursor);
  return output;
}

function toStreamDataFingerprint(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "__unserializable__";
  }
}

interface DebugRequestInput {
  label: string;
  provider?: string;
  model?: string;
  attempt: number;
  selfHealAttempt: boolean;
  selfHealEnabled: boolean;
  stream: boolean;
  requestPayload: LLMRequest;
}

interface DebugResponseInput {
  label: string;
  attempt: number;
  selfHealAttempt: boolean;
  selfHealEnabled: boolean;
  via: "complete" | "stream";
  text: string;
  reasoning: string;
  parseSource: string;
  usage?: LLMUsage;
  finishReason?: string;
}

function emitDebugRequest(
  config: NormalizedDebugConfig,
  input: DebugRequestInput,
): void {
  const requestBody =
    input.requestPayload.body !== undefined
      ? JSON.stringify(input.requestPayload.body, null, 2)
      : "(none)";
  const requestMessages =
    input.requestPayload.messages !== undefined
      ? JSON.stringify(input.requestPayload.messages, null, 2)
      : "(none)";

  const lines = [
    color(
      config,
      title(
        config,
        [
          `[${input.label}][request]`,
          `attempt=${input.attempt}`,
          `selfHealEnabled=${input.selfHealEnabled}`,
          `selfHealAttempt=${input.selfHealAttempt}`,
        ].join(" "),
      ),
      "cyan",
    ),
    dim(
      config,
      [
        `provider=${input.provider ?? "unknown"}`,
        `model=${input.model ?? "unknown"}`,
        `stream=${input.stream}`,
      ].join(" "),
    ),
    color(config, "prompt:", "yellow"),
    input.requestPayload.prompt ?? "(none)",
    color(config, "messages:", "yellow"),
    requestMessages,
    color(config, "systemPrompt:", "yellow"),
    input.requestPayload.systemPrompt ?? "(none)",
    color(config, "request.body:", "yellow"),
    requestBody,
  ];

  emitDebug(config, lines.join("\n"));
}

function emitDebugResponse(
  config: NormalizedDebugConfig,
  input: DebugResponseInput,
): void {
  const text = input.text.length > 0 ? input.text : "(none)";
  const reasoning = input.reasoning.length > 0 ? input.reasoning : "(none)";
  const metadata = [
    `via=${input.via}`,
    `textChars=${input.text.length}`,
    `reasoningChars=${input.reasoning.length}`,
  ];
  if (config.verbose) {
    metadata.push(`parseSourceChars=${input.parseSource.length}`);
  }
  metadata.push(
    `finishReason=${input.finishReason ?? "unknown"}`,
    `usage=${JSON.stringify(input.usage ?? {})}`,
  );
  const lines = [
    color(
      config,
      title(
        config,
        [
          `[${input.label}][response]`,
          `attempt=${input.attempt}`,
          `selfHealEnabled=${input.selfHealEnabled}`,
          `selfHealAttempt=${input.selfHealAttempt}`,
        ].join(" "),
      ),
      "green",
    ),
    dim(config, metadata.join(" ")),
    color(config, "text:", "yellow"),
    text,
    color(config, "reasoning:", "yellow"),
    reasoning,
  ];
  if (config.verbose) {
    lines.push(color(config, "parseSource:", "yellow"), input.parseSource);
  }

  emitDebug(config, lines.join("\n"));
}

function emitDebug(config: NormalizedDebugConfig, message: string): void {
  if (!config.enabled) {
    return;
  }

  config.logger(message);
}
