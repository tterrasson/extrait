import type {
  LLMLogprobs,
  LLMMessage,
  LLMRequest,
  LLMToolCall,
  LLMUsage,
  ReasoningBlock,
  StreamTurnTransition,
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

const sharedOutdent = createOutdent({
  trimLeadingNewline: true,
  trimTrailingNewline: true,
  newline: "\n",
});


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
    turnIndex?: number;
    toolCalls?: LLMToolCall[];
  }) => void;
  onTurnTransition?: (transition: StreamTurnTransition) => void;
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
  reasoningBlocks?: ReasoningBlock[];
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
  logprobs?: LLMLogprobs;
  reasoningBlocks?: ReasoningBlock[];
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
        onTurnTransition?: NormalizedStreamConfig<TSnapshot>["onTurnTransition"];
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
    onTurnTransition: option.onTurnTransition,
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
      logger: defaultDebugLogger,
    };
  }

  if (!option) {
    return {
      enabled: false,
      colors: true,
      verbose: false,
      logger: defaultDebugLogger,
    };
  }

  return {
    enabled: option.enabled ?? true,
    colors: option.colors ?? true,
    verbose: option.verbose ?? false,
    logger: option.logger ?? defaultDebugLogger,
  };
}

function defaultDebugLogger(line: string): void {
  const { log } = globalThis.console;
  log(line);
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

export { withToolTimeout, applyToolTimeout } from "./generate-tool-timeout";
export { callModel } from "./generate-model-call";
export {
  aggregateUsage,
  appendReasoningBlock,
  composeParseSource,
  mergeUsage,
  normalizeModelOutput,
  toStreamDataFingerprint,
} from "./generate-output";
export type { DebugRequestInput, DebugResponseInput } from "./generate-debug";
export { emitDebugRequest, emitDebugResponse } from "./generate-debug";
