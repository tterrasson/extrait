import type { z } from "zod";
import type { ParseLLMOutputOptions, ParseLLMOutputResult, StructuredMode } from "./parse";
import type {
  LLMLogprobs,
  LLMMessage,
  LLMRequest,
  LLMToolCall,
  LLMUsage,
  ReasoningBlock,
  StreamTurnTransition,
} from "./llm";

export interface StructuredTraceEvent {
  stage:
    | "llm.request"
    | "llm.response"
    | "llm.stream.delta"
    | "llm.stream.data"
    | "parse"
    | "self-heal"
    | "result";
  attempt: number;
  selfHeal: boolean;
  message: string;
  details?: unknown;
}

export interface StructuredPromptContext {
  mode: StructuredMode;
}

export interface StructuredPromptPayload {
  prompt?: string;
  systemPrompt?: string;
  messages?: LLMMessage[];
}

export interface StructuredPromptResolver {
  resolvePrompt(context: StructuredPromptContext): StructuredPromptPayload;
}

export type StructuredPromptValue = string | StructuredPromptPayload | StructuredPromptResolver;

export type StructuredPromptBuilder =
  | StructuredPromptValue
  | ((context: StructuredPromptContext) => StructuredPromptValue);

export interface StructuredDebugOptions {
  enabled?: boolean;
  colors?: boolean;
  verbose?: boolean;
  logger?: (line: string) => void;
}

export interface StructuredSelfHealOptions {
  enabled?: boolean;
  maxAttempts?: number;
  stopOnNoProgress?: boolean;
  maxContextChars?: number;
}

export type StructuredSelfHealInput = boolean | number | StructuredSelfHealOptions;

export interface StructuredTimeoutOptions {
  /** Timeout in ms for each LLM HTTP request. Creates an AbortSignal.timeout internally if no signal is already provided. */
  request?: number;
  /** Timeout in ms for each MCP tool call. */
  tool?: number;
}

export type StructuredStreamData<T> =
  T extends Array<infer TItem>
    ? Array<StructuredStreamData<TItem>>
    : T extends object
      ? { [K in keyof T]?: StructuredStreamData<T[K]> | null }
      : T | null;

export interface StructuredStreamDelta {
  text: string;
  reasoning: string;
}

export interface StructuredStreamSnapshot<T = unknown> {
  text: string;
  reasoning: string;
  reasoningBlocks?: ReasoningBlock[];
  data: StructuredStreamData<T> | null;
}

export interface StructuredStreamEvent<T = unknown> {
  delta: StructuredStreamDelta;
  snapshot: StructuredStreamSnapshot<T>;
  done: boolean;
  usage?: LLMUsage;
  finishReason?: string;
  turnIndex?: number;
  toolCalls?: LLMToolCall[];
}

export interface StructuredStreamOptions<T = unknown> {
  enabled?: boolean;
  onData?: (event: StructuredStreamEvent<T>) => void;
  onTurnTransition?: (transition: StreamTurnTransition) => void;
  to?: "stdout";
  /**
   * Minimum delay (ms) between two recomputations of `snapshot.data` while
   * streaming. Between recomputations, events reuse the last parsed value, so
   * `snapshot.data` can lag behind `snapshot.text` by up to this much. The
   * final `done` event always reparses. `0` reparses on every event.
   *
   * Unset (default) is adaptive: exact while the accumulated text is small
   * (≤ 2 ko), then coalesced to 25 ms so long generations stay linear.
   */
  dataInterval?: number;
}

export type StructuredStreamInput<T = unknown> = boolean | StructuredStreamOptions<T>;

export interface StructuredCallOptions<TSchema extends z.ZodTypeAny> {
  mode?: StructuredMode;
  outdent?: boolean;
  parse?: ParseLLMOutputOptions;
  selfHeal?: StructuredSelfHealInput;
  stream?: StructuredStreamInput<z.infer<TSchema>>;
  debug?: boolean | StructuredDebugOptions;
  observe?: (event: StructuredTraceEvent) => void;
  systemPrompt?: string;
  request?: Omit<LLMRequest, "prompt" | "systemPrompt" | "messages">;
  schemaInstruction?: string;
  timeout?: StructuredTimeoutOptions;
}

export interface StructuredOptions<TSchema extends z.ZodTypeAny> extends StructuredCallOptions<TSchema> {
  schema: TSchema;
  prompt: StructuredPromptBuilder;
}

export interface StructuredAttempt<T> {
  attempt: number;
  selfHeal: boolean;
  via: "complete" | "stream";
  text: string;
  reasoning: string;
  json: unknown | null;
  candidates: string[];
  repairLog: string[];
  zodIssues: z.core.$ZodIssue[];
  success: boolean;
  usage?: LLMUsage;
  finishReason?: string;
  logprobs?: LLMLogprobs;
  reasoningBlocks?: ReasoningBlock[];
  parsed: ParseLLMOutputResult<T>;
}

export interface StructuredResult<T> {
  data: T;
  text: string;
  reasoning: string;
  json: unknown | null;
  attempts: StructuredAttempt<T>[];
  usage?: LLMUsage;
  finishReason?: string;
  logprobs?: LLMLogprobs;
  reasoningBlocks?: ReasoningBlock[];
}

export interface StructuredError {
  name: "StructuredParseError";
  text: string;
  reasoning: string;
  candidates: string[];
  zodIssues?: z.core.$ZodIssue[];
  repairLog?: string[];
  attempt: number;
}
