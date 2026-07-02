import type {
  StructuredDebugOptions,
  StructuredPromptBuilder,
  StructuredTimeoutOptions,
} from "./structured";

export type HTTPHeaders = Record<string, string>;

export interface MCPToolSchema {
  type?: string;
  properties?: Record<string, unknown>;
  required?: string[];
  [key: string]: unknown;
}

export interface MCPToolDescriptor {
  name: string;
  description?: string;
  inputSchema?: MCPToolSchema;
}

export interface MCPListToolsResult {
  tools: MCPToolDescriptor[];
  nextCursor?: string;
}

export interface MCPCallToolParams {
  name: string;
  arguments?: Record<string, unknown>;
  _meta?: Record<string, unknown>;
}

export interface MCPToolClient {
  id: string;
  listTools(params?: { cursor?: string }): Promise<MCPListToolsResult>;
  callTool(params: MCPCallToolParams): Promise<unknown>;
  close?(): Promise<void>;
}

export interface LLMTextContent {
  type: "text";
  text: string;
}

export interface LLMImageContent {
  type: "image_url";
  image_url: { url: string };
}

export type LLMMessageContent = string | (LLMTextContent | LLMImageContent)[];

export interface LLMToolCallRef {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ReasoningBlock {
  turnIndex: number;
  text: string;
}

export interface StreamTurnTransition {
  turnIndex: number;
  kind: "reasoningComplete" | "toolCallsEmit" | "toolResultsReceived" | "streamEnd";
  reasoningText?: string;
  toolCalls?: LLMToolCall[];
}

export interface LLMMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: LLMMessageContent;
  [key: string]: unknown;
}

export type LLMReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "max";

export interface LLMRequest {
  prompt?: string;
  systemPrompt?: string;
  messages?: LLMMessage[];
  temperature?: number;
  reasoningEffort?: LLMReasoningEffort;
  maxTokens?: number;
  mcpClients?: MCPToolClient[];
  toolChoice?: LLMToolChoice;
  parallelToolCalls?: boolean;
  maxToolRounds?: number;
  onToolExecution?: (execution: LLMToolExecution) => void;
  transformToolOutput?: LLMToolOutputTransformer;
  transformToolArguments?: LLMToolArgumentsTransformer;
  transformToolCallParams?: LLMToolCallParamsTransformer;
  unknownToolError?: (toolName: string) => string;
  toolDebug?: boolean | LLMToolDebugOptions;
  onTurnTransition?: (transition: StreamTurnTransition) => void;
  body?: Record<string, unknown>;
  signal?: AbortSignal;
}

export interface LLMUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cost?: number;
}

/** One alternative token considered at a position, with its log probability. */
export interface LLMTopLogprob {
  token: string;
  logprob: number;
  bytes?: number[] | null;
}

/** The chosen token at a position, plus (optionally) the top alternatives. */
export interface LLMTokenLogprob extends LLMTopLogprob {
  top_logprobs?: LLMTopLogprob[];
}

/**
 * Token log probabilities, mirroring the OpenAI chat-completions `logprobs`
 * shape (`choices[].logprobs`). Populated only when the request opts in (via
 * `body.logprobs`); otherwise left undefined so callers see identical behavior.
 */
export interface LLMLogprobs {
  content?: LLMTokenLogprob[] | null;
  refusal?: LLMTokenLogprob[] | null;
}

export interface LLMResponse {
  text: string;
  reasoning?: string;
  reasoningBlocks?: ReasoningBlock[];
  raw?: unknown;
  usage?: LLMUsage;
  finishReason?: string;
  logprobs?: LLMLogprobs;
  toolCalls?: LLMToolCall[];
  toolExecutions?: LLMToolExecution[];
}

export interface LLMStreamChunk {
  textDelta: string;
  reasoningDelta?: string;
  turnIndex?: number;
  toolCalls?: LLMToolCall[];
  raw?: unknown;
  done?: boolean;
  usage?: LLMUsage;
  finishReason?: string;
  logprobs?: LLMLogprobs;
}

export interface LLMStreamCallbacks {
  onStart?: () => void;
  onToken?: (token: string) => void;
  onChunk?: (chunk: LLMStreamChunk) => void;
  onComplete?: (response: LLMResponse) => void;
}

export interface EmbeddingRequest {
  input: string | string[];
  model?: string;
  dimensions?: number;
  body?: Record<string, unknown>;
}

export interface EmbeddingResult {
  embeddings: number[][];
  model: string;
  usage?: LLMUsage;
  raw?: unknown;
}

export interface LLMAdapter {
  provider?: string;
  model?: string;
  complete(request: LLMRequest): Promise<LLMResponse>;
  stream?(request: LLMRequest, callbacks?: LLMStreamCallbacks): Promise<LLMResponse>;
  embed?(request: EmbeddingRequest): Promise<EmbeddingResult>;
}

export interface LLMToolCall {
  id: string;
  type: string;
  name?: string;
  arguments?: unknown;
  output?: unknown;
  error?: string;
}

export interface LLMToolExecution {
  callId: string;
  type: string;
  name?: string;
  clientId?: string;
  remoteName?: string;
  arguments?: unknown;
  output?: unknown;
  error?: string;
  round?: number;
  provider?: string;
  model?: string;
  handledLocally: boolean;
  startedAt: string;
  durationMs?: number;
}

export type LLMToolOutputTransformer = (
  output: unknown,
  execution: Omit<LLMToolExecution, "output" | "durationMs">,
) => unknown | Promise<unknown>;

export type LLMToolArgumentsTransformer = (
  args: Record<string, unknown>,
  context: { name: string; remoteName: string; clientId: string },
) => Record<string, unknown> | Promise<Record<string, unknown>>;

export type LLMToolCallParamsTransformer = (
  params: MCPCallToolParams,
  context: { name: string; remoteName: string; clientId: string },
) => MCPCallToolParams | Promise<MCPCallToolParams>;

export interface LLMToolDebugOptions {
  enabled?: boolean;
  logger?: (line: string) => void;
  includeRequest?: boolean;
  includeResult?: boolean;
  includeResultOnError?: boolean;
  pretty?: boolean;
}

export type LLMToolChoice =
  | "none"
  | "auto"
  | "required"
  | {
      type: "function";
      function: {
        name: string;
      };
    }
  | Record<string, unknown>;

export interface GenerateTraceEvent {
  stage: "llm.request" | "llm.response" | "llm.stream.delta" | "llm.stream.data" | "result";
  attempt: number;
  message: string;
  details?: unknown;
}

export interface GenerateStreamDelta {
  text: string;
  reasoning: string;
}

export interface GenerateStreamSnapshot {
  text: string;
  reasoning: string;
  reasoningBlocks?: ReasoningBlock[];
}

export interface GenerateStreamEvent {
  delta: GenerateStreamDelta;
  snapshot: GenerateStreamSnapshot;
  done: boolean;
  usage?: LLMUsage;
  finishReason?: string;
  turnIndex?: number;
  toolCalls?: LLMToolCall[];
}

export interface GenerateStreamOptions {
  enabled?: boolean;
  onData?: (event: GenerateStreamEvent) => void;
  onTurnTransition?: (transition: StreamTurnTransition) => void;
  to?: "stdout";
}

export type GenerateStreamInput = boolean | GenerateStreamOptions;

export interface GenerateCallOptions {
  outdent?: boolean;
  stream?: GenerateStreamInput;
  debug?: boolean | StructuredDebugOptions;
  observe?: (event: GenerateTraceEvent) => void;
  systemPrompt?: string;
  request?: Omit<LLMRequest, "prompt" | "systemPrompt" | "messages">;
  timeout?: StructuredTimeoutOptions;
}

export interface GenerateOptions extends GenerateCallOptions {
  prompt: StructuredPromptBuilder;
}

export interface GenerateAttempt {
  attempt: number;
  via: "complete" | "stream";
  text: string;
  reasoning: string;
  usage?: LLMUsage;
  finishReason?: string;
  logprobs?: LLMLogprobs;
  reasoningBlocks?: ReasoningBlock[];
}

export interface GenerateResult {
  text: string;
  reasoning: string;
  attempts: GenerateAttempt[];
  usage?: LLMUsage;
  finishReason?: string;
  logprobs?: LLMLogprobs;
  reasoningBlocks?: ReasoningBlock[];
}
