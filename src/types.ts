import type { z } from "zod";

export type StructuredMode = "loose" | "strict";
export type HTTPHeaders = Record<string, string>;

export interface ExtractionCandidate {
  id: string;
  source: "fenced" | "scan" | "raw";
  content: string;
  language?: string | null;
  parseHint?: ExtractionParseHint;
  start: number;
  end: number;
  score: number;
}

export interface ExtractionParseHint {
  success: boolean;
  parsed: unknown | null;
  repaired: string | null;
  usedRepair: boolean;
  stage: "parse" | "repair";
  error: string;
}

export interface ExtractionHeuristicsOptions {
  firstPassMin: number;
  firstPassCap: number;
  firstPassMultiplier: number;
  secondPassMin: number;
  secondPassCap: number;
  secondPassMultiplier: number;
  hintMaxLength: number;
}

export interface ExtractJsonCandidatesOptions {
  maxCandidates?: number;
  acceptArrays?: boolean;
  allowRepairHints?: boolean;
  heuristics?: Partial<ExtractionHeuristicsOptions>;
}

export interface ParseTraceEvent {
  stage: "extract" | "repair" | "parse" | "validate" | "result";
  level: "info" | "error";
  message: string;
  candidateId?: string;
  details?: unknown;
}

export interface ParseLLMOutputOptions {
  repair?: boolean;
  maxCandidates?: number;
  acceptArrays?: boolean;
  extraction?: Partial<ExtractionHeuristicsOptions>;
  onTrace?: (event: ParseTraceEvent) => void;
}

export interface PipelineError {
  stage: "extract" | "repair" | "parse" | "validate" | "llm" | "self-heal";
  message: string;
  candidateId?: string;
  details?: unknown;
}

export interface CandidateDiagnostics {
  candidateId: string;
  source: ExtractionCandidate["source"];
  usedRepair: boolean;
  parseSuccess: boolean;
  validationSuccess: boolean;
  selected: boolean;
  stage: "repair" | "parse" | "validate" | "success";
  message?: string;
  zodIssues?: z.ZodIssue[];
}

export interface ThinkBlock {
  id: string;
  content: string;
  raw: string;
  start: number;
  end: number;
}

export interface ThinkDiagnostics {
  unterminatedCount: number;
  nestedCount: number;
  hiddenChars: number;
}

export interface ParseLLMOutputResult<T> {
  success: boolean;
  data: T | null;
  raw: string;
  sanitizedRaw: string;
  thinkBlocks: ThinkBlock[];
  thinkDiagnostics: ThinkDiagnostics;
  parsed: unknown | null;
  candidate: ExtractionCandidate | null;
  repaired: string | null;
  candidates: ExtractionCandidate[];
  diagnostics: CandidateDiagnostics[];
  errors: PipelineError[];
  zodIssues: z.ZodIssue[];
}

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
}

export interface MCPToolClient {
  id: string;
  listTools(params?: { cursor?: string }): Promise<MCPListToolsResult>;
  callTool(params: MCPCallToolParams): Promise<unknown>;
  close?(): Promise<void>;
}

export interface LLMRequest {
  prompt: string;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  mcpClients?: MCPToolClient[];
  toolChoice?: LLMToolChoice;
  parallelToolCalls?: boolean;
  maxToolRounds?: number;
  onToolExecution?: (execution: LLMToolExecution) => void;
  toolDebug?: boolean | LLMToolDebugOptions;
  body?: Record<string, unknown>;
}

export interface LLMUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cost?: number;
}

export interface LLMResponse {
  text: string;
  raw?: unknown;
  usage?: LLMUsage;
  finishReason?: string;
  toolCalls?: LLMToolCall[];
  toolExecutions?: LLMToolExecution[];
}

export interface LLMStreamChunk {
  textDelta: string;
  raw?: unknown;
  done?: boolean;
  usage?: LLMUsage;
  finishReason?: string;
}

export interface LLMStreamCallbacks {
  onStart?: () => void;
  onToken?: (token: string) => void;
  onChunk?: (chunk: LLMStreamChunk) => void;
  onComplete?: (response: LLMResponse) => void;
}

export interface LLMAdapter {
  provider?: string;
  model?: string;
  complete(request: LLMRequest): Promise<LLMResponse>;
  stream?(request: LLMRequest, callbacks?: LLMStreamCallbacks): Promise<LLMResponse>;
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
  prompt: string;
  systemPrompt?: string;
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
  logger?: (line: string) => void;
}

export interface StructuredSelfHealOptions {
  enabled?: boolean;
  maxAttempts?: number;
  stopOnNoProgress?: boolean;
  maxContextChars?: number;
}

export type StructuredSelfHealInput = boolean | number | StructuredSelfHealOptions;

export type StructuredStreamData<T> =
  T extends Array<infer TItem>
    ? Array<StructuredStreamData<TItem>>
    : T extends object
      ? { [K in keyof T]?: StructuredStreamData<T[K]> | null }
      : T | null;

export interface StructuredStreamEvent<T = unknown> {
  data: StructuredStreamData<T> | null;
  raw: string;
  done: boolean;
  usage?: LLMUsage;
  finishReason?: string;
}

export interface StructuredStreamOptions<T = unknown> {
  enabled?: boolean;
  onData?: (event: StructuredStreamEvent<T>) => void;
  to?: "stdout";
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
  request?: Omit<LLMRequest, "prompt" | "systemPrompt">;
  schemaInstruction?: string;
}

export interface StructuredOptions<TSchema extends z.ZodTypeAny> extends StructuredCallOptions<TSchema> {
  schema: TSchema;
  prompt: StructuredPromptBuilder;
}

export interface StructuredAttempt<T> {
  attempt: number;
  selfHeal: boolean;
  via: "complete" | "stream";
  raw: string;
  thinkBlocks: ThinkBlock[];
  json: unknown | null;
  candidates: string[];
  repairLog: string[];
  zodIssues: z.ZodIssue[];
  success: boolean;
  usage?: LLMUsage;
  finishReason?: string;
  parsed: ParseLLMOutputResult<T>;
}

export interface StructuredResult<T> {
  data: T;
  raw: string;
  thinkBlocks: ThinkBlock[];
  json: unknown | null;
  attempts: StructuredAttempt<T>[];
  usage?: LLMUsage;
  finishReason?: string;
}

export interface StructuredError {
  name: "StructuredParseError";
  raw: string;
  thinkBlocks: ThinkBlock[];
  candidates: string[];
  zodIssues?: z.ZodIssue[];
  repairLog?: string[];
  attempt: number;
}

export interface MarkdownCodeBlock {
  language: string | null;
  code: string;
  start: number;
  end: number;
}

export interface MarkdownCodeOptions {
  language?: string;
  firstOnly?: boolean;
}
