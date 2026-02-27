import { jsonrepair } from "jsonrepair";
import type { z } from "zod";
import { resolveSchemaInstruction, formatPrompt, withFormat } from "./format";
import { createOutdent } from "./outdent";
import { formatZodIssues, parseLLMOutput } from "./parse";
import { sanitizeThink } from "./think";
import { color, dim, title } from "./utils/debug-colors";
import type {
  LLMAdapter,
  LLMRequest,
  LLMUsage,
  ParseLLMOutputOptions,
  ParseTraceEvent,
  StructuredAttempt,
  StructuredCallOptions,
  StructuredDebugOptions,
  StructuredError,
  StructuredMode,
  StructuredOptions,
  StructuredPromptBuilder,
  StructuredPromptPayload,
  StructuredPromptResolver,
  StructuredPromptValue,
  StructuredResult,
  StructuredTraceEvent,
} from "./types";

export class StructuredParseError extends Error implements StructuredError {
  override readonly name = "StructuredParseError" as const;
  readonly raw: string;
  readonly thinkBlocks: StructuredError["thinkBlocks"];
  readonly candidates: string[];
  readonly zodIssues?: z.ZodIssue[];
  readonly repairLog?: string[];
  readonly attempt: number;

  constructor(input: {
    message?: string;
    raw: string;
    thinkBlocks: StructuredError["thinkBlocks"];
    candidates: string[];
    zodIssues?: z.ZodIssue[];
    repairLog?: string[];
    attempt: number;
  }) {
    super(input.message ?? `Structured parsing failed after ${input.attempt} attempt(s).`);
    this.raw = input.raw;
    this.thinkBlocks = input.thinkBlocks;
    this.candidates = input.candidates;
    this.zodIssues = input.zodIssues;
    this.repairLog = input.repairLog;
    this.attempt = input.attempt;
  }
}

export interface BuildDefaultStructuredPromptOptions {
  objectInstruction?: string;
  styleInstruction?: string;
}

export interface SelfHealPromptTextOptions {
  fixInstruction?: string;
  returnInstruction?: string;
  noIssuesMessage?: string;
  validationErrorsLabel?: string;
  rawOutputLabel?: string;
  contextLabel?: string;
}

type ParseDefaults = Pick<ParseLLMOutputOptions, "repair" | "maxCandidates" | "acceptArrays">;

export const DEFAULT_STRUCTURED_OBJECT_INSTRUCTION = "Return exactly one strict JSON object.";
export const DEFAULT_STRUCTURED_STYLE_INSTRUCTION = "No prose. No markdown.";

interface ResolvedSelfHealPromptText {
  fixInstruction: string;
  returnInstruction: string;
  noIssuesMessage: string;
  validationErrorsLabel: string;
  rawOutputLabel: string;
  contextLabel: string;
}

const DEFAULT_SELF_HEAL_PROMPT_TEXT: ResolvedSelfHealPromptText = {
  fixInstruction: "Fix the following output so it validates against the schema.",
  returnInstruction: "Return only the corrected JSON object.",
  noIssuesMessage: "No detailed validation issues",
  validationErrorsLabel: "Validation errors:",
  rawOutputLabel: "Raw output to fix:",
  contextLabel: "Self-heal context JSON:",
};

export const DEFAULT_SELF_HEAL_FIX_INSTRUCTION = DEFAULT_SELF_HEAL_PROMPT_TEXT.fixInstruction;
export const DEFAULT_SELF_HEAL_RETURN_INSTRUCTION = DEFAULT_SELF_HEAL_PROMPT_TEXT.returnInstruction;
export const DEFAULT_SELF_HEAL_NO_ISSUES_MESSAGE = DEFAULT_SELF_HEAL_PROMPT_TEXT.noIssuesMessage;
export const DEFAULT_SELF_HEAL_VALIDATION_LABEL =
  DEFAULT_SELF_HEAL_PROMPT_TEXT.validationErrorsLabel;
export const DEFAULT_SELF_HEAL_RAW_OUTPUT_LABEL = DEFAULT_SELF_HEAL_PROMPT_TEXT.rawOutputLabel;
export const DEFAULT_SELF_HEAL_CONTEXT_LABEL = DEFAULT_SELF_HEAL_PROMPT_TEXT.contextLabel;
export const DEFAULT_SELF_HEAL_PROTOCOL = "extrait.self-heal.v2";
export const DEFAULT_SELF_HEAL_MAX_CONTEXT_CHARS = 12_000;
export const DEFAULT_SELF_HEAL_STOP_ON_NO_PROGRESS = true;
const DEFAULT_SELF_HEAL_MAX_ERRORS = 8;
const RE_SIMPLE_IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const RE_ESCAPE_QUOTE = /"/g;
const RE_WHITESPACE = /\s+/g;
const DEFAULT_SELF_HEAL_MAX_DIAGNOSTICS = 8;
const structuredOutdent = createOutdent({
  trimLeadingNewline: true,
  trimTrailingNewline: true,
  newline: "\n",
});
export const DEFAULT_STRICT_PARSE_OPTIONS: ParseDefaults = {
  repair: false,
  maxCandidates: 3,
  acceptArrays: true,
};
export const DEFAULT_LOOSE_PARSE_OPTIONS: ParseDefaults = {
  repair: true,
  maxCandidates: 5,
  acceptArrays: true,
};
export const DEFAULT_SELF_HEAL_BY_MODE = {
  loose: { enabled: true, maxAttempts: 1 },
  strict: { enabled: false, maxAttempts: 0 },
} as const;

export function buildDefaultStructuredPrompt(
  task: string,
  options: BuildDefaultStructuredPromptOptions = {},
): string {
  const objectInstruction = resolvePromptLine(
    options.objectInstruction,
    DEFAULT_STRUCTURED_OBJECT_INSTRUCTION,
  );
  const styleInstruction = resolvePromptLine(
    options.styleInstruction,
    DEFAULT_STRUCTURED_STYLE_INSTRUCTION,
  );
  return [task.trim(), "", objectInstruction, styleInstruction].join("\n");
}

interface SelfHealPromptInput {
  rawOutput: string;
  issues: z.ZodIssue[];
  schema: z.ZodTypeAny;
  schemaInstruction?: string;
  selectedOutput?: string;
  selectedInput?: "candidate" | "sanitized" | "raw";
  sanitizedOutput?: string;
  parserErrors?: Array<{ stage: string; message: string; candidateId?: string }>;
  diagnostics?: Array<{
    candidateId: string;
    stage: string;
    usedRepair: boolean;
    message?: string;
  }>;
  repairLog?: string[];
  attempt?: number;
  previousAttempt?: number;
  maxContextChars?: number;
  text?: SelfHealPromptTextOptions;
}

export function buildSelfHealPrompt(input: SelfHealPromptInput): string {
  const text = resolveSelfHealPromptText(input.text);
  const issueText = input.issues.length > 0 ? formatZodIssues(input.issues) : text.noIssuesMessage;
  const outputFormat = withFormat(input.schema, {
    schemaInstruction: input.schemaInstruction,
  });
  const maxContextChars = normalizePositiveInt(
    input.maxContextChars,
    DEFAULT_SELF_HEAL_MAX_CONTEXT_CHARS,
  );
  const selectedOutput = input.selectedOutput ?? input.rawOutput;
  const contextPayload = {
    protocol: DEFAULT_SELF_HEAL_PROTOCOL,
    attempt: input.attempt,
    previousAttempt: input.previousAttempt,
    selectedInput: input.selectedInput ?? "raw",
    issueSummary: issueText,
    validationIssues: input.issues.map((issue) => ({
      path: formatIssuePath(issue.path),
      code: issue.code,
      message: issue.message,
      expected: "expected" in issue ? issue.expected : undefined,
      received: "received" in issue ? issue.received : undefined,
    })),
    parserErrors: (input.parserErrors ?? []).map((error) => ({
      stage: error.stage,
      message: error.message,
      candidateId: error.candidateId,
    })),
    diagnostics: (input.diagnostics ?? []).map((diagnostic) => ({
      candidateId: diagnostic.candidateId,
      stage: diagnostic.stage,
      usedRepair: diagnostic.usedRepair,
      message: diagnostic.message,
    })),
    repairLog: input.repairLog ?? [],
    selectedOutput: truncateForPrompt(selectedOutput, maxContextChars),
    sanitizedOutput: input.sanitizedOutput
      ? truncateForPrompt(input.sanitizedOutput, maxContextChars)
      : undefined,
    rawOutput: truncateForPrompt(input.rawOutput, maxContextChars),
  };

  return [
    text.fixInstruction,
    text.returnInstruction,
    "",
    outputFormat,
    "",
    text.validationErrorsLabel,
    issueText,
    "",
    text.rawOutputLabel,
    truncateForPrompt(input.rawOutput, maxContextChars),
    "",
    text.contextLabel,
    JSON.stringify(contextPayload, null, 2),
  ].join("\n");
}

export async function structured<TSchema extends z.ZodTypeAny>(
  adapter: LLMAdapter,
  schema: TSchema,
  prompt: StructuredPromptBuilder,
  options?: StructuredCallOptions<TSchema>,
): Promise<StructuredResult<z.infer<TSchema>>>;
export async function structured<TSchema extends z.ZodTypeAny>(
  adapter: LLMAdapter,
  options: StructuredOptions<TSchema>,
): Promise<StructuredResult<z.infer<TSchema>>>;
export async function structured<TSchema extends z.ZodTypeAny>(
  adapter: LLMAdapter,
  schemaOrOptions: TSchema | StructuredOptions<TSchema>,
  promptInput?: StructuredPromptBuilder,
  callOptions?: StructuredCallOptions<TSchema>,
): Promise<StructuredResult<z.infer<TSchema>>> {
  const normalized = normalizeStructuredInput(schemaOrOptions, promptInput, callOptions);
  const mode = normalized.mode ?? "loose";
  const selfHealConfig = normalizeSelfHealConfig(normalized.selfHeal, mode);
  const parseOptions = mergeParseOptions(mode, normalized.parse);
  const streamConfig = normalizeStreamConfig(normalized.stream);
  const debugConfig = normalizeDebugConfig(normalized.debug);
  const attempts: StructuredAttempt<z.infer<TSchema>>[] = [];
  const useOutdent = normalized.outdent ?? true;

  const resolvedPrompt = applyPromptOutdent(resolvePrompt(normalized.prompt, { mode }), useOutdent);
  const resolvedSystemPrompt = applyOutdentToOptionalPrompt(normalized.systemPrompt, useOutdent);
  const prompt = shouldInjectFormat(resolvedPrompt.prompt, normalized.schemaInstruction)
    ? formatPrompt(normalized.schema, resolvedPrompt.prompt, {
        schemaInstruction: normalized.schemaInstruction,
      })
    : resolvedPrompt.prompt.trim();
  const systemPrompt = mergeSystemPrompts(resolvedPrompt.systemPrompt, resolvedSystemPrompt);

  const first = await executeAttempt(adapter, {
    prompt,
    schema: normalized.schema,
    parseOptions,
    stream: streamConfig,
    request: normalized.request,
    systemPrompt,
    observe: normalized.observe,
    debug: debugConfig,
    attemptNumber: 1,
    selfHeal: false,
    selfHealEnabled: selfHealConfig.enabled,
  });
  attempts.push(first.trace);

  if (first.trace.success) {
    return buildSuccessResult(first.trace.parsed.data as z.infer<TSchema>, attempts);
  }

  if (!selfHealConfig.enabled) {
    throw toStructuredError(attempts.at(-1));
  }

  for (let index = 0; index < selfHealConfig.maxAttempts; index += 1) {
    const previous = attempts.at(-1);
    if (!previous) {
      break;
    }

    const attemptNumber = index + 2;

    emitObserve(normalized.observe, {
      stage: "self-heal",
      attempt: attemptNumber,
      selfHeal: true,
      message: "Starting self-heal attempt.",
      details: {
        previousIssues: previous.zodIssues.length,
      },
    });

    const selfHealSource = resolveSelfHealSource(previous);
    const repairPrompt = buildSelfHealPrompt({
      rawOutput: previous.raw,
      issues: previous.zodIssues,
      schema: normalized.schema,
      schemaInstruction: normalized.schemaInstruction,
      selectedOutput: selfHealSource.text,
      selectedInput: selfHealSource.kind,
      sanitizedOutput: previous.parsed.sanitizedRaw,
      parserErrors: previous.parsed.errors
        .slice(0, DEFAULT_SELF_HEAL_MAX_ERRORS)
        .map((error) => ({
          stage: error.stage,
          message: error.message,
          candidateId: error.candidateId,
        })),
      diagnostics: previous.parsed.diagnostics
        .slice(0, DEFAULT_SELF_HEAL_MAX_DIAGNOSTICS)
        .map((diagnostic) => ({
          candidateId: diagnostic.candidateId,
          stage: diagnostic.stage,
          usedRepair: diagnostic.usedRepair,
          message: diagnostic.message,
        })),
      repairLog: previous.repairLog,
      attempt: attemptNumber,
      previousAttempt: previous.attempt,
      maxContextChars: selfHealConfig.maxContextChars,
    });

    const healed = await executeAttempt(adapter, {
      prompt: repairPrompt,
      schema: normalized.schema,
      parseOptions,
      stream: streamConfig,
      request: normalized.request,
      systemPrompt,
      observe: normalized.observe,
      debug: debugConfig,
      attemptNumber,
      selfHeal: true,
      selfHealEnabled: selfHealConfig.enabled,
    });

    attempts.push(healed.trace);

    if (healed.trace.success) {
      return buildSuccessResult(healed.trace.parsed.data as z.infer<TSchema>, attempts);
    }

    if (selfHealConfig.stopOnNoProgress && isSelfHealStalled(previous, healed.trace)) {
      emitObserve(normalized.observe, {
        stage: "self-heal",
        attempt: attemptNumber,
        selfHeal: true,
        message: "Stopping self-heal: no progress detected.",
        details: {
          previousIssues: previous.zodIssues.length,
          currentIssues: healed.trace.zodIssues.length,
        },
      });
      break;
    }
  }

  throw toStructuredError(attempts.at(-1));
}

function normalizeStructuredInput<TSchema extends z.ZodTypeAny>(
  schemaOrOptions: TSchema | StructuredOptions<TSchema>,
  promptInput?: StructuredPromptBuilder,
  callOptions?: StructuredCallOptions<TSchema>,
): StructuredOptions<TSchema> {
  if (isStructuredOptions(schemaOrOptions)) {
    return schemaOrOptions;
  }

  if (!promptInput) {
    throw new Error("Missing prompt in structured(adapter, schema, prompt, options?) call.");
  }

  return {
    ...(callOptions ?? {}),
    schema: schemaOrOptions,
    prompt: promptInput,
  };
}

function isStructuredOptions<TSchema extends z.ZodTypeAny>(
  value: TSchema | StructuredOptions<TSchema>,
): value is StructuredOptions<TSchema> {
  return typeof value === "object" && value !== null && "schema" in value && "prompt" in value;
}

function resolvePrompt(
  prompt: StructuredPromptBuilder,
  context: { mode: StructuredMode },
): StructuredPromptPayload {
  const resolved = typeof prompt === "function" ? prompt(context) : prompt;
  return normalizePromptValue(resolved, context);
}

function normalizePromptValue(
  value: StructuredPromptValue,
  context: { mode: StructuredMode },
): StructuredPromptPayload {
  if (typeof value === "string") {
    return {
      prompt: value,
    };
  }

  if (isPromptResolver(value)) {
    return normalizePromptPayload(value.resolvePrompt(context));
  }

  return normalizePromptPayload(value);
}

function isPromptResolver(value: StructuredPromptValue): value is StructuredPromptResolver {
  return (
    typeof value === "object" &&
    value !== null &&
    "resolvePrompt" in value &&
    typeof value.resolvePrompt === "function"
  );
}

function normalizePromptPayload(value: StructuredPromptPayload): StructuredPromptPayload {
  if (typeof value.prompt !== "string") {
    throw new Error("Structured prompt payload must include a string prompt.");
  }

  return {
    prompt: value.prompt,
    systemPrompt: typeof value.systemPrompt === "string" ? value.systemPrompt : undefined,
  };
}

function applyPromptOutdent(payload: StructuredPromptPayload, enabled: boolean): StructuredPromptPayload {
  if (!enabled) {
    return payload;
  }

  return {
    prompt: structuredOutdent.string(payload.prompt),
    systemPrompt: applyOutdentToOptionalPrompt(payload.systemPrompt, enabled),
  };
}

function applyOutdentToOptionalPrompt(value: string | undefined, enabled: boolean): string | undefined {
  if (!enabled || typeof value !== "string") {
    return value;
  }

  return structuredOutdent.string(value);
}

function mergeSystemPrompts(primary?: string, secondary?: string): string | undefined {
  const prompts = [primary, secondary]
    .map((value) => value?.trim())
    .filter((value): value is string => Boolean(value));

  if (prompts.length === 0) {
    return undefined;
  }

  return prompts.join("\n\n");
}

function shouldInjectFormat(prompt: string, schemaInstruction: string | undefined): boolean {
  const instruction = resolveSchemaInstruction(schemaInstruction);
  return !prompt.trimStart().startsWith(instruction);
}

function mergeParseOptions(
  mode: StructuredMode,
  options: ParseLLMOutputOptions | undefined,
): ParseLLMOutputOptions {
  const defaults: ParseLLMOutputOptions =
    mode === "strict" ? DEFAULT_STRICT_PARSE_OPTIONS : DEFAULT_LOOSE_PARSE_OPTIONS;

  return {
    ...defaults,
    ...options,
  };
}

function normalizeSelfHealConfig(
  option: StructuredCallOptions<z.ZodTypeAny>["selfHeal"],
  mode: StructuredMode,
): {
  enabled: boolean;
  maxAttempts: number;
  stopOnNoProgress: boolean;
  maxContextChars: number;
} {
  if (typeof option === "number") {
    const maxAttempts = Math.max(0, Math.floor(option));
    return {
      enabled: maxAttempts > 0,
      maxAttempts,
      stopOnNoProgress: DEFAULT_SELF_HEAL_STOP_ON_NO_PROGRESS,
      maxContextChars: DEFAULT_SELF_HEAL_MAX_CONTEXT_CHARS,
    };
  }

  if (typeof option === "boolean") {
    return {
      enabled: option,
      maxAttempts: option ? 1 : 0,
      stopOnNoProgress: DEFAULT_SELF_HEAL_STOP_ON_NO_PROGRESS,
      maxContextChars: DEFAULT_SELF_HEAL_MAX_CONTEXT_CHARS,
    };
  }

  if (!option) {
    const modeDefaults =
      mode === "loose" ? DEFAULT_SELF_HEAL_BY_MODE.loose : DEFAULT_SELF_HEAL_BY_MODE.strict;
    return {
      enabled: modeDefaults.enabled,
      maxAttempts: modeDefaults.maxAttempts,
      stopOnNoProgress: DEFAULT_SELF_HEAL_STOP_ON_NO_PROGRESS,
      maxContextChars: DEFAULT_SELF_HEAL_MAX_CONTEXT_CHARS,
    };
  }

  const enabled = option.enabled ?? true;
  const stopOnNoProgress = option.stopOnNoProgress ?? DEFAULT_SELF_HEAL_STOP_ON_NO_PROGRESS;
  const maxContextChars = normalizePositiveInt(
    option.maxContextChars,
    DEFAULT_SELF_HEAL_MAX_CONTEXT_CHARS,
  );
  return {
    enabled,
    maxAttempts: enabled ? Math.max(1, option.maxAttempts ?? 1) : 0,
    stopOnNoProgress,
    maxContextChars,
  };
}

function resolvePromptLine(value: string | undefined, fallback: string): string {
  const trimmed = value?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : fallback;
}

function resolveSelfHealPromptText(
  text?: SelfHealPromptTextOptions,
): ResolvedSelfHealPromptText {
  return {
    fixInstruction: resolvePromptLine(
      text?.fixInstruction,
      DEFAULT_SELF_HEAL_PROMPT_TEXT.fixInstruction,
    ),
    returnInstruction: resolvePromptLine(
      text?.returnInstruction,
      DEFAULT_SELF_HEAL_PROMPT_TEXT.returnInstruction,
    ),
    noIssuesMessage: resolvePromptLine(
      text?.noIssuesMessage,
      DEFAULT_SELF_HEAL_PROMPT_TEXT.noIssuesMessage,
    ),
    validationErrorsLabel: resolvePromptLine(
      text?.validationErrorsLabel,
      DEFAULT_SELF_HEAL_PROMPT_TEXT.validationErrorsLabel,
    ),
    rawOutputLabel: resolvePromptLine(
      text?.rawOutputLabel,
      DEFAULT_SELF_HEAL_PROMPT_TEXT.rawOutputLabel,
    ),
    contextLabel: resolvePromptLine(text?.contextLabel, DEFAULT_SELF_HEAL_PROMPT_TEXT.contextLabel),
  };
}

function normalizePositiveInt(value: number | undefined, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }

  const normalized = Math.floor(value);
  return normalized > 0 ? normalized : fallback;
}

function truncateForPrompt(value: string, maxChars: number): string {
  if (value.length <= maxChars) {
    return value;
  }

  const marker = `\n...[truncated ${value.length - maxChars} chars]`;
  const head = Math.max(1, maxChars - marker.length);
  return `${value.slice(0, head)}${marker}`;
}

function formatIssuePath(path: Array<string | number>): string {
  if (path.length === 0) {
    return "$";
  }

  let out = "$";
  for (const segment of path) {
    if (typeof segment === "number") {
      out += `[${segment}]`;
      continue;
    }

    if (RE_SIMPLE_IDENTIFIER.test(segment)) {
      out += `.${segment}`;
      continue;
    }

    out += `["${segment.replace(RE_ESCAPE_QUOTE, '\\"')}"]`;
  }

  return out;
}

function resolveSelfHealSource<T>(
  attempt: StructuredAttempt<T>,
): { kind: "candidate" | "sanitized" | "raw"; text: string } {
  const candidate = attempt.parsed.candidate?.content?.trim();
  if (candidate) {
    return {
      kind: "candidate",
      text: candidate,
    };
  }

  const sanitized = attempt.parsed.sanitizedRaw?.trim();
  if (sanitized) {
    return {
      kind: "sanitized",
      text: sanitized,
    };
  }

  return {
    kind: "raw",
    text: attempt.raw,
  };
}

function isSelfHealStalled<T>(
  previous: StructuredAttempt<T>,
  current: StructuredAttempt<T>,
): boolean {
  if (current.success) {
    return false;
  }

  if (current.zodIssues.length < previous.zodIssues.length) {
    return false;
  }

  if (current.parsed.errors.length < previous.parsed.errors.length) {
    return false;
  }

  return buildSelfHealFailureFingerprint(previous) === buildSelfHealFailureFingerprint(current);
}

function buildSelfHealFailureFingerprint<T>(attempt: StructuredAttempt<T>): string {
  const issues = attempt.zodIssues
    .map(
      (issue) =>
        `${formatIssuePath(issue.path)}:${issue.code}:${normalizeWhitespace(issue.message)}`,
    )
    .sort()
    .join("|");
  const errors = attempt.parsed.errors
    .map(
      (error) =>
        `${error.stage}:${error.candidateId ?? "-"}:${normalizeWhitespace(error.message)}`,
    )
    .sort()
    .join("|");
  const source = normalizeWhitespace(resolveSelfHealSource(attempt).text).slice(0, 512);
  return [issues, errors, source].join("::");
}

function normalizeWhitespace(value: string): string {
  return value.replace(RE_WHITESPACE, " ").trim();
}

interface NormalizedStreamConfig {
  enabled: boolean;
  onData?: (event: {
    data: unknown | null;
    raw: string;
    done: boolean;
    usage?: LLMUsage;
    finishReason?: string;
  }) => void;
  to?: "stdout";
}

function normalizeStreamConfig(
  option: StructuredCallOptions<z.ZodTypeAny>["stream"],
): NormalizedStreamConfig {
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
    onData: option.onData as NormalizedStreamConfig["onData"],
    to: option.to,
  };
}

interface NormalizedDebugConfig {
  enabled: boolean;
  colors: boolean;
  logger: (line: string) => void;
}

function normalizeDebugConfig(
  option: StructuredDebugOptions | boolean | undefined,
): NormalizedDebugConfig {
  if (typeof option === "boolean") {
    return {
      enabled: option,
      colors: true,
      logger: (line: string) => console.log(line),
    };
  }

  if (!option) {
    return {
      enabled: false,
      colors: true,
      logger: (line: string) => console.log(line),
    };
  }

  return {
    enabled: option.enabled ?? true,
    colors: option.colors ?? true,
    logger: option.logger ?? ((line: string) => console.log(line)),
  };
}

interface ExecuteAttemptInput<TSchema extends z.ZodTypeAny> {
  prompt: string;
  schema: TSchema;
  parseOptions: ParseLLMOutputOptions;
  stream: NormalizedStreamConfig;
  request?: StructuredCallOptions<TSchema>["request"];
  systemPrompt?: string;
  observe?: StructuredCallOptions<TSchema>["observe"];
  debug: NormalizedDebugConfig;
  attemptNumber: number;
  selfHeal: boolean;
  selfHealEnabled: boolean;
}

async function executeAttempt<TSchema extends z.ZodTypeAny>(
  adapter: LLMAdapter,
  input: ExecuteAttemptInput<TSchema>,
): Promise<{ response: ModelCallResult; trace: StructuredAttempt<z.infer<TSchema>> }> {
  const response = await callModel(adapter, {
    prompt: input.prompt,
    systemPrompt: input.systemPrompt,
    request: input.request,
    stream: input.stream,
    observe: input.observe,
    debug: input.debug,
    attempt: input.attemptNumber,
    selfHeal: input.selfHeal,
    selfHealEnabled: input.selfHealEnabled,
  });

  const parsed = parseWithObserve(response.text, input.schema, input.parseOptions, {
    observe: input.observe,
    attempt: input.attemptNumber,
    selfHeal: input.selfHeal,
  });

  const trace: StructuredAttempt<z.infer<TSchema>> = {
    attempt: input.attemptNumber,
    selfHeal: input.selfHeal,
    via: response.via,
    raw: response.text,
    thinkBlocks: parsed.thinkBlocks,
    json: parsed.parsed,
    candidates: parsed.candidates.map((candidate) => candidate.content),
    repairLog: collectRepairLog(parsed),
    zodIssues: parsed.zodIssues,
    success: parsed.success,
    usage: response.usage,
    finishReason: response.finishReason,
    parsed,
  };

  return {
    response,
    trace,
  };
}

interface ModelCallOptions {
  prompt: string;
  systemPrompt?: string;
  request?: StructuredCallOptions<z.ZodTypeAny>["request"];
  stream: NormalizedStreamConfig;
  observe?: (event: StructuredTraceEvent) => void;
  debug: NormalizedDebugConfig;
  attempt: number;
  selfHeal: boolean;
  selfHealEnabled: boolean;
}

interface ModelCallResult {
  text: string;
  via: "complete" | "stream";
  usage?: LLMUsage;
  finishReason?: string;
}

async function callModel(adapter: LLMAdapter, options: ModelCallOptions): Promise<ModelCallResult> {
  const requestPayload: LLMRequest = {
    prompt: options.prompt,
    systemPrompt: options.systemPrompt,
    temperature: options.request?.temperature,
    maxTokens: options.request?.maxTokens,
    mcpClients: options.request?.mcpClients,
    toolChoice: options.request?.toolChoice,
    parallelToolCalls: options.request?.parallelToolCalls,
    maxToolRounds: options.request?.maxToolRounds,
    onToolExecution: options.request?.onToolExecution,
    toolDebug: options.request?.toolDebug,
    body: options.request?.body,
    signal: options.request?.signal,
  };

  emitDebugRequest(options.debug, {
    provider: adapter.provider,
    model: adapter.model,
    attempt: options.attempt,
    selfHealAttempt: options.selfHeal,
    selfHealEnabled: options.selfHealEnabled,
    stream: options.stream.enabled && !!adapter.stream,
    requestPayload,
  });

  emitObserve(options.observe, {
    stage: "llm.request",
    attempt: options.attempt,
    selfHeal: options.selfHeal,
    message: "Sending LLM request.",
    details: {
      provider: adapter.provider,
      model: adapter.model,
      stream: options.stream.enabled && !!adapter.stream,
    },
  });

  if (options.stream.enabled && adapter.stream) {
    let latestUsage: LLMUsage | undefined;
    let latestFinishReason: string | undefined;
    let streamedRaw = "";
    let sawToken = false;
    let lastDataFingerprint: string | undefined;

    const emitStreamingData = (
      raw: string,
      done: boolean,
      usage?: LLMUsage,
      finishReason?: string,
    ): void => {
      const data = parseStreamingStructuredData(raw);
      if (data === null && !done) {
        return;
      }

      const fingerprint = toStreamDataFingerprint(data ?? null);
      if (!done && fingerprint === lastDataFingerprint) {
        return;
      }

      lastDataFingerprint = fingerprint;
      options.stream.onData?.({
        data: data ?? null,
        raw,
        done,
        usage,
        finishReason,
      });

      emitObserve(options.observe, {
        stage: "llm.stream.data",
        attempt: options.attempt,
        selfHeal: options.selfHeal,
        message: done ? "Streaming structured data completed." : "Streaming structured data updated.",
        details: {
          done,
          finishReason,
        },
      });
    };

    const handleTextDelta = (delta: string): void => {
      if (!delta) {
        return;
      }

      streamedRaw += delta;

      if (options.stream.to === "stdout") {
        process.stdout.write(delta);
      }

      emitObserve(options.observe, {
        stage: "llm.stream.delta",
        attempt: options.attempt,
        selfHeal: options.selfHeal,
        message: "Received stream delta.",
        details: {
          chars: delta.length,
        },
      });

      emitStreamingData(streamedRaw, false);
    };

    const response = await adapter.stream(requestPayload, {
      onToken: (token) => {
        sawToken = true;
        handleTextDelta(token);
      },
      onChunk: (chunk) => {
        if (!sawToken && chunk.textDelta) {
          handleTextDelta(chunk.textDelta);
        }

        if (chunk.usage) {
          latestUsage = mergeUsage(latestUsage, chunk.usage);
        }

        if (chunk.finishReason) {
          latestFinishReason = chunk.finishReason;
        }
      },
    });

    const finalText =
      typeof response.text === "string" && response.text.length > 0 ? response.text : streamedRaw;
    const usage = mergeUsage(latestUsage, response.usage);
    const finishReason = response.finishReason ?? latestFinishReason;
    emitStreamingData(finalText, true, usage, finishReason);

    emitObserve(options.observe, {
      stage: "llm.response",
      attempt: options.attempt,
      selfHeal: options.selfHeal,
      message: "Streaming response completed.",
      details: {
        via: "stream",
        chars: finalText.length,
        finishReason,
      },
    });

    emitDebugResponse(options.debug, {
      attempt: options.attempt,
      selfHealAttempt: options.selfHeal,
      selfHealEnabled: options.selfHealEnabled,
      via: "stream",
      responseText: finalText,
      usage,
      finishReason,
    });

    return {
      text: finalText,
      via: "stream",
      usage,
      finishReason,
    };
  }

  const response = await adapter.complete(requestPayload);

  emitObserve(options.observe, {
    stage: "llm.response",
    attempt: options.attempt,
    selfHeal: options.selfHeal,
    message: "Completion response received.",
    details: {
      via: "complete",
      chars: response.text.length,
      finishReason: response.finishReason,
    },
  });

  emitDebugResponse(options.debug, {
    attempt: options.attempt,
    selfHealAttempt: options.selfHeal,
    selfHealEnabled: options.selfHealEnabled,
    via: "complete",
    responseText: response.text,
    usage: response.usage,
    finishReason: response.finishReason,
  });

  return {
    text: response.text,
    via: "complete",
    usage: response.usage,
    finishReason: response.finishReason,
  };
}

function parseStreamingStructuredData(raw: string): unknown | null {
  const sanitized = sanitizeThink(raw);
  const start = findFirstJsonRootStart(sanitized.visibleText);
  if (start < 0) {
    return null;
  }

  const candidate = sanitized.visibleText.slice(start).trim();
  if (!candidate) {
    return null;
  }

  try {
    const repaired = jsonrepair(candidate);
    const parsed = JSON.parse(repaired);
    if (typeof parsed !== "object" || parsed === null) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function findFirstJsonRootStart(input: string): number {
  let inString = false;
  let escaped = false;

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    if (!char) {
      continue;
    }

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }

      if (char === "\\") {
        escaped = true;
        continue;
      }

      if (char === '"') {
        inString = false;
      }

      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === "{" || char === "[") {
      return index;
    }
  }

  const objectStart = input.indexOf("{");
  const arrayStart = input.indexOf("[");

  if (objectStart < 0) {
    return arrayStart;
  }
  if (arrayStart < 0) {
    return objectStart;
  }

  return Math.min(objectStart, arrayStart);
}

function toStreamDataFingerprint(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "__unserializable__";
  }
}

function parseWithObserve<TSchema extends z.ZodTypeAny>(
  output: string,
  schema: TSchema,
  parseOptions: ParseLLMOutputOptions,
  context: {
    observe?: (event: StructuredTraceEvent) => void;
    attempt: number;
    selfHeal: boolean;
  },
) {
  const userParseTrace = parseOptions.onTrace;

  return parseLLMOutput(output, schema, {
    ...parseOptions,
    onTrace: (event: ParseTraceEvent) => {
      userParseTrace?.(event);
      emitObserve(context.observe, {
        stage: "parse",
        attempt: context.attempt,
        selfHeal: context.selfHeal,
        message: event.message,
        details: {
          level: event.level,
          stage: event.stage,
          candidateId: event.candidateId,
          details: event.details,
        },
      });
    },
  });
}

function collectRepairLog(parsed: {
  diagnostics: Array<{ usedRepair: boolean; message?: string }>;
}): string[] {
  const logs = parsed.diagnostics
    .filter((diagnostic) => diagnostic.usedRepair)
    .map((diagnostic) => diagnostic.message)
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0);

  return [...new Set(logs)];
}

function buildSuccessResult<T>(data: T, attempts: StructuredAttempt<T>[]): StructuredResult<T> {
  const final = attempts.at(-1);

  return {
    data,
    raw: final?.raw ?? "",
    thinkBlocks: final?.thinkBlocks ?? [],
    json: final?.json ?? null,
    attempts,
    usage: aggregateUsage(attempts),
    finishReason: final?.finishReason,
  };
}

function aggregateUsage<T>(attempts: StructuredAttempt<T>[]): LLMUsage | undefined {
  let usage: LLMUsage | undefined;

  for (const attempt of attempts) {
    usage = mergeUsage(usage, attempt.usage);
  }

  return usage;
}

function mergeUsage(base: LLMUsage | undefined, next: LLMUsage | undefined): LLMUsage | undefined {
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

function toStructuredError<T>(attempt: StructuredAttempt<T> | undefined): StructuredParseError {
  if (!attempt) {
    return new StructuredParseError({
      message: "Structured parsing failed before any model response.",
      raw: "",
      thinkBlocks: [],
      candidates: [],
      zodIssues: [],
      repairLog: [],
      attempt: 0,
    });
  }

  return new StructuredParseError({
    raw: attempt.raw,
    thinkBlocks: attempt.thinkBlocks,
    candidates: attempt.candidates,
    zodIssues: attempt.zodIssues,
    repairLog: attempt.repairLog,
    attempt: attempt.attempt,
  });
}

function emitObserve(
  observe: StructuredCallOptions<z.ZodTypeAny>["observe"],
  event: StructuredTraceEvent,
): void {
  observe?.(event);
}

interface DebugRequestInput {
  provider?: string;
  model?: string;
  attempt: number;
  selfHealAttempt: boolean;
  selfHealEnabled: boolean;
  stream: boolean;
  requestPayload: LLMRequest;
}

interface DebugResponseInput {
  attempt: number;
  selfHealAttempt: boolean;
  selfHealEnabled: boolean;
  via: "complete" | "stream";
  responseText: string;
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

  const lines = [
    color(
      config,
      title(
        config,
        [
          "[structured][request]",
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
    input.requestPayload.prompt,
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
  const lines = [
    color(
      config,
      title(
        config,
        [
          "[structured][response]",
          `attempt=${input.attempt}`,
          `selfHealEnabled=${input.selfHealEnabled}`,
          `selfHealAttempt=${input.selfHealAttempt}`,
        ].join(" "),
      ),
      "green",
    ),
    dim(
      config,
      [
        `via=${input.via}`,
        `chars=${input.responseText.length}`,
        `finishReason=${input.finishReason ?? "unknown"}`,
        `usage=${JSON.stringify(input.usage ?? {})}`,
      ].join(" "),
    ),
    color(config, "text:", "yellow"),
    input.responseText,
  ];

  emitDebug(config, lines.join("\n"));
}

function emitDebug(config: NormalizedDebugConfig, message: string): void {
  if (!config.enabled) {
    return;
  }

  config.logger(message);
}
