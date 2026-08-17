import type { z } from "zod";
import { resolveSchemaInstruction, formatPrompt, withFormat } from "./format";
import { formatZodIssues, parseLLMOutput } from "./parse";
import { createStreamingStructuredParser } from "./structured-streaming";
import {
  aggregateUsage,
  applyOutdentToOptionalPrompt,
  applyPromptOutdent,
  applyToolTimeout,
  callModel as callModelShared,
  composeParseSource,
  mergeSystemPrompts,
  normalizeDebugConfig,
  normalizeStreamConfig,
  resolvePrompt,
  type ModelCallOptions,
  type ModelCallResult,
  type NormalizedDebugConfig,
  type NormalizedStreamConfig,
} from "./generate-shared";
import type {
  LLMAdapter,
  LLMMessage,
  ParseLLMOutputOptions,
  ParseTraceEvent,
  ReasoningBlock,
  StreamTurnTransition,
  StructuredAttempt,
  StructuredCallOptions,
  StructuredError,
  StructuredMode,
  StructuredOptions,
  StructuredPromptBuilder,
  StructuredResult,
  StructuredTimeoutOptions,
  StructuredTraceEvent,
} from "./types";

export class StructuredParseError extends Error implements StructuredError {
  override readonly name = "StructuredParseError" as const;
  readonly text: string;
  readonly reasoning: string;
  readonly candidates: string[];
  readonly zodIssues?: z.core.$ZodIssue[];
  readonly repairLog?: string[];
  readonly attempt: number;

  constructor(input: {
    message?: string;
    text: string;
    reasoning: string;
    candidates: string[];
    zodIssues?: z.core.$ZodIssue[];
    repairLog?: string[];
    attempt: number;
  }) {
    super(input.message ?? `Structured parsing failed after ${input.attempt} attempt(s).`);
    this.text = input.text;
    this.reasoning = input.reasoning;
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
  issues: z.core.$ZodIssue[];
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
  const truncatedRawOutput = truncateForPrompt(input.rawOutput, maxContextChars);
  const truncatedSelectedOutput = truncateForPrompt(selectedOutput, maxContextChars);
  const truncatedSanitizedOutput = input.sanitizedOutput
    ? truncateForPrompt(input.sanitizedOutput, maxContextChars)
    : undefined;
  const contextPayload = {
    protocol: DEFAULT_SELF_HEAL_PROTOCOL,
    attempt: input.attempt,
    previousAttempt: input.previousAttempt,
    selectedInput: input.selectedInput ?? "raw",
    issueSummary: issueText,
    validationIssues: input.issues.map((issue) => ({
      path: formatIssuePath(issue.path as Array<string | number>),
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
    // The raw output is already rendered verbatim below, and the selected /
    // sanitized variants are usually the very same string. Echoing all four
    // billed up to 4x `maxContextChars` per heal attempt, so only keep the
    // variants that actually differ.
    selectedOutput: truncatedSelectedOutput === truncatedRawOutput ? undefined : truncatedSelectedOutput,
    sanitizedOutput: truncatedSanitizedOutput === truncatedRawOutput ? undefined : truncatedSanitizedOutput,
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
    // Delimit the model-authored payloads so their content cannot read as part
    // of the surrounding repair instructions. The payloads are escaped so a
    // model emitting the closing tag itself cannot break out of the container.
    "<raw_output>",
    escapeSelfHealDelimiters(truncatedRawOutput),
    "</raw_output>",
    "",
    text.contextLabel,
    "<self_heal_context>",
    escapeSelfHealDelimiters(JSON.stringify(contextPayload, null, 2)),
    "</self_heal_context>",
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
  const streamConfig = normalizeStreamConfig<{
    text: string;
    reasoning: string;
    reasoningBlocks?: ReasoningBlock[];
    data: unknown | null;
  }>(normalized.stream as {
    enabled?: boolean;
    onData?: (event: {
      delta: { text: string; reasoning: string };
      snapshot: {
        text: string;
        reasoning: string;
        reasoningBlocks?: ReasoningBlock[];
        data: unknown | null;
      };
      done: boolean;
    }) => void;
    onTurnTransition?: (transition: StreamTurnTransition) => void;
    to?: "stdout";
  } | boolean | undefined);
  const debugConfig = normalizeDebugConfig(normalized.debug);
  const attempts: StructuredAttempt<z.infer<TSchema>>[] = [];
  const useOutdent = normalized.outdent ?? true;

  const resolvedPrompt = applyPromptOutdent(resolvePrompt(normalized.prompt, { mode }), useOutdent);
  const resolvedSystemPrompt = applyOutdentToOptionalPrompt(normalized.systemPrompt, useOutdent);
  const preparedPrompt = prepareStructuredPromptPayload(
    resolvedPrompt,
    resolvedSystemPrompt,
    normalized.schema,
    normalized.schemaInstruction,
  );

  const resolvedRequest =
    normalized.timeout?.tool !== undefined && normalized.request?.mcpClients !== undefined
      ? {
          ...normalized.request,
          mcpClients: applyToolTimeout(normalized.request.mcpClients, normalized.timeout.tool),
        }
      : normalized.request;

  const first = await executeAttempt(adapter, {
    prompt: preparedPrompt.prompt,
    messages: preparedPrompt.messages,
    schema: normalized.schema,
    parseOptions,
    stream: streamConfig,
    request: resolvedRequest,
    systemPrompt: preparedPrompt.systemPrompt,
    observe: normalized.observe,
    debug: debugConfig,
    attemptNumber: 1,
    selfHeal: false,
    selfHealEnabled: selfHealConfig.enabled,
    timeout: normalized.timeout,
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
      rawOutput: composeParseSource(previous.text, previous.reasoning),
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

    const healPayload = buildSelfHealPayload(preparedPrompt, repairPrompt);

    const healed = await executeAttempt(adapter, {
      prompt: healPayload.prompt,
      messages: healPayload.messages,
      schema: normalized.schema,
      parseOptions,
      stream: streamConfig,
      request: resolvedRequest,
      systemPrompt: healPayload.systemPrompt,
      observe: normalized.observe,
      debug: debugConfig,
      attemptNumber,
      selfHeal: true,
      selfHealEnabled: selfHealConfig.enabled,
      timeout: normalized.timeout,
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
    ...callOptions,
    schema: schemaOrOptions,
    prompt: promptInput,
  };
}

function isStructuredOptions<TSchema extends z.ZodTypeAny>(
  value: TSchema | StructuredOptions<TSchema>,
): value is StructuredOptions<TSchema> {
  return typeof value === "object" && value !== null && "schema" in value && "prompt" in value;
}

function prepareStructuredPromptPayload<TSchema extends z.ZodTypeAny>(
  payload: ReturnType<typeof resolvePrompt>,
  systemPrompt: string | undefined,
  schema: TSchema,
  schemaInstruction: string | undefined,
): { prompt?: string; systemPrompt?: string; messages?: LLMMessage[] } {
  if (Array.isArray(payload.messages) && payload.messages.length > 0) {
    const messages = payload.messages.map((message) => ({ ...message }));
    const mergedSystemPrompt = mergeSystemPrompts(payload.systemPrompt, systemPrompt);
    const systemMessages = mergedSystemPrompt ? [{ role: "system" as const, content: mergedSystemPrompt }] : [];

    return {
      messages: injectStructuredFormatIntoMessages([...systemMessages, ...messages], schema, schemaInstruction),
    };
  }

  const resolvedPrompt = payload.prompt?.trim();
  if (!resolvedPrompt) {
    throw new Error("Structured prompt payload must include a non-empty prompt or messages.");
  }

  return {
    prompt: shouldInjectFormat(resolvedPrompt, schemaInstruction)
      ? formatPrompt(schema, resolvedPrompt, {
          schemaInstruction,
        })
      : resolvedPrompt,
    systemPrompt: mergeSystemPrompts(payload.systemPrompt, systemPrompt),
  };
}

/**
 * Builds the request payload for a self-heal attempt.
 *
 * When the original call used `messages`, the repair prompt is appended as a new
 * user turn on top of the original conversation instead of replacing it: the
 * prior turns, the system prompt (merged into the messages upstream), and any
 * image blocks all stay visible to the model. Re-extracting from an image the
 * model can no longer see would otherwise be impossible.
 */
function buildSelfHealPayload(
  payload: { prompt?: string; systemPrompt?: string; messages?: LLMMessage[] },
  repairPrompt: string,
): { prompt?: string; systemPrompt?: string; messages?: LLMMessage[] } {
  if (payload.messages && payload.messages.length > 0) {
    return {
      messages: [
        ...payload.messages.map((message) => ({ ...message })),
        { role: "user" as const, content: repairPrompt },
      ],
    };
  }

  return {
    prompt: repairPrompt,
    systemPrompt: payload.systemPrompt,
  };
}

function injectStructuredFormatIntoMessages<TSchema extends z.ZodTypeAny>(
  messages: LLMMessage[],
  schema: TSchema,
  schemaInstruction: string | undefined,
): LLMMessage[] {
  const lastUserIndex = findLastUserMessageIndex(messages);
  if (lastUserIndex === -1) {
    throw new Error("Structured prompts with messages must include at least one user message.");
  }

  const target = messages[lastUserIndex];

  // Multimodal content (array with text + image blocks): inject schema into the text block
  // and preserve image blocks intact.
  if (Array.isArray(target?.content)) {
    const parts = target.content as Array<{ type: string; text?: string; [key: string]: unknown }>;
    const textIndex = parts.findIndex((p) => p.type === "text");
    const existingText = textIndex !== -1 ? (parts[textIndex]?.text ?? "").trim() : "";
    const formatted = shouldInjectFormat(existingText, schemaInstruction)
      ? formatPrompt(schema, existingText, { schemaInstruction })
      : existingText;

    let newParts: typeof parts;
    if (textIndex !== -1) {
      newParts = parts.map((p, i) => (i === textIndex ? { ...p, text: formatted } : p));
    } else {
      newParts = [{ type: "text", text: formatted }, ...parts];
    }

    return messages.map((message, index) =>
      index === lastUserIndex ? { ...message, content: newParts as LLMMessage["content"] } : message,
    );
  }

  const content = typeof target?.content === "string" ? target.content.trim() : stringifyPromptContent(target?.content);
  const formatted = shouldInjectFormat(content, schemaInstruction)
    ? formatPrompt(schema, content, { schemaInstruction })
    : content.trim();

  return messages.map((message, index) =>
    index === lastUserIndex
      ? {
          ...message,
          content: formatted,
        }
      : message,
  );
}

function findLastUserMessageIndex(messages: LLMMessage[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      return index;
    }
  }

  return -1;
}

function stringifyPromptContent(content: unknown): string {
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

// Model-authored payloads are wrapped in fixed tags; neutralize any occurrence
// of those tags inside the payload so the container cannot be closed early.
const RE_SELF_HEAL_DELIMITER = /<(\/?)(raw_output|self_heal_context)>/gi;

export function escapeSelfHealDelimiters(value: string): string {
  return value.replace(RE_SELF_HEAL_DELIMITER, (_match, slash: string, tag: string) => (
    `&lt;${slash}${tag}&gt;`
  ));
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
    text: composeParseSource(attempt.text, attempt.reasoning),
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
        `${formatIssuePath(issue.path as Array<string | number>)}:${issue.code}:${normalizeWhitespace(issue.message)}`,
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

interface ExecuteAttemptInput<TSchema extends z.ZodTypeAny> {
  prompt?: string;
  messages?: LLMMessage[];
  schema: TSchema;
  parseOptions: ParseLLMOutputOptions;
  stream: NormalizedStreamConfig<{
    text: string;
    reasoning: string;
    reasoningBlocks?: ReasoningBlock[];
    data: unknown | null;
  }>;
  request?: StructuredCallOptions<TSchema>["request"];
  systemPrompt?: string;
  observe?: StructuredCallOptions<TSchema>["observe"];
  debug: NormalizedDebugConfig;
  attemptNumber: number;
  selfHeal: boolean;
  selfHealEnabled: boolean;
  timeout?: StructuredTimeoutOptions;
}

type StructuredModelCallOptions = Omit<
  ModelCallOptions<
    { text: string; reasoning: string; reasoningBlocks?: ReasoningBlock[]; data: unknown | null },
    StructuredTraceEvent
  >,
  "buildEvent" | "buildSnapshot" | "debugLabel"
>;

async function executeAttempt<TSchema extends z.ZodTypeAny>(
  adapter: LLMAdapter,
  input: ExecuteAttemptInput<TSchema>,
): Promise<{ response: ModelCallResult; trace: StructuredAttempt<z.infer<TSchema>> }> {
  const response = await callModel(adapter, {
    prompt: input.prompt,
    messages: input.messages,
    systemPrompt: input.systemPrompt,
    request: input.request,
    stream: input.stream,
    observe: input.observe,
    debug: input.debug,
    attempt: input.attemptNumber,
    selfHeal: input.selfHeal,
    selfHealEnabled: input.selfHealEnabled,
    timeout: input.timeout,
  });

  const parsed = parseWithObserve(response.parseSource, input.schema, input.parseOptions, {
    observe: input.observe,
    attempt: input.attemptNumber,
    selfHeal: input.selfHeal,
  });

  const trace: StructuredAttempt<z.infer<TSchema>> = {
    attempt: input.attemptNumber,
    selfHeal: input.selfHeal,
    via: response.via,
    text: response.text,
    reasoning: response.reasoning,
    json: parsed.parsed,
    candidates: parsed.candidates.map((candidate) => candidate.content),
    repairLog: collectRepairLog(parsed),
    zodIssues: parsed.zodIssues,
    success: parsed.success,
    usage: response.usage,
    finishReason: response.finishReason,
    ...(response.logprobs ? { logprobs: response.logprobs } : {}),
    ...(response.reasoningBlocks ? { reasoningBlocks: response.reasoningBlocks } : {}),
    parsed,
  };

  return {
    response,
    trace,
  };
}

// Parsing itself is incremental, but each preview still copies the containers
// that are open, which grows with the size of the one being filled. Below this
// size that copy is free; above it the default coalesces to a UI-frame-scale
// interval so a long array does not pay for it on every chunk. The worst case
// (bench:stream, structured+flatArray0) prices a whole exact-mode stream at
// ~3 ms up to this size; the quadratic term only bites tens of kilobytes in.
const AUTO_DATA_INTERVAL_EXACT_MAX_CHARS = 8_192;
const AUTO_DATA_INTERVAL_MS = 25;

async function callModel(
  adapter: LLMAdapter,
  options: StructuredModelCallOptions,
): Promise<ModelCallResult> {
  // stream.dataInterval coalesces the preview: between recomputations the last
  // value is reused, so `snapshot.data` may lag behind `snapshot.text` by up to
  // that many ms. The terminal (done) snapshot always recomputes. When unset,
  // the interval is adaptive, exact on every event while the output stays
  // small, coalesced above AUTO_DATA_INTERVAL_EXACT_MAX_CHARS.
  const dataInterval = options.stream.dataInterval;
  const parsePreview = createStreamingStructuredParser();
  let coalescedData: unknown = null;
  let coalescedDataAt: number | undefined;

  return callModelShared(adapter, {
    ...options,
    buildEvent: ({ stage, message, details }) => ({
      stage,
      attempt: options.attempt,
      selfHeal: options.selfHeal,
      message,
      details,
    }),
    buildSnapshot: (normalized, meta) => {
      const interval =
        dataInterval ??
        (normalized.text.length <= AUTO_DATA_INTERVAL_EXACT_MAX_CHARS ? 0 : AUTO_DATA_INTERVAL_MS);
      const now = performance.now();
      if (
        meta.done ||
        interval <= 0 ||
        coalescedDataAt === undefined ||
        now - coalescedDataAt >= interval
      ) {
        coalescedData = parsePreview.update(normalized.text) ?? null;
        coalescedDataAt = now;
      }
      return {
        text: normalized.text,
        reasoning: normalized.reasoning,
        ...(normalized.reasoningBlocks ? { reasoningBlocks: normalized.reasoningBlocks } : {}),
        data: coalescedData,
      };
    },
    debugLabel: "structured",
  });
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
    text: final?.text ?? "",
    reasoning: final?.reasoning ?? "",
    json: final?.json ?? null,
    attempts,
    usage: aggregateUsage(attempts),
    finishReason: final?.finishReason,
    ...(final?.logprobs ? { logprobs: final.logprobs } : {}),
    ...(final?.reasoningBlocks ? { reasoningBlocks: final.reasoningBlocks } : {}),
  };
}

function toStructuredError<T>(attempt: StructuredAttempt<T> | undefined): StructuredParseError {
  if (!attempt) {
    return new StructuredParseError({
      message: "Structured parsing failed before any model response.",
      text: "",
      reasoning: "",
      candidates: [],
      zodIssues: [],
      repairLog: [],
      attempt: 0,
    });
  }

  return new StructuredParseError({
    text: attempt.text,
    reasoning: attempt.reasoning,
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
