import { sanitizeThink } from "./think";
import type { LLMUsage, ReasoningBlock, StreamTurnTransition, ThinkBlock } from "./types";
import type { NormalizedModelOutput } from "./generate-shared";

const RE_THINK_TAGS = /<\/?think\s*>/gi;

export function normalizeModelOutput(
  text: string,
  dedicatedReasoning?: string,
  reasoningBlocks?: ReasoningBlock[],
): NormalizedModelOutput {
  const sanitized = sanitizeThink(text);
  const visibleText = stripThinkBlocks(text, sanitized.thinkBlocks);
  const reasoning = joinReasoningSegments([
    dedicatedReasoning,
    ...sanitized.thinkBlocks.map((block) => block.content),
  ]);

  return {
    text: visibleText,
    reasoning,
    reasoningBlocks: normalizeReasoningBlocks(reasoningBlocks),
    thinkBlocks: sanitized.thinkBlocks,
    parseSource: composeParseSource(visibleText, reasoning),
  };
}

function normalizeReasoningBlocks(blocks: ReasoningBlock[] | undefined): ReasoningBlock[] | undefined {
  if (!Array.isArray(blocks)) {
    return undefined;
  }

  const normalized = blocks
    .map((block) => ({
      turnIndex: block.turnIndex,
      text: block.text.replace(RE_THINK_TAGS, "").trim(),
    }))
    .filter((block) => Number.isFinite(block.turnIndex) && block.text.length > 0);

  return normalized.length > 0 ? normalized : undefined;
}

export function appendReasoningBlock(
  blocks: ReasoningBlock[] | undefined,
  transition: StreamTurnTransition,
): ReasoningBlock[] | undefined {
  const text = transition.reasoningText?.replace(RE_THINK_TAGS, "").trim();
  if (!text) {
    return blocks;
  }

  const next = [...(blocks ?? []), { turnIndex: transition.turnIndex, text }];
  return normalizeReasoningBlocks(next);
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

export function toStreamDataFingerprint(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "__unserializable__";
  }
}
