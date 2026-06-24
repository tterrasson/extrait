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
    sanitizeReasoningText(dedicatedReasoning),
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

function sanitizeReasoningText(value: string | undefined): string | undefined {
  const sanitized = value?.replace(RE_THINK_TAGS, "").trim();
  return sanitized ? sanitized : undefined;
}

const THINK_TAG_VARIANTS = ["<think>", "</think>"] as const;
const MAX_THINK_TAG_PREFIX = Math.max(...THINK_TAG_VARIANTS.map((tag) => tag.length)) - 1;

/**
 * Drops a trailing run of `value` that could be the start of a `<think>` /
 * `</think>` tag whose remaining characters have not streamed in yet (e.g. a
 * chunk ending in `<th`). Streaming consumers diff successive snapshots to emit
 * deltas; without this, a partial tag is emitted as a delta and then can never
 * be retracted once it resolves into a real tag that sanitization removes.
 *
 * Only safe for incremental streaming snapshots — never apply it to a final
 * result, where a legitimate trailing `<` must survive.
 */
export function withoutTrailingThinkTagPrefix(value: string): string {
  const max = Math.min(value.length, MAX_THINK_TAG_PREFIX);
  for (let length = max; length > 0; length -= 1) {
    const suffix = value.slice(value.length - length);
    // A strict prefix of a tag (tag.length > suffix.length) is a partial tag;
    // a complete tag is left untouched — sanitization already handled it.
    if (THINK_TAG_VARIANTS.some((tag) => tag.length > suffix.length && tag.startsWith(suffix))) {
      return value.slice(0, value.length - length);
    }
  }
  return value;
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
