import type { LLMReasoningEffort } from "../types";

/** Values accepted by the OpenAI `reasoning.effort` / `reasoning_effort` field. */
export type OpenAIReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh";

/** Values accepted by the Anthropic `output_config.effort` field. */
export type AnthropicReasoningEffort = "low" | "medium" | "high" | "xhigh" | "max";

/**
 * OpenAI's scale tops out at `xhigh`, so the canonical `max` lands there: it is
 * the same ceiling under the name the other vendor happens to use.
 */
const OPENAI_EFFORT: Record<LLMReasoningEffort, OpenAIReasoningEffort> = {
  none: "none",
  minimal: "minimal",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
  max: "xhigh",
};

/**
 * Anthropic has no effort level below `low` and none for "do not reason at all":
 * `minimal` collapses onto `low`, and `none` maps to no effort field at all so
 * the adapter can send `thinking: { type: "disabled" }` instead. `xhigh` and
 * `max` are genuinely distinct levels here and are kept apart.
 */
const ANTHROPIC_EFFORT: Record<LLMReasoningEffort, AnthropicReasoningEffort | null> = {
  none: null,
  minimal: "low",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
  max: "max",
};

/**
 * Look an effort up in a provider table. An unknown value can only reach here
 * from untyped JavaScript, and silently dropping it would run the request at
 * the provider default: the caller asking for `low` would quietly pay for
 * `high`. Fail loudly instead.
 */
function lookupEffort<T>(
  table: Record<LLMReasoningEffort, T>,
  effort: LLMReasoningEffort,
): T {
  if (!Object.hasOwn(table, effort)) {
    throw new RangeError(
      `Unknown reasoningEffort ${JSON.stringify(effort)}. Expected one of: ${Object.keys(table).join(", ")}.`,
    );
  }
  return table[effort];
}

export function toOpenAIReasoningEffort(
  effort: LLMReasoningEffort | undefined,
): OpenAIReasoningEffort | undefined {
  return effort === undefined ? undefined : lookupEffort(OPENAI_EFFORT, effort);
}

export function toAnthropicReasoningEffort(
  effort: LLMReasoningEffort | undefined,
): AnthropicReasoningEffort | undefined {
  return effort === undefined ? undefined : (lookupEffort(ANTHROPIC_EFFORT, effort) ?? undefined);
}
