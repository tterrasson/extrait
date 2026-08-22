import { describe, expect, test } from "bun:test";
import {
  toAnthropicReasoningEffort,
  toOpenAIReasoningEffort,
} from "@/providers/reasoning-effort";
import type { LLMReasoningEffort } from "@/types";

describe("reasoning effort mapping", () => {
  test("passes an undefined effort through untouched", () => {
    expect(toOpenAIReasoningEffort(undefined)).toBeUndefined();
    expect(toAnthropicReasoningEffort(undefined)).toBeUndefined();
  });

  test("throws on an effort outside the canonical scale", () => {
    const bogus = "ultra" as LLMReasoningEffort;
    expect(() => toOpenAIReasoningEffort(bogus)).toThrow(RangeError);
    expect(() => toAnthropicReasoningEffort(bogus)).toThrow(RangeError);
  });

  test("does not resolve inherited Object keys as efforts", () => {
    const inherited = "toString" as LLMReasoningEffort;
    expect(() => toOpenAIReasoningEffort(inherited)).toThrow(RangeError);
    expect(() => toAnthropicReasoningEffort(inherited)).toThrow(RangeError);
  });
});
