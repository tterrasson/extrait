import { describe, expect, test } from "bun:test";
import { normalizeModelOutput, withoutTrailingThinkTagPrefix } from "@/generate-output";

describe("normalizeModelOutput dedicated reasoning", () => {
  // Mirrors the call shape used by streaming consumers that diff snapshots of
  // the dedicated reasoning field, e.g. normalizeModelOutput("", rawReasoning).
  // A model that emits a bare/empty <think> tag in the reasoning field must not
  // surface the literal tag as reasoning, otherwise consumers stream a stray
  // "<think>" delta for what is actually empty reasoning.
  test("drops a bare think tag from the reasoning field", () => {
    expect(normalizeModelOutput("", "<think>").reasoning).toBe("");
    expect(normalizeModelOutput("", "</think>").reasoning).toBe("");
    expect(normalizeModelOutput("", "<think></think>").reasoning).toBe("");
    expect(normalizeModelOutput("", "  <think>\n</think>  ").reasoning).toBe("");
  });

  test("keeps reasoning content while stripping wrapping think tags", () => {
    expect(normalizeModelOutput("", "<think>real plan</think>").reasoning).toBe("real plan");
    expect(normalizeModelOutput("", "plan</think>leak").reasoning).toBe("planleak");
  });

  test("merges dedicated reasoning with think blocks found in the text", () => {
    const result = normalizeModelOutput("answer <think>inline</think>", "dedicated");
    expect(result.text).toBe("answer ");
    expect(result.reasoning).toBe("dedicated\n\ninline");
  });
});

describe("withoutTrailingThinkTagPrefix", () => {
  test("withholds a trailing partial open/close tag", () => {
    expect(withoutTrailingThinkTagPrefix("done <")).toBe("done ");
    expect(withoutTrailingThinkTagPrefix("done <th")).toBe("done ");
    expect(withoutTrailingThinkTagPrefix("done <think")).toBe("done ");
    expect(withoutTrailingThinkTagPrefix("done </")).toBe("done ");
    expect(withoutTrailingThinkTagPrefix("done </think")).toBe("done ");
  });

  test("keeps text that only resembles a tag mid-string or completes a word", () => {
    expect(withoutTrailingThinkTagPrefix("a < b")).toBe("a < b");
    expect(withoutTrailingThinkTagPrefix("I was <thinking")).toBe("I was <thinking");
    expect(withoutTrailingThinkTagPrefix("plain text")).toBe("plain text");
    expect(withoutTrailingThinkTagPrefix("")).toBe("");
  });

  test("preserves the fragment across successive snapshots without leaking it", () => {
    // Simulates the streaming diff: a `<think>` tag split across deltas must
    // never surface as a delta, and the reasoning after it must still be emitted.
    let raw = "";
    let emitted = "";
    const deltas: string[] = [];
    for (const chunk of ["<th", "ink>", "real ", "plan</thi", "nk>"]) {
      raw += chunk;
      const stable = withoutTrailingThinkTagPrefix(normalizeModelOutput("", raw).reasoning);
      if (stable.startsWith(emitted)) {
        const delta = stable.slice(emitted.length);
        if (delta.length > 0) {
          deltas.push(delta);
          emitted = stable;
        }
      }
    }
    // Final flush: full normalized reasoning, no withholding.
    const final = normalizeModelOutput("", raw).reasoning;
    if (final.startsWith(emitted) && final.length > emitted.length) {
      deltas.push(final.slice(emitted.length));
    }

    expect(deltas.join("")).toBe("real plan");
    expect(deltas.join("")).not.toContain("<");
    expect(final).toBe("real plan");
  });
});
