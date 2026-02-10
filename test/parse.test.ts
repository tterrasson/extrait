import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { formatZodIssues, parseLLMOutput } from "../src/parse";

const Schema = z
  .object({
    val: z.array(z.string()).default([]),
    va2: z.coerce.number(),
  })
  .passthrough();

describe("parseLLMOutput", () => {
  test("extracts and repairs a JSON-ish surrounded by text", () => {
    const input = `Yes here is the result: { val: ['a', 'b',], va2: '42', extra: true, }`;

    const result = parseLLMOutput(input, Schema, {
      repair: true,
      maxCandidates: 5,
    });

    expect(result.success).toBe(true);
    expect(result.data).toEqual({
      val: ["a", "b"],
      va2: 42,
      extra: true,
    });
  });

  test("continues across multiple candidates until the first valid one", () => {
    const input = [
      "Attempt 1: {'wrong': true}",
      "```json",
      "{\"va2\": \"7\", \"val\": [\"ok\"]}",
      "```",
    ].join("\n");

    const result = parseLLMOutput(input, Schema, {
      repair: true,
      maxCandidates: 5,
    });

    expect(result.success).toBe(true);
    expect(result.data).toEqual({
      val: ["ok"],
      va2: 7,
    });
  });

  test("applies Zod defaults", () => {
    const input = `{"va2":"9"}`;

    const result = parseLLMOutput(input, Schema, {
      repair: true,
    });

    expect(result.success).toBe(true);
    expect(result.data).toEqual({
      val: [],
      va2: 9,
    });
  });

  test("repair false rejects invalid JSON without attempting repair", () => {
    const result = parseLLMOutput("{ val: ['a'], va2: '42' }", Schema, {
      repair: false,
    });

    expect(result.success).toBe(false);
    expect(result.diagnostics.every((d) => !d.usedRepair || !d.parseSuccess)).toBe(true);
  });

  test("empty input returns 0 candidates and fails", () => {
    const result = parseLLMOutput("", Schema);

    expect(result.success).toBe(false);
    expect(result.candidates.length).toBe(0);
    expect(result.errors.some((e) => e.stage === "extract")).toBe(true);
  });

  test("extracts think blocks and ignores them for JSON extraction", () => {
    const input = [
      "<think>",
      '{"va2":"0","val":["hidden"]}',
      "</think>",
      '{"va2":"9","val":["visible"]}',
    ].join("\n");

    const result = parseLLMOutput(input, Schema, {
      repair: true,
    });

    expect(result.success).toBe(true);
    expect(result.data).toEqual({
      va2: 9,
      val: ["visible"],
    });
    expect(result.thinkBlocks.length).toBe(1);
    expect(result.thinkBlocks[0]?.content).toContain('"hidden"');
    expect(result.sanitizedRaw).not.toContain('"hidden"');
    expect(result.thinkDiagnostics.unterminatedCount).toBe(0);
  });

  test("fails closed when think block is not terminated", () => {
    const input = 'prefix <think>{"va2":"9","val":["hidden"]}';

    const result = parseLLMOutput(input, Schema, {
      repair: true,
    });

    expect(result.success).toBe(false);
    expect(result.thinkBlocks.length).toBe(1);
    expect(result.thinkDiagnostics.unterminatedCount).toBe(1);
    expect(result.sanitizedRaw).not.toContain('"hidden"');
  });

  test("multiple candidates: selected diagnostic is marked", () => {
    const input = [
      '{"invalid": true}',
      '{"va2": "5", "val": ["ok"]}',
    ].join(" ");

    const result = parseLLMOutput(input, Schema, {
      repair: true,
      maxCandidates: 5,
    });

    expect(result.success).toBe(true);
    const selected = result.diagnostics.find((d) => d.selected);
    expect(selected).toBeDefined();
  });

  test("repair enabled with deeply broken input still provides diagnostics", () => {
    const result = parseLLMOutput("just plain text no json at all", Schema, {
      repair: true,
    });

    expect(result.success).toBe(false);
    expect(result.diagnostics.length).toBeGreaterThan(0);
  });

  test("falls back to non-hinted candidates when top-K hinted candidates are invalid", () => {
    const noisyCandidates = Array.from(
      { length: 10 },
      (_, index) => `{"long_invalid_key_name_${index}":true}`,
    ).join(" ");
    const input = `${noisyCandidates} {"va2":"9"}`;

    const result = parseLLMOutput(input, Schema, {
      repair: true,
      maxCandidates: 11,
    });

    expect(result.success).toBe(true);
    expect(result.data).toEqual({
      va2: 9,
      val: [],
    });
  });

  test("does not build repair hints when repair is disabled", () => {
    const input = "{ val: ['a'], va2: '42' }";
    const result = parseLLMOutput(input, Schema, {
      repair: false,
      maxCandidates: 5,
    });

    expect(result.success).toBe(false);
    expect(result.candidates.every((candidate) => !candidate.parseHint?.usedRepair)).toBe(true);
  });

  test("forwards extraction heuristics to candidate extraction", () => {
    const input = '{"va2":"42","val":["a"]}';
    const result = parseLLMOutput(input, Schema, {
      extraction: {
        secondPassMin: 0,
        secondPassCap: 0,
        secondPassMultiplier: 0,
      },
    });

    expect(result.success).toBe(true);
    expect(result.candidates[0]?.parseHint).toBeUndefined();
  });
});

describe("formatZodIssues", () => {
  test("formats nested paths", () => {
    const result = formatZodIssues([
      { path: ["user", "name"], message: "Required", code: "invalid_type", expected: "string", received: "undefined" } as z.ZodIssue,
      { path: [], message: "Unrecognized key", code: "unrecognized_keys", keys: [] } as unknown as z.ZodIssue,
    ]);

    expect(result).toBe("user.name: Required\n<root>: Unrecognized key");
  });

  test("returns a default message for an empty array", () => {
    expect(formatZodIssues([])).toBe("Validation failed without details.");
  });
});
