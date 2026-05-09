import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { formatZodIssues, parseLLMOutput } from "@/parse";

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

  test("repair can recover invalid escaped punctuation inside strings", () => {
    const input = '{"va2":"42","val":["I love this product\\!"]}';

    const strict = parseLLMOutput(input, Schema, {
      repair: false,
    });
    expect(strict.success).toBe(false);
    expect(strict.errors.some((error) => error.stage === "parse")).toBe(true);

    const repaired = parseLLMOutput(input, Schema, {
      repair: true,
    });
    expect(repaired.success).toBe(true);
    expect(repaired.data).toEqual({
      va2: 42,
      val: ["I love this product!"],
    });
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

describe("parseLLMOutput - repair and hint edge cases", () => {
  test("repair:false on invalid JSON returns parse-stage error without repair attempt", () => {
    // Exercises tryParseJsonCandidate with allowRepair=false + JSON.parse failure (lines 241-249)
    const input = "{ val: ['a'], va2: '42' }";
    const result = parseLLMOutput(input, Schema, { repair: false });

    expect(result.success).toBe(false);
    const parseErrors = result.errors.filter((e) => e.stage === "parse");
    expect(parseErrors.length).toBeGreaterThan(0);
  });

  test("hint with usedRepair:true is discarded when repair:false", () => {
    // parseAttemptFromHint branch: hint.success=true && hint.usedRepair && !allowRepair → return null (line 297-298)
    const repairableInput = '{"va2":"42","val":["I love this product\\!"]}';
    const result = parseLLMOutput(repairableInput, Schema, {
      repair: false,
      maxCandidates: 5,
    });

    expect(result.success).toBe(false);
    // No candidate should have succeeded via repair when repair is disabled
    expect(result.candidates.every((c) => !c.parseHint?.usedRepair || !c.parseHint?.success)).toBe(true);
  });

  test("repair path is used on invalid JSON: usedRepair is true in successful repair", () => {
    // Exercises tryParseJsonCandidate repair path (lines 253-275): invalid JSON → repaired → parsed
    const repairable = '{"va2": 42, "val": ["I love this product\\!"]}';
    const result = parseLLMOutput(repairable, Schema, { repair: true });

    expect(result.success).toBe(true);
    // The winning candidate should have used repair
    const winning = result.candidates.find((c) => c.parseHint?.success && c.parseHint?.usedRepair);
    expect(winning).toBeDefined();
  });

  test("markSelectedDiagnostic marks exactly one diagnostic when multiple exist", () => {
    // Exercises the loop body in markSelectedDiagnostic (lines 349-352)
    const input = [
      '{"va2": "1"}',
      '{"va2": "2", "val": ["a"]}',
      '{"va2": "3", "val": ["b", "c"]}',
    ].join(" ");

    const result = parseLLMOutput(input, Schema, { repair: true, maxCandidates: 5 });

    expect(result.success).toBe(true);
    const selected = result.diagnostics.filter((d) => d.selected);
    expect(selected).toHaveLength(1);
    const notSelected = result.diagnostics.filter((d) => !d.selected);
    expect(notSelected.length).toBe(result.diagnostics.length - 1);
  });

  test("hint failed without repair is forwarded as-is when repair:false (parseAttemptFromHint fallback)", () => {
    // parseAttemptFromHint branch: hint.success=false, hint.usedRepair=false, allowRepair=false → return hint error
    // This is achieved with invalid JSON and repair:false but with hints pre-built by extraction
    const input = '{"va2":1} {"invalid json here';
    const result = parseLLMOutput(input, Schema, { repair: false, maxCandidates: 5 });

    // The first candidate should succeed (valid JSON), verifying parse works
    expect(result.success).toBe(true);
    expect(result.data?.va2).toBe(1);
  });

  test("falls back to runtime parsing when extraction hints are disabled", () => {
    const input = "{ val: ['a'], va2: '42' }";
    const result = parseLLMOutput(input, Schema, {
      repair: true,
      extraction: {
        secondPassMin: 0,
        secondPassCap: 0,
        secondPassMultiplier: 0,
      },
    });

    expect(result.success).toBe(true);
    expect(result.candidates[0]?.parseHint).toBeUndefined();
    expect(result.repaired).toBeString();
    expect(result.data).toEqual({
      val: ["a"],
      va2: 42,
    });
  });

  test("surfaces parse-stage errors when hints are disabled and repair is off", () => {
    const result = parseLLMOutput("{ val: ['a'], va2: '42' }", Schema, {
      repair: false,
      extraction: {
        secondPassMin: 0,
        secondPassCap: 0,
        secondPassMultiplier: 0,
      },
    });

    expect(result.success).toBe(false);
    expect(result.errors.some((error) => error.stage === "parse")).toBe(true);
    expect(result.diagnostics.some((diagnostic) => diagnostic.stage === "parse")).toBe(true);
  });

  test("propagates repair-hint failures with repair stage diagnostics", () => {
    const result = parseLLMOutput('{"a":1,,"b":2}', Schema, {
      repair: true,
    });

    expect(result.success).toBe(false);
    expect(result.errors.some((error) => error.stage === "repair")).toBe(true);
    expect(
      result.diagnostics.some(
        (diagnostic) =>
          diagnostic.stage === "repair" &&
          diagnostic.usedRepair &&
          !diagnostic.parseSuccess &&
          diagnostic.validationSuccess === false,
      ),
    ).toBe(true);
  });

  test("keeps the least-bad candidate to expose the most actionable zod issues", () => {
    const input = [
      '{"va2":"oops","val":[1]}',
      '{"foo":"bar"}',
    ].join(" ");

    const result = parseLLMOutput(input, Schema, {
      repair: true,
      maxCandidates: 5,
    });

    expect(result.success).toBe(false);
    expect(result.zodIssues).toHaveLength(1);
    expect(result.candidate?.content).toContain('"foo":"bar"');
    const selected = result.diagnostics.find((diagnostic) => diagnostic.selected);
    expect(selected?.candidateId).toBe(result.candidate?.id);
  });

  test("emits trace events for extraction and final failure", () => {
    const traces: string[] = [];
    parseLLMOutput("plain text without json", Schema, {
      repair: true,
      onTrace(event) {
        traces.push(`${event.level}:${event.stage}:${event.message}`);
      },
    });

    expect(traces.some((line) => line.includes("info:extract:Extracted"))).toBe(true);
    expect(traces.some((line) => line.includes("error:result:No candidate could be validated."))).toBe(
      true,
    );
  });
});

describe("formatZodIssues", () => {
  test("formats nested paths", () => {
    const result = formatZodIssues([
      {
        path: ["user", "name"],
        message: "Required",
        code: "invalid_type",
        expected: "string",
        received: "undefined"
      } as z.core.$ZodIssue,
      {
        path: [],
        message: "Unrecognized key",
        code: "unrecognized_keys",
        keys: []
      } as unknown as z.core.$ZodIssue,
    ]);

    expect(result).toBe("user.name: Required\n<root>: Unrecognized key");
  });

  test("returns a default message for an empty array", () => {
    expect(formatZodIssues([])).toBe("Validation failed without details.");
  });
});
