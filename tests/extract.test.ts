import { describe, expect, test } from "bun:test";
import { extractJsonCandidates } from "@/extract";
import { sanitizeThink } from "@/think";
import { extractFirstMarkdownCode, extractMarkdownCodeBlocks } from "@/markdown";

describe("extractJsonCandidates", () => {
  test("prioritizes fenced markdown blocks", () => {
    const input = [
      "Some text before",
      "```json",
      "{\"a\": 1}",
      "```",
      "Then another object {\"a\":2}",
    ].join("\n");

    const candidates = extractJsonCandidates(input, { maxCandidates: 5 });

    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates[0]?.source).toBe("fenced");
    expect(candidates[0]?.content).toContain('"a": 1');
  });

  test("ignores braces inside strings", () => {
    const input = `noise {"text":"bonjour {monde}","value":1} suffix`;

    const candidates = extractJsonCandidates(input);

    expect(candidates[0]?.content).toBe('{"text":"bonjour {monde}","value":1}');
  });

  test("extracts JSON when prose contains apostrophes before the object", () => {
    const input = `Voici l'analyse du message: {"sentiment":"POSITIVE","confidence":0.8}`;

    const candidates = extractJsonCandidates(input);

    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates[0]?.content).toBe('{"sentiment":"POSITIVE","confidence":0.8}');
  });

  test("acceptArrays false ignores top-level arrays", () => {
    const input = 'before [1,2,3] {"ok":true} after';
    const candidates = extractJsonCandidates(input, { acceptArrays: false });

    expect(candidates.every((c) => !c.content.startsWith("["))).toBe(true);
    expect(candidates.some((c) => c.content.includes('"ok"'))).toBe(true);
  });

  test("falls back to raw when no JSON is found", () => {
    const candidates = extractJsonCandidates("just plain text");

    expect(candidates.length).toBe(1);
    expect(candidates[0]?.source).toBe("raw");
  });

  test("maxCandidates limits the number of results", () => {
    const input = '{"a":1} {"b":2} {"c":3} {"d":4}';
    const candidates = extractJsonCandidates(input, { maxCandidates: 2 });

    expect(candidates.length).toBeLessThanOrEqual(2);
  });

  test("does not sanitize <think> blocks by itself", () => {
    const input = [
      "<think>",
      '{"hidden":true}',
      "</think>",
      '{"visible":true}',
    ].join("\n");

    const candidates = extractJsonCandidates(input);

    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.some((candidate) => candidate.content.includes('"hidden"'))).toBe(true);
  });

  test("works with sanitized think input", () => {
    const input = [
      "<think>",
      '{"hidden":true}',
      "</think>",
      '{"visible":true}',
    ].join("\n");
    const sanitized = sanitizeThink(input);
    const candidates = extractJsonCandidates(sanitized.visibleText);

    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.some((candidate) => candidate.content.includes('"hidden"'))).toBe(false);
    expect(candidates.some((candidate) => candidate.content.includes('"visible"'))).toBe(true);
  });
  test("deduplicates identical candidates", () => {
    const input = '{"a":1} {"a":1} {"b":2}';
    const candidates = extractJsonCandidates(input, { maxCandidates: 10 });

    const contents = candidates.map((c) => c.content);
    const unique = [...new Set(contents)];
    expect(contents.length).toBe(unique.length);
  });

  test("handles escaped quotes in scan mode", () => {
    const input = 'text {"msg":"hello \\"world\\""} end';
    const candidates = extractJsonCandidates(input);

    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates[0]?.content).toContain('"msg"');
    expect(candidates[0]?.content).toContain('\\"world\\"');
  });

  test("object candidates score higher than array candidates", () => {
    const input = '[1,2,3] {"obj":true}';
    const candidates = extractJsonCandidates(input, { maxCandidates: 10 });

    const objCandidate = candidates.find((c) => c.content.startsWith("{"));
    const arrCandidate = candidates.find((c) => c.content.startsWith("["));

    expect(objCandidate).toBeDefined();
    expect(arrCandidate).toBeDefined();
    expect(objCandidate!.score).toBeGreaterThan(arrCandidate!.score);
  });

  test("non-json code block gets lower score bonus than json block", () => {
    const input = [
      "```python",
      '{"a": 1}',
      "```",
      "```json",
      '{"b": 2}',
      "```",
    ].join("\n");

    const candidates = extractJsonCandidates(input, { maxCandidates: 5 });

    const jsonCandidate = candidates.find((c) => c.content.includes('"b"'));
    const pyCandidate = candidates.find((c) => c.content.includes('"a"'));

    expect(jsonCandidate).toBeDefined();
    expect(pyCandidate).toBeDefined();
    expect(jsonCandidate!.score).toBeGreaterThan(pyCandidate!.score);
  });

  test("two-pass attaches parse hints only to top-ranked candidates", () => {
    const input = Array.from({ length: 14 }, (_, index) => `{"idx":${index}}`).join(" ");
    const candidates = extractJsonCandidates(input, { maxCandidates: 10 });

    const hinted = candidates.filter((candidate) => candidate.parseHint);

    expect(candidates.length).toBe(10);
    expect(hinted.length).toBeGreaterThan(0);
    expect(hinted.length).toBeLessThanOrEqual(8);
    expect(hinted.every((candidate) => candidate.parseHint?.success)).toBe(true);
  });

  test("heuristics can disable second-pass parse hints", () => {
    const input = Array.from({ length: 14 }, (_, index) => `{"idx":${index}}`).join(" ");
    const candidates = extractJsonCandidates(input, {
      maxCandidates: 10,
      heuristics: {
        secondPassMin: 0,
        secondPassCap: 0,
        secondPassMultiplier: 0,
      },
    });

    expect(candidates.every((candidate) => !candidate.parseHint)).toBe(true);
  });

  test("heuristics can cap parse hint size", () => {
    const input = '{"ok":true}';
    const candidates = extractJsonCandidates(input, {
      maxCandidates: 1,
      heuristics: {
        hintMaxLength: 5,
      },
    });

    expect(candidates).toHaveLength(1);
    expect(candidates[0]?.parseHint).toBeUndefined();
  });

  test("deduplicates before first-pass slicing to preserve candidate diversity", () => {
    const duplicated = Array.from({ length: 40 }, () => '{"dup":1}').join(" ");
    const input = `${duplicated} {"target":2}`;
    const candidates = extractJsonCandidates(input, { maxCandidates: 2 });

    expect(candidates.length).toBe(2);
    expect(candidates.some((candidate) => candidate.content === '{"dup":1}')).toBe(true);
    expect(candidates.some((candidate) => candidate.content === '{"target":2}')).toBe(true);
  });

  test("fenced block with empty content is skipped", () => {
    const input = "```json\n\n```\n{\"ok\":true}";
    const candidates = extractJsonCandidates(input, { maxCandidates: 5 });

    // empty fenced block not included; scan finds {"ok":true}
    expect(candidates.every((c) => c.content !== "")).toBe(true);
    expect(candidates.some((c) => c.content.includes('"ok"'))).toBe(true);
  });

  test("fenced block with non-JSON content is skipped", () => {
    const input = ["```text", "just plain text", "```", '{"ok":true}'].join("\n");
    const candidates = extractJsonCandidates(input, { maxCandidates: 5 });

    expect(candidates.every((c) => c.source !== "fenced")).toBe(true);
  });

  test("javascript/typescript language gets a score bonus", () => {
    const input = [
      "```javascript",
      '{"a": 1}',
      "```",
      "```json",
      '{"b": 2}',
      "```",
    ].join("\n");
    const candidates = extractJsonCandidates(input, { maxCandidates: 5 });

    const jsCandidate = candidates.find((c) => c.content.includes('"a"'));
    const jsonCandidate = candidates.find((c) => c.content.includes('"b"'));

    expect(jsCandidate).toBeDefined();
    expect(jsonCandidate).toBeDefined();
    // json bonus (140) > js bonus (40), so json should score higher
    expect(jsonCandidate!.score).toBeGreaterThan(jsCandidate!.score);
    // both should have a higher score than an unknown language block
    expect(jsCandidate!.score).toBeGreaterThan(0);
  });

  test("good and bad candidates: fallback shape is appended after filtered candidates", () => {
    // This mix triggers the branch where filtered.length > 0 AND a fallback exists
    const input = ['{"va2":1,"val":["ok"]}', '["just","array"]'].join(" ");
    const candidates = extractJsonCandidates(input, { acceptArrays: true, maxCandidates: 5 });

    // We should get both shapes
    const hasObject = candidates.some((c) => c.content.startsWith("{"));
    const hasArray = candidates.some((c) => c.content.startsWith("["));
    expect(hasObject).toBe(true);
    expect(hasArray).toBe(true);
  });
});

describe("extractMarkdownCodeBlocks", () => {
  test("extracts markdown code by language", () => {
    const input = [
      "```ts",
      "export const x = 1;",
      "```",
      "```json",
      "{\"ok\": true}",
      "```",
    ].join("\n");

    const ts = extractFirstMarkdownCode(input, { language: "ts" });
    const all = extractMarkdownCodeBlocks(input);

    expect(ts?.code).toBe("export const x = 1;");
    expect(all.length).toBe(2);
  });
});
