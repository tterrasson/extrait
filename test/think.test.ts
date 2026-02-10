import { describe, expect, test } from "bun:test";
import { sanitizeThink } from "../src/think";

describe("think blocks", () => {
  test("extracts think content", () => {
    const input = [
      "before",
      "<think>",
      "internal reasoning",
      "</think>",
      "after",
    ].join("\n");

    const result = sanitizeThink(input);

    expect(result.thinkBlocks.length).toBe(1);
    expect(result.thinkBlocks[0]?.content).toBe("internal reasoning");
    expect(result.diagnostics.unterminatedCount).toBe(0);
  });

  test("masks think content while preserving line breaks", () => {
    const input = "a\n<think>\nreason\n</think>\nb";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks.length).toBe(1);
    expect(result.visibleText.split("\n").length).toBe(input.split("\n").length);
    expect(result.visibleText).not.toContain("reason");
    expect(result.visibleText).toContain("a");
    expect(result.visibleText).toContain("b");
  });

  test("supports nested think blocks", () => {
    const input = "x <think>a <think>b</think> c</think> y";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.visibleText).not.toContain("</think>");
    expect(result.visibleText).toContain("x");
    expect(result.visibleText).toContain("y");
    expect(result.diagnostics.nestedCount).toBe(1);
  });

  test("fails closed on unterminated think blocks", () => {
    const input = 'prefix <think>{"hidden":true}';
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.visibleText).not.toContain('"hidden"');
    expect(result.diagnostics.unterminatedCount).toBe(1);
  });
});
