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

  test("ignores non-think HTML-like tags", () => {
    const input = "text <div>content</div> after";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(0);
    expect(result.visibleText).toBe(input);
    expect(result.diagnostics.unterminatedCount).toBe(0);
  });

  test("ignores tags that start with 'think' but are longer identifiers", () => {
    const input = "text <thinking>content</thinking> after";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(0);
    expect(result.visibleText).toBe(input);
  });

  test("handles think tag with attributes", () => {
    const input = '<think data="value">hidden content</think>visible';
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("hidden content");
    expect(result.visibleText).not.toContain("hidden");
    expect(result.visibleText).toContain("visible");
  });

  test("handles closing tag with whitespace after </", () => {
    const input = "<think>content</ think>after";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("content");
    expect(result.visibleText).not.toContain("content");
    expect(result.visibleText).toContain("after");
  });

  test("handles closing tag with whitespace between think and >", () => {
    const input = "<think>content</think  >after";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("content");
    expect(result.visibleText).not.toContain("content");
    expect(result.visibleText).toContain("after");
  });

  test("handles think tag with quoted attribute containing >", () => {
    const input = '<think attr="a > b">inner</think>outer';
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("inner");
    expect(result.visibleText).toContain("outer");
    expect(result.visibleText).not.toContain("inner");
  });

  test("multiple non-think tags before a real think block", () => {
    const input = "a < b and c > d <think>hidden</think> visible";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("hidden");
    expect(result.visibleText).toContain("visible");
    expect(result.visibleText).not.toContain("hidden");
  });

  test("closing tag without > is treated as unterminated", () => {
    // </think sans > final : le closing tag est malformé, le bloc reste ouvert
    const input = "<think>content</think";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.diagnostics.unterminatedCount).toBe(1);
    expect(result.visibleText).not.toContain("content");
  });

  test("opening tag without > consumes until end of input", () => {
    // <think sans > : findTagEnd retourne -1, le bloc s'étend jusqu'à la fin
    const input = "before <think hidden content";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.diagnostics.unterminatedCount).toBe(1);
    expect(result.visibleText).not.toContain("hidden content");
    expect(result.visibleText).toContain("before");
  });

  test("multiple sequential think blocks are all extracted", () => {
    const input = "<think>first</think> middle <think>second</think> end";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(2);
    expect(result.thinkBlocks[0]?.content).toBe("first");
    expect(result.thinkBlocks[1]?.content).toBe("second");
    expect(result.visibleText).toContain("middle");
    expect(result.visibleText).toContain("end");
    expect(result.visibleText).not.toContain("first");
    expect(result.visibleText).not.toContain("second");
  });

  test("closing tag outside any open block is silently ignored", () => {
    // </think> sans <think> ouvert : depth=0 donc on continue sans erreur
    const input = "before </think> after";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(0);
    expect(result.diagnostics.unterminatedCount).toBe(0);
    expect(result.visibleText).toBe("before </think> after");
  });

  test("hiddenChars counts non-linebreak characters in masked blocks", () => {
    const input = "<think>abc\ndef</think>";
    const result = sanitizeThink(input);

    // "<think>abc\ndef</think>" : non-linebreak chars = "<think>abc" (10) + "def</think>" (11) = 21
    expect(result.diagnostics.hiddenChars).toBeGreaterThan(0);
    expect(result.diagnostics.hiddenChars).toBe(
      input.split("").filter((c) => c !== "\n" && c !== "\r").length,
    );
  });

  test("handles uppercase and mixed-case think tags", () => {
    const input = "<THINK>hidden</THINK>visible";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("hidden");
    expect(result.visibleText).not.toContain("hidden");
    expect(result.visibleText).toContain("visible");
  });

  test("mixed case open and close tags", () => {
    const input = "<Think>hidden</THINK>visible";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("hidden");
    expect(result.visibleText).toContain("visible");
  });

  test("empty input returns empty result", () => {
    const result = sanitizeThink("");

    expect(result.thinkBlocks).toHaveLength(0);
    expect(result.visibleText).toBe("");
    expect(result.diagnostics.unterminatedCount).toBe(0);
    expect(result.diagnostics.nestedCount).toBe(0);
    expect(result.diagnostics.hiddenChars).toBe(0);
  });

  test("think tag at very end of string with no content", () => {
    const input = "before<think>";
    const result = sanitizeThink(input);

    // Le bloc est ouvert mais jamais fermé -> unterminated
    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("");
    expect(result.diagnostics.unterminatedCount).toBe(1);
    expect(result.visibleText).not.toContain("<think>");
    expect(result.visibleText).toContain("before");
  });

  test("self-closing think tag <think/> consumes rest of input as unterminated", () => {
    // <think/> est traité comme ouverture (findTagEnd trouve le '>'), sans closing tag
    // -> le bloc reste unterminated et avale tout ce qui suit
    const input = "before <think/> after";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.diagnostics.unterminatedCount).toBe(1);
    expect(result.visibleText).not.toContain("after");
    expect(result.visibleText).toContain("before");
  });

  test("think tag with single-quoted attribute", () => {
    const input = "<think attr='value'>hidden</think>visible";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("hidden");
    expect(result.visibleText).toContain("visible");
  });

  test("input with only whitespace", () => {
    const result = sanitizeThink("   \n\t  ");

    expect(result.thinkBlocks).toHaveLength(0);
    expect(result.visibleText).toBe("   \n\t  ");
    expect(result.diagnostics.unterminatedCount).toBe(0);
  });

  test("closing tag with attributes is rejected (not a valid close tag)", () => {
    // </think attr="x"> n'est pas un closing tag valide XML — le parser
    // doit le rejeter car après "think" il y a un espace mais pas de '>'
    // ce qui le rend malformé, le bloc reste unterminated
    const input = '<think>content</think attr="x">after';
    const result = sanitizeThink(input);

    // Le closing tag malformé n'est pas reconnu -> bloc unterminated
    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.diagnostics.unterminatedCount).toBe(1);
  });

  test("stray < before a think tag does not break parsing", () => {
    const input = "a < b <think>hidden</think> visible";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.thinkBlocks[0]?.content).toBe("hidden");
    expect(result.visibleText).toContain("a < b");
    expect(result.visibleText).toContain("visible");
  });

  test("deeply nested blocks count correctly", () => {
    const input = "<think>a<think>b<think>c</think></think></think>";
    const result = sanitizeThink(input);

    expect(result.thinkBlocks).toHaveLength(1);
    expect(result.diagnostics.nestedCount).toBe(2);
    expect(result.thinkBlocks[0]?.content).toBe("a<think>b<think>c</think></think>");
  });
});
