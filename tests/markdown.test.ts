import { describe, expect, test } from "bun:test";
import { extractFirstMarkdownCode, extractMarkdownCodeBlocks } from "@/markdown";

describe("markdown code fences", () => {
  test("extracts a code block without final newline", () => {
    const input = "```json\n{\"key\": \"value\"}```";
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.language).toBe("json");
    expect(blocks[0]?.code).toBe('{"key": "value"}');
  });

  test("extracts a code block with final newline", () => {
    const input = "```json\n{\"key\": \"value\"}\n```";
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.language).toBe("json");
    expect(blocks[0]?.code).toBe('{"key": "value"}');
  });

  test("extracts multiple code blocks", () => {
    const input = `
Here is some code:
\`\`\`javascript
const x = 1;
\`\`\`

And more:
\`\`\`python
y = 2
\`\`\``;

    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(2);
    expect(blocks[0]?.language).toBe("javascript");
    expect(blocks[0]?.code).toBe("const x = 1;");
    expect(blocks[1]?.language).toBe("python");
    expect(blocks[1]?.code).toBe("y = 2");
  });

  test("extracts a block without language specifier", () => {
    const input = "```\ncode here\n```";
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.language).toBe(null);
    expect(blocks[0]?.code).toBe("code here");
  });

  test("filters by specific language", () => {
    const input = `
\`\`\`typescript
const x: number = 1;
\`\`\`
\`\`\`javascript
const y = 2;
\`\`\``;

    const blocks = extractMarkdownCodeBlocks(input, { language: "typescript" });

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.language).toBe("typescript");
  });

  test("extractFirstMarkdownCode returns the first block", () => {
    const input = `
\`\`\`json
{"first": true}
\`\`\`
\`\`\`json
{"second": true}
\`\`\``;

    const block = extractFirstMarkdownCode(input);

    expect(block).not.toBeNull();
    expect(block?.code).toBe('{"first": true}');
  });

  test("returns null if no block found", () => {
    const input = "No code here";
    const block = extractFirstMarkdownCode(input);

    expect(block).toBeNull();
  });

  test("supports fences longer than three backticks", () => {
    const input = [
      "````ts",
      "const marker = \"```\";",
      "````",
    ].join("\n");
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.language).toBe("ts");
    expect(blocks[0]?.code).toBe('const marker = "```";');
  });

  test("supports tilde fenced blocks", () => {
    const input = ["~~~json", '{"ok": true}', "~~~"].join("\n");
    const blocks = extractMarkdownCodeBlocks(input, { language: "json" });

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.code).toBe('{"ok": true}');
  });

  test("extracts only the language token from info string", () => {
    const input = ["```json title=test", '{"ok": true}', "```"].join("\n");
    const block = extractFirstMarkdownCode(input, { language: "json" });

    expect(block).not.toBeNull();
    expect(block?.language).toBe("json");
  });

  test("language filter non-matching block is excluded from results", () => {
    const input = ["```python", "print(1)", "```", "```json", '{"ok":true}', "```"].join("\n");
    const blocks = extractMarkdownCodeBlocks(input, { language: "json" });

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.language).toBe("json");
  });

  test("unclosed fence is not extracted", () => {
    const input = "```json\n{\"ok\": true}\nno closing fence";
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(0);
  });

  test("closing fence with extra non-whitespace is not a valid closer", () => {
    const input = ["```json", '{"ok": true}', "```extra", "```"].join("\n");
    const blocks = extractMarkdownCodeBlocks(input);

    // "```extra" is not a valid closing fence, "```" is
    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.code).toContain('"ok"');
  });

  test("closing fence with fewer backticks than opening is not a valid closer", () => {
    const input = ["````json", '{"ok": true}', "```", "````"].join("\n");
    const blocks = extractMarkdownCodeBlocks(input);

    // 3-backtick close doesn't match 4-backtick open; 4-backtick close does
    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.code).toContain('"ok"');
  });

  test("mismatched fence marker (tilde vs backtick) does not close", () => {
    const input = ["```json", '{"ok": true}', "~~~", "```"].join("\n");
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.code).toContain('"ok"');
  });

  test("language specifier with dot prefix is normalized", () => {
    const input = ["```.json", '{"ok": true}', "```"].join("\n");
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.language).toBe("json");
  });

  test("inline closing fence on same line as content", () => {
    const input = "```json\n{\"ok\": true}```";
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.code).toBe('{"ok": true}');
  });

  test("closing fence with trailing spaces is valid", () => {
    const input = "```json\n{\"ok\": true}\n```   \n";
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.code).toBe('{"ok": true}');
  });

  test("handles Windows CRLF line endings", () => {
    const input = "```json\r\n{\"ok\": true}\r\n```\r\n";
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.language).toBe("json");
    expect(blocks[0]?.code).toBe('{"ok": true}');
  });

  test("opening fence with 4+ spaces indent is not a valid fence", () => {
    const input = "    ```json\n{\"ok\": true}\n```";
    const blocks = extractMarkdownCodeBlocks(input);

    expect(blocks).toHaveLength(0);
  });
});
