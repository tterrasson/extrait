import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { prompt } from "@/prompt";

describe("prompt", () => {
  test("dedents and removes leading/trailing empty lines", () => {
    const out = prompt`
      Ligne 1
      Ligne 2
    `;

    expect(out).toBe("Ligne 1\nLigne 2");
  });

  test("serializes interpolated objects as JSON", () => {
    const out = prompt`
      Return this object:
      ${{
        ok: true,
        count: 2,
      }}
    `;

    expect(out).toContain('"ok": true');
    expect(out).toContain('"count": 2');
  });

  test("supports fluent system/user messages with template dedent", () => {
    const built = prompt()
      .system`
        System line
      `
      .user`
        User line 1
        User line 2
      `
      .build();

    expect(built.messages).toEqual([
      { role: "system", content: "System line" },
      { role: "user", content: "User line 1\nUser line 2" },
    ]);
  });

  test("supports classic dynamic string fallback", () => {
    const dynamicString = "Prompt dynamique";
    const built = prompt().user(dynamicString).build();

    expect(built.messages).toEqual([{ role: "user", content: dynamicString }]);
  });

  test("supports assistant messages and preserves turn order", () => {
    const built = prompt()
      .system`System line`
      .user`Hello`
      .assistant`Hi there`
      .user`Need help`
      .build();

    expect(built.messages).toEqual([
      { role: "system", content: "System line" },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
      { role: "user", content: "Need help" },
    ]);
  });

  test("handles lines with mixed indentation", () => {
    const result = prompt`
      Line 1
        Line 2
      Line 3
    `;

    expect(result).toBe("Line 1\n  Line 2\nLine 3");
  });

  test("handles empty lines at beginning and end", () => {
    const result = prompt`

      Content here

    `;

    expect(result).toBe("Content here");
  });

  test("preserves empty lines in the middle", () => {
    const result = prompt`
      Line 1

      Line 2
    `;

    expect(result).toBe("Line 1\n\nLine 2");
  });

  test("handles indentation with tabs", () => {
    const result = prompt`
		Tab line 1
			Tab line 2
    `;

    expect(result).toContain("Tab line");
  });

  test("normalizes CRLF line endings", () => {
    const result = prompt`Line 1\r\nLine 2\r\nLine 3`;

    expect(result).toBe("Line 1\nLine 2\nLine 3");
  });

  test("handles value interpolation", () => {
    const name = "Alice";
    const age = 30;
    const result = prompt`
      Name: ${name}
      Age: ${age}
    `;

    expect(result).toBe("Name: Alice\nAge: 30");
  });

  test("keeps template dedent when interpolation contains leading newlines", () => {
    const noisyValue = "\n\nStructured result:";
    const result = prompt`
      Contexte: Exemple

      Génère un résumé à propos de: """Bun"""

      console.log("${noisyValue}");
    `;

    expect(result).toBe(
      'Contexte: Exemple\n\nGénère un résumé à propos de: """Bun"""\n\nconsole.log("\n\nStructured result:");',
    );
  });

  test("handles object interpolation", () => {
    const obj = { key: "value" };
    const result = prompt`Object: ${obj}`;

    expect(result).toContain('"key"');
    expect(result).toContain('"value"');
  });

  test("handles null and undefined values", () => {
    const result = prompt`
      Null: ${null}
      Undefined: ${undefined}
    `;

    expect(result).toBe("Null: \nUndefined: ");
  });

  test("accepts LLMMessageContent array in user()", () => {
    const content = [
      { type: "text" as const, text: "Describe this image." },
      { type: "image_url" as const, image_url: { url: "data:image/png;base64,abc123" } },
    ];
    const built = prompt().system`You are an assistant.`.user(content).build();

    expect(built.messages).toEqual([
      { role: "system", content: "You are an assistant." },
      { role: "user", content },
    ]);
  });

  test("accepts LLMMessageContent array in assistant()", () => {
    const content = [
      { type: "text" as const, text: "Here is my analysis." },
      { type: "image_url" as const, image_url: { url: "data:image/png;base64,xyz" } },
    ];
    const built = prompt().assistant(content).build();

    expect(built.messages).toEqual([{ role: "assistant", content }]);
  });
});
