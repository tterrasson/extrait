import { describe, expect, test } from "bun:test";
import { z } from "zod";
import {
  DEFAULT_SCHEMA_INSTRUCTION,
  formatPrompt,
  resolveSchemaInstruction,
  withFormat,
} from "@/format";

describe("formatPrompt", () => {
  test("combines the schema format and the task", () => {
    const schema = z.object({ title: z.string() });
    const result = formatPrompt(schema, "Summarize the article");

    expect(result).toContain(DEFAULT_SCHEMA_INSTRUCTION);
    expect(result).toContain("title: string,");
    expect(result).toContain("Summarize the article");
  });

  test("returns only the schema when the task is empty", () => {
    const schema = z.object({ x: z.number() });
    const onlySchema = withFormat(schema);
    const result = formatPrompt(schema, "   ");

    expect(result).toBe(onlySchema);
  });
});

describe("withFormat", () => {
  test("accepts a custom schemaInstruction", () => {
    const result = withFormat(z.object({ x: z.string() }), {
      schemaInstruction: "Output JSON matching this type:",
    });

    expect(result).toContain("Output JSON matching this type:");
    expect(result).not.toContain(DEFAULT_SCHEMA_INSTRUCTION);
  });

  test("falls back to the default instruction if the custom instruction is empty", () => {
    const result = withFormat(z.object({ x: z.string() }), {
      schemaInstruction: "   ",
    });

    expect(result).toContain(DEFAULT_SCHEMA_INSTRUCTION);
  });
});

describe("resolveSchemaInstruction", () => {
  test("returns the exported default instruction when input is empty", () => {
    expect(resolveSchemaInstruction("   ")).toBe(DEFAULT_SCHEMA_INSTRUCTION);
  });
});
