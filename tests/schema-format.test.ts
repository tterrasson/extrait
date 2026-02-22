import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { formatZodSchemaLikeTypeScript } from "../src/schema";

describe("formatZodSchemaLikeTypeScript", () => {
  test("formats primitive types", () => {
    expect(formatZodSchemaLikeTypeScript(z.string())).toBe("string");
    expect(formatZodSchemaLikeTypeScript(z.number())).toBe("number");
    expect(formatZodSchemaLikeTypeScript(z.boolean())).toBe("boolean");
  });

  test("formats number().int() as 'int'", () => {
    expect(formatZodSchemaLikeTypeScript(z.number().int())).toBe("int");
  });

  test("formats enums", () => {
    const result = formatZodSchemaLikeTypeScript(z.enum(["red", "green", "blue"]));
    expect(result).toBe('"red" | "green" | "blue"');
  });

  test("formats literals", () => {
    expect(formatZodSchemaLikeTypeScript(z.literal("fixed"))).toBe('"fixed"');
    expect(formatZodSchemaLikeTypeScript(z.literal(42))).toBe("42");
  });

  test("formats an array of objects", () => {
    const schema = z.array(z.object({ name: z.string() }));
    const result = formatZodSchemaLikeTypeScript(schema);

    expect(result).toContain("name: string,");
    expect(result).toEndWith("[]");
  });

  test("formats optional and nullable fields in an object", () => {
    const schema = z.object({
      required: z.string(),
      opt: z.string().optional(),
      nul: z.string().nullable(),
      both: z.string().optional().nullable(),
    });

    const result = formatZodSchemaLikeTypeScript(schema);

    expect(result).toContain("required: string,");
    expect(result).toContain("opt: string,");
    expect(result).toContain("nul: string | null,");
    expect(result).toContain("both: string | null,");
  });

  test("formats unions and adds parentheses in an array", () => {
    const union = z.union([z.string(), z.number()]);
    expect(formatZodSchemaLikeTypeScript(union)).toBe("string | number");

    const arrayOfUnion = z.array(union);
    expect(formatZodSchemaLikeTypeScript(arrayOfUnion)).toBe("(string | number)[]");
  });

  test("formats records", () => {
    const schema = z.record(z.string(), z.number());
    expect(formatZodSchemaLikeTypeScript(schema)).toBe("Record<string, number>");
  });

  test("includes descriptions as inline comments", () => {
    const schema = z.object({
      name: z.string().describe("The user name"),
      age: z.number().describe("Age in years"),
    });

    const result = formatZodSchemaLikeTypeScript(schema);

    expect(result).toContain("name: string,  // The user name");
    expect(result).toContain("age: number,  // Age in years");
  });
});
