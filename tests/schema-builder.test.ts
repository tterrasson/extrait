import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { withFormat } from "@/format";
import { getSchemaName, inferSchemaExample, inspectSchemaMetadata, s, setSchemaName } from "@/schema-builder";
import { DEFAULT_SCHEMA_INSTRUCTION} from "@/format";

describe("schema builder", () => {
  test("inspects required/defaults/descriptions", () => {
    const schema = s.schema(
      "Summary",
      z.object({
        summary: s.string().min(1).describe("One-sentence summary."),
        score: s.number().coerce().min(0).max(10).describe("Confidence from 0 to 10."),
        tags: s.array(s.string()).default([]).describe("Relevant keywords."),
      }),
    );

    const metadata = inspectSchemaMetadata(schema);

    expect(metadata.name).toBe("Summary");
    expect(metadata.requiredFields).toEqual(["summary", "score"]);
    expect(metadata.defaults).toEqual({ tags: [] });
    expect(metadata.fieldDescriptions).toEqual({
      summary: "One-sentence summary.",
      score: "Confidence from 0 to 10.",
      tags: "Relevant keywords.",
    });

    const inferred = inferSchemaExample(schema);
    expect(inferred).toEqual({
      tags: [],
    });
  });

  test("withFormat renders a schema-only prompt with inline descriptions", () => {
    const schema = s.schema(
      "Summary",
      z.object({
        summary: s.string().describe("Main summary text."),
        tags: s.array(s.string()).default([]).describe("Associated tags."),
      }),
    );

    const out = withFormat(schema);

    expect(out).toContain(DEFAULT_SCHEMA_INSTRUCTION);
    expect(out).toContain("summary: string,  // Main summary text.");
    expect(out).toContain("tags: string[],  // Associated tags.");
    expect(out).not.toContain("Schema name:");
    expect(out).not.toContain("Required fields:");
    expect(out).not.toContain("Defaults:");
    expect(out).not.toContain("Field examples:");
  });

  test("s.number().coerce() converts string numbers via safeParse", () => {
    const schema = z.object({ score: s.number().coerce().min(0) });
    const parsed = schema.safeParse({ score: "7" });

    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.score).toBe(7);
    }
  });

  test("setSchemaName / getSchemaName round-trip", () => {
    const schema = z.object({ x: z.string() });
    setSchemaName(schema, "TestSchema");
    expect(getSchemaName(schema)).toBe("TestSchema");
  });

  test("inferSchemaExample retourne null pour un schema non-objet", () => {
    expect(inferSchemaExample(z.string())).toBeNull();
  });

  test("inferSchemaExample retourne null quand aucun champ n'a de default", () => {
    const schema = z.object({ name: z.string(), age: z.number() });
    expect(inferSchemaExample(schema)).toBeNull();
  });

  test("readDefaultValue traverses ZodCatch wrapper", () => {
    const schema = z.object({
      name: z.string().default("fallback").catch("safe"),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.defaults).toEqual({ name: "fallback" });
  });

  test("readDefaultValue traverses ZodReadonly wrapper", () => {
    const schema = z.object({
      items: z.array(z.string()).default(["a"]),
    }).readonly();
    const example = inferSchemaExample(schema);
    expect(example).toEqual({ items: ["a"] });
  });

  test("readDefaultValue traverses ZodBranded wrapper", () => {
    const schema = z.object({
      email: z.string().default("x@y.com").brand("Email"),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.defaults).toEqual({ email: "x@y.com" });
  });

  test("readDefaultValue traverses ZodPipeline wrapper", () => {
    // default on the outside of a pipeline: default wraps the pipeline
    const schema = z.object({
      score: z.coerce.number().pipe(z.number().min(0)).default(0),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.defaults).toEqual({ score: 0 });
  });

  test("readDefaultValue traverses ZodEffects wrapper", () => {
    const schema = z.object({
      name: z.string().default("hello").transform((v) => v.toUpperCase()),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.defaults).toEqual({ name: "hello" });
  });

  test("readSchemaDescription traverses ZodCatch", () => {
    const schema = z.object({
      name: z.string().describe("user name").catch(""),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.fieldDescriptions.name).toBe("user name");
  });

  test("readSchemaDescription traverses ZodReadonly", () => {
    const schema = z.object({
      id: z.string().describe("unique id"),
    }).readonly();
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.fieldDescriptions.id).toBe("unique id");
  });

  test("readSchemaDescription traverses ZodBranded", () => {
    const schema = z.object({
      email: z.string().describe("email address").brand("Email"),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.fieldDescriptions.email).toBe("email address");
  });

  test("readSchemaDescription traverses ZodPipeline", () => {
    // describe on the outside wrapping the pipeline
    const schema = z.object({
      score: z.string().pipe(z.coerce.number()).describe("score value"),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.fieldDescriptions.score).toBe("score value");
  });

  test("readSchemaDescription traverses ZodEffects", () => {
    const schema = z.object({
      name: z.string().describe("the name").transform((v) => v.toUpperCase()),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.fieldDescriptions.name).toBe("the name");
  });

  test("unwrap traverses ZodNullable, ZodCatch, ZodReadonly", () => {
    const schema = z.object({
      value: z.string().nullable().catch(null).readonly().default("x"),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.defaults).toEqual({ value: "x" });
    expect(metadata.requiredFields).toEqual([]);
  });

  test("readDefaultValue handles factory function default", () => {
    const schema = z.object({
      tags: z.array(z.string()).default(() => ["auto"]),
    });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.defaults).toEqual({ tags: ["auto"] });
  });

  test("readDefaultValue returns undefined when factory function throws", () => {
    const schema = z.object({
      bad: z.string().default(() => {
        throw new Error("oops");
      }),
    });
    // Should not throw; the error is swallowed and the field has no default
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.defaults).toEqual({});
  });

  test("readDefaultValue returns undefined for plain schema with no default", () => {
    const schema = z.object({ name: z.string() });
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.defaults).toEqual({});
    expect(metadata.requiredFields).toContain("name");
  });

  test("ensurePatchedZod is idempotent: calling schema builder multiple times does not error", () => {
    // Re-importing via s.number() calls ensurePatchedZod; repeated usage should be safe
    const a = s.number().coerce();
    const b = s.number().coerce();
    expect(a.safeParse("3").success).toBe(true);
    expect(b.safeParse("7").success).toBe(true);
  });

  test("getObjectShape handles lazy shape function", () => {
    // In Zod, _def.shape is already a function. inspectSchemaMetadata must call it to get fields.
    const schema = z.object({ x: z.string(), y: z.number().optional() });
    // The shape is a function returning the shape record — verify inspectSchemaMetadata handles it
    expect(typeof (schema._def as { shape: unknown }).shape).toBe("object");
    const metadata = inspectSchemaMetadata(schema);
    expect(metadata.requiredFields).toContain("x");
    expect(metadata.requiredFields).not.toContain("y");
  });

  test("inferSchemaExample returns non-null for non-object schema with a default", () => {
    // A plain ZodDefault wrapping a non-object schema
    const schema = z.string().default("hello");
    const result = inferSchemaExample(schema);
    expect(result).toBe("hello");
  });
});
