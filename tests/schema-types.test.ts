import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { formatZodSchemaLikeTypeScript } from "@/schema";

describe("formatZodSchemaLikeTypeScript – additional types", () => {
  test("formats bigint", () => {
    expect(formatZodSchemaLikeTypeScript(z.bigint())).toBe("bigint");
  });

  test("formats date", () => {
    expect(formatZodSchemaLikeTypeScript(z.date())).toBe("Date");
  });

  test("formats undefined", () => {
    expect(formatZodSchemaLikeTypeScript(z.undefined())).toBe("undefined");
  });

  test("formats null", () => {
    expect(formatZodSchemaLikeTypeScript(z.null())).toBe("null");
  });

  test("formats any", () => {
    expect(formatZodSchemaLikeTypeScript(z.any())).toBe("any");
  });

  test("formats unknown", () => {
    expect(formatZodSchemaLikeTypeScript(z.unknown())).toBe("unknown");
  });

  test("formats never", () => {
    expect(formatZodSchemaLikeTypeScript(z.never())).toBe("never");
  });

  test("formats void", () => {
    expect(formatZodSchemaLikeTypeScript(z.void())).toBe("void");
  });
});

describe("formatZodSchemaLikeTypeScript – native enums", () => {
  test("formats numeric native enum", () => {
    enum Status {
      Active = 1,
      Inactive = 2,
    }
    const result = formatZodSchemaLikeTypeScript(z.nativeEnum(Status));
    expect(result).toContain("1");
    expect(result).toContain("2");
  });

  test("formats string native enum", () => {
    enum Color {
      Red = "red",
      Blue = "blue",
    }
    const result = formatZodSchemaLikeTypeScript(z.nativeEnum(Color));
    expect(result).toBe('"red" | "blue"');
  });
});

describe("formatZodSchemaLikeTypeScript – tuples", () => {
  test("formats a tuple", () => {
    const result = formatZodSchemaLikeTypeScript(z.tuple([z.string(), z.number()]));
    expect(result).toBe("[string, number]");
  });

  test("formats an empty tuple", () => {
    const result = formatZodSchemaLikeTypeScript(z.tuple([]));
    expect(result).toBe("[]");
  });
});

describe("formatZodSchemaLikeTypeScript – discriminated unions", () => {
  test("formats a discriminated union", () => {
    const schema = z.discriminatedUnion("type", [
      z.object({ type: z.literal("a"), value: z.string() }),
      z.object({ type: z.literal("b"), count: z.number() }),
    ]);
    const result = formatZodSchemaLikeTypeScript(schema);
    expect(result).toContain('"a"');
    expect(result).toContain('"b"');
    expect(result).toContain("|");
  });
});

describe("formatZodSchemaLikeTypeScript – intersection", () => {
  test("formats an intersection", () => {
    const schema = z.intersection(
      z.object({ a: z.string() }),
      z.object({ b: z.number() }),
    );
    const result = formatZodSchemaLikeTypeScript(schema);
    expect(result).toContain("&");
    expect(result).toContain("a: string");
    expect(result).toContain("b: number");
  });
});

describe("formatZodSchemaLikeTypeScript – map and set", () => {
  test("formats a map", () => {
    const result = formatZodSchemaLikeTypeScript(z.map(z.string(), z.number()));
    expect(result).toBe("Map<string, number>");
  });

  test("formats a set", () => {
    const result = formatZodSchemaLikeTypeScript(z.set(z.string()));
    expect(result).toBe("Set<string>");
  });
});

describe("formatZodSchemaLikeTypeScript – wrapper types", () => {
  test("unwraps branded type", () => {
    const result = formatZodSchemaLikeTypeScript(z.string().brand("Email"));
    expect(result).toBe("string");
  });

  test("unwraps catch wrapper", () => {
    const result = formatZodSchemaLikeTypeScript(z.string().catch("default"));
    expect(result).toBe("string");
  });

  test("unwraps readonly on object", () => {
    const result = formatZodSchemaLikeTypeScript(
      z.object({ x: z.string() }).readonly(),
    );
    expect(result).toContain("x: string");
  });

  test("unwraps pipeline to output type", () => {
    const result = formatZodSchemaLikeTypeScript(
      z.string().pipe(z.coerce.number()),
    );
    expect(result).toBe("number");
  });
});

describe("formatZodSchemaLikeTypeScript – descriptions through wrappers", () => {
  test("reads description through branded type", () => {
    const schema = z.object({
      email: z.string().describe("user email").brand("Email"),
    });
    const result = formatZodSchemaLikeTypeScript(schema);
    expect(result).toContain("// user email");
  });

  test("reads description through catch wrapper", () => {
    const schema = z.object({
      name: z.string().describe("user name").catch(""),
    });
    const result = formatZodSchemaLikeTypeScript(schema);
    expect(result).toContain("// user name");
  });

  test("reads description through readonly wrapper", () => {
    const schema = z.object({
      id: z.string().describe("unique id"),
    }).readonly();
    const result = formatZodSchemaLikeTypeScript(schema);
    expect(result).toContain("// unique id");
  });

  test("reads description on pipeline wrapper", () => {
    const schema = z.object({
      score: z.string().pipe(z.coerce.number()).describe("numeric score"),
    });
    const result = formatZodSchemaLikeTypeScript(schema);
    expect(result).toContain("// numeric score");
  });
});

describe("formatZodSchemaLikeTypeScript – edge cases", () => {
  test("formats empty object as {}", () => {
    expect(formatZodSchemaLikeTypeScript(z.object({}))).toBe("{}");
  });

  test("formats lazy schema as unknown", () => {
    const lazySchema: z.ZodTypeAny = z.lazy(() => z.string());
    expect(formatZodSchemaLikeTypeScript(lazySchema)).toBe("unknown");
  });

  test("formats default value as optional in object", () => {
    const schema = z.object({
      name: z.string(),
      role: z.string().default("user"),
    });
    const result = formatZodSchemaLikeTypeScript(schema);
    expect(result).toContain("name: string");
    expect(result).toContain("role: string");
  });

  test("formats effects/transform (unwraps through ZodEffects)", () => {
    const schema = z.string().transform((v) => v.toUpperCase());
    const result = formatZodSchemaLikeTypeScript(schema);
    expect(result).toBe("string");
  });
});
