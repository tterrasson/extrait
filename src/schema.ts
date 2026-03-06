import type { z } from "zod";

const RE_SIMPLE_IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const RE_WHITESPACE = /\s+/g;

type ZodLike = z.ZodTypeAny & {
  _def?: {
    type?: string;
    [key: string]: unknown;
  };
};

interface UnwrappedSchema {
  schema: ZodLike;
  optional: boolean;
  nullable: boolean;
}

export function formatZodSchemaLikeTypeScript(schema: z.ZodTypeAny): string {
  return formatType(schema as ZodLike, 0, new WeakSet<ZodLike>());
}

function formatType(schema: ZodLike, depth: number, seen: WeakSet<ZodLike>): string {
  const unwrapped = unwrap(schema);
  let result = formatCore(unwrapped.schema, depth, seen);

  if (unwrapped.nullable) {
    result = `${result} | null`;
  }

  return result;
}

function unwrap(schema: ZodLike): UnwrappedSchema {
  let current = schema;
  let optional = false;
  let nullable = false;

  while (true) {
    const typeName = current?._def?.type;
    if (!typeName) {
      break;
    }

    if (typeName === "optional") {
      optional = true;
      current = (current._def?.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "default") {
      optional = true;
      current = (current._def?.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "nullable") {
      nullable = true;
      current = (current._def?.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "pipe") {
      // transform: _def.out.type === "transform" → follow input; real pipeline → follow output
      const outType = (current._def?.out as ZodLike)?._def?.type;
      if (outType === "transform") {
        current = (current._def?.in as ZodLike) ?? current;
      } else {
        current = (current._def?.out as ZodLike) ?? current;
      }
      continue;
    }

    if (typeName === "catch" || typeName === "readonly") {
      current = (current._def?.innerType as ZodLike) ?? current;
      continue;
    }

    break;
  }

  return {
    schema: current,
    optional,
    nullable,
  };
}

function formatCore(schema: ZodLike, depth: number, seen: WeakSet<ZodLike>): string {
  if (seen.has(schema)) {
    return "unknown";
  }

  seen.add(schema);
  const typeName = schema?._def?.type;

  switch (typeName) {
    case "string":
      return "string";
    case "number":
      return isIntegerNumber(schema) ? "int" : "number";
    case "boolean":
      return "boolean";
    case "bigint":
      return "bigint";
    case "date":
      return "Date";
    case "undefined":
      return "undefined";
    case "null":
      return "null";
    case "any":
      return "any";
    case "unknown":
      return "unknown";
    case "never":
      return "never";
    case "void":
      return "void";
    case "literal": {
      const value = (schema._def?.values as unknown[])?.[0];
      return JSON.stringify(value);
    }
    case "enum": {
      // covers both z.enum() and z.nativeEnum() — both use _def.entries in Zod 4
      const entries = schema._def?.entries as Record<string, unknown> | undefined;
      const values = Object.values(entries ?? {});
      const unique = [...new Set(values.filter((v) => typeof v !== "string" || Number.isNaN(Number(v))))];
      return unique.map((v) => JSON.stringify(v)).join(" | ") || "string";
    }
    case "array": {
      const inner = formatType((schema._def?.element as ZodLike) ?? schema, depth, seen);
      return requiresParentheses(inner) ? `(${inner})[]` : `${inner}[]`;
    }
    case "tuple": {
      const items = ((schema._def?.items as ZodLike[] | undefined) ?? []).map((item) =>
        formatType(item, depth, seen),
      );
      return `[${items.join(", ")}]`;
    }
    case "union": {
      // covers both z.union() and z.discriminatedUnion() — both use _def.options array in Zod 4
      const options = ((schema._def?.options as ZodLike[] | undefined) ?? []).map((option) =>
        formatType(option, depth, seen),
      );
      return options.join(" | ") || "unknown";
    }
    case "intersection": {
      const left = formatType((schema._def?.left as ZodLike) ?? schema, depth, seen);
      const right = formatType((schema._def?.right as ZodLike) ?? schema, depth, seen);
      return `${left} & ${right}`;
    }
    case "record": {
      const keyType = formatType((schema._def?.keyType as ZodLike) ?? schema, depth, seen);
      const valueType = formatType((schema._def?.valueType as ZodLike) ?? schema, depth, seen);
      return `Record<${keyType}, ${valueType}>`;
    }
    case "map": {
      const keyType = formatType((schema._def?.keyType as ZodLike) ?? schema, depth, seen);
      const valueType = formatType((schema._def?.valueType as ZodLike) ?? schema, depth, seen);
      return `Map<${keyType}, ${valueType}>`;
    }
    case "set": {
      const valueType = formatType((schema._def?.valueType as ZodLike) ?? schema, depth, seen);
      return `Set<${valueType}>`;
    }
    case "object":
      return formatObject(schema, depth, seen);
    case "lazy":
      return "unknown";
    default:
      return "unknown";
  }
}

function formatObject(schema: ZodLike, depth: number, seen: WeakSet<ZodLike>): string {
  const indent = "  ".repeat(depth);
  const innerIndent = "  ".repeat(depth + 1);
  const rawShape = schema._def?.shape;
  const shape =
    typeof rawShape === "function"
      ? (rawShape as () => Record<string, ZodLike>)()
      : ((rawShape as Record<string, ZodLike> | undefined) ?? {});
  const entries = Object.entries(shape);

  if (entries.length === 0) {
    return "{}";
  }

  const lines = entries.map(([key, value]) => {
    const unwrapped = unwrap(value);
    const type = formatType(unwrapped.schema, depth + 1, seen);
    const typeWithNull = unwrapped.nullable ? `${type} | null` : type;
    const description = readSchemaDescription(value);
    const descriptionComment = description ? `  // ${description}` : "";
    return `${innerIndent}${formatKey(key)}: ${typeWithNull},${descriptionComment}`;
  });

  return `{\n${lines.join("\n")}\n${indent}}`;
}

function formatKey(key: string): string {
  return RE_SIMPLE_IDENTIFIER.test(key) ? key : JSON.stringify(key);
}

function requiresParentheses(typeText: string): boolean {
  return typeText.includes(" | ") || typeText.includes(" & ");
}

function isIntegerNumber(schema: ZodLike): boolean {
  const checks = (schema._def?.checks as Array<{ isInt?: boolean }> | undefined) ?? [];
  return checks.some((check) => check.isInt === true);
}

function readSchemaDescription(schema: ZodLike): string | undefined {
  let current = schema;

  while (current?._def?.type) {
    const desc = (current as { description?: unknown }).description;
    if (typeof desc === "string" && desc.trim().length > 0) {
      return sanitizeDescription(desc);
    }

    const typeName = current._def.type;
    if (typeName === "optional" || typeName === "default" || typeName === "nullable") {
      current = (current._def.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "pipe") {
      current = (current._def.in as ZodLike) ?? current;
      continue;
    }

    if (typeName === "catch" || typeName === "readonly") {
      current = (current._def.innerType as ZodLike) ?? current;
      continue;
    }

    break;
  }

  return undefined;
}

function sanitizeDescription(value: string): string {
  return value.replace(RE_WHITESPACE, " ").trim();
}
