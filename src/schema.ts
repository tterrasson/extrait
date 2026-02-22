import type { z } from "zod";

const RE_SIMPLE_IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const RE_WHITESPACE = /\s+/g;

type ZodLike = z.ZodTypeAny & {
  _def?: {
    typeName?: string;
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
    const typeName = current?._def?.typeName;
    if (!typeName) {
      break;
    }

    if (typeName === "ZodOptional") {
      optional = true;
      current = (current._def?.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodDefault") {
      optional = true;
      current = (current._def?.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodNullable") {
      nullable = true;
      current = (current._def?.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodEffects") {
      current = (current._def?.schema as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodBranded") {
      current = (current._def?.type as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodCatch") {
      current = (current._def?.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodReadonly") {
      current = (current._def?.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodPipeline") {
      current = (current._def?.out as ZodLike) ?? current;
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
  const typeName = schema?._def?.typeName;

  switch (typeName) {
    case "ZodString":
      return "string";
    case "ZodNumber":
      return isIntegerNumber(schema) ? "int" : "number";
    case "ZodBoolean":
      return "boolean";
    case "ZodBigInt":
      return "bigint";
    case "ZodDate":
      return "Date";
    case "ZodUndefined":
      return "undefined";
    case "ZodNull":
      return "null";
    case "ZodAny":
      return "any";
    case "ZodUnknown":
      return "unknown";
    case "ZodNever":
      return "never";
    case "ZodVoid":
      return "void";
    case "ZodLiteral": {
      const value = schema._def?.value;
      return JSON.stringify(value);
    }
    case "ZodEnum": {
      const values = (schema._def?.values as string[] | undefined) ?? [];
      return values.map((value) => JSON.stringify(value)).join(" | ") || "string";
    }
    case "ZodNativeEnum": {
      const values = Object.values((schema._def?.values as Record<string, unknown> | undefined) ?? {});
      const unique = [...new Set(values.filter((value) => typeof value !== "string" || Number.isNaN(Number(value))))];
      return unique.map((value) => JSON.stringify(value)).join(" | ") || "string";
    }
    case "ZodArray": {
      const inner = formatType((schema._def?.type as ZodLike) ?? schema, depth, seen);
      return requiresParentheses(inner) ? `(${inner})[]` : `${inner}[]`;
    }
    case "ZodTuple": {
      const items = ((schema._def?.items as ZodLike[] | undefined) ?? []).map((item) =>
        formatType(item, depth, seen),
      );
      return `[${items.join(", ")}]`;
    }
    case "ZodUnion": {
      const options = ((schema._def?.options as ZodLike[] | undefined) ?? []).map((option) =>
        formatType(option, depth, seen),
      );
      return options.join(" | ") || "unknown";
    }
    case "ZodDiscriminatedUnion": {
      const options = Array.from(
        (((schema._def?.options as Map<unknown, ZodLike> | undefined) ?? new Map()).values()),
      ).map((option) => formatType(option, depth, seen));
      return options.join(" | ") || "unknown";
    }
    case "ZodIntersection": {
      const left = formatType((schema._def?.left as ZodLike) ?? schema, depth, seen);
      const right = formatType((schema._def?.right as ZodLike) ?? schema, depth, seen);
      return `${left} & ${right}`;
    }
    case "ZodRecord": {
      const keyType = formatType((schema._def?.keyType as ZodLike) ?? schema, depth, seen);
      const valueType = formatType((schema._def?.valueType as ZodLike) ?? schema, depth, seen);
      return `Record<${keyType}, ${valueType}>`;
    }
    case "ZodMap": {
      const keyType = formatType((schema._def?.keyType as ZodLike) ?? schema, depth, seen);
      const valueType = formatType((schema._def?.valueType as ZodLike) ?? schema, depth, seen);
      return `Map<${keyType}, ${valueType}>`;
    }
    case "ZodSet": {
      const valueType = formatType((schema._def?.valueType as ZodLike) ?? schema, depth, seen);
      return `Set<${valueType}>`;
    }
    case "ZodObject":
      return formatObject(schema, depth, seen);
    case "ZodLazy":
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
  const checks = (schema._def?.checks as Array<{ kind?: string }> | undefined) ?? [];
  return checks.some((check) => check.kind === "int");
}

function readSchemaDescription(schema: ZodLike): string | undefined {
  let current = schema;

  while (current?._def?.typeName) {
    const direct = current._def.description;
    if (typeof direct === "string" && direct.trim().length > 0) {
      return sanitizeDescription(direct);
    }

    const fallback = (current as { description?: unknown }).description;
    if (typeof fallback === "string" && fallback.trim().length > 0) {
      return sanitizeDescription(fallback);
    }

    const typeName = current._def.typeName;
    if (typeName === "ZodOptional" || typeName === "ZodDefault" || typeName === "ZodNullable") {
      current = (current._def.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodEffects") {
      current = (current._def.schema as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodBranded") {
      current = (current._def.type as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodCatch" || typeName === "ZodReadonly") {
      current = (current._def.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "ZodPipeline") {
      current = (current._def.out as ZodLike) ?? current;
      continue;
    }

    break;
  }

  return undefined;
}

function sanitizeDescription(value: string): string {
  return value.replace(RE_WHITESPACE, " ").trim();
}
