import { z } from "zod";

type ZodLike = z.ZodTypeAny & {
  def?: {
    type?: string;
    [key: string]: unknown;
  };
};

const NAME_SYMBOL = Symbol.for("extrait.schema.name");

type MetaCarrier = z.ZodTypeAny & {
  [NAME_SYMBOL]?: string;
};

// Extend zod types with custom methods
declare module "zod" {
  interface ZodNumber {
    coerce(): z.ZodNumber;
  }
}

let didPatchZod = false;

function ensurePatchedZod(): void {
  if (didPatchZod) {
    return;
  }

  const zodNumberPrototype = z.ZodNumber.prototype as {
    coerce?: (this: z.ZodNumber) => z.ZodNumber;
  };

  if (!zodNumberPrototype.coerce) {
    // Defined as non-enumerable so the patch never shows up when other consumers
    // of the shared Zod instance enumerate a schema's own/inherited keys.
    Object.defineProperty(zodNumberPrototype, "coerce", {
      value: function coerceNumber(): z.ZodNumber {
        return z.coerce.number() as z.ZodNumber;
      },
      enumerable: false,
      writable: true,
      configurable: true,
    });
  }

  didPatchZod = true;
}

ensurePatchedZod();

export interface SchemaMetadataSummary {
  name?: string;
  description?: string;
  requiredFields: string[];
  defaults: Record<string, unknown>;
  fieldDescriptions: Record<string, string>;
}

export const s = {
  schema<TSchema extends z.ZodTypeAny>(name: string, schema: TSchema): TSchema {
    return setSchemaName(schema, name);
  },

  string(): z.ZodString {
    return z.string();
  },

  number(): z.ZodNumber {
    return z.number();
  },

  boolean(): z.ZodBoolean {
    return z.boolean();
  },

  array<TSchema extends z.ZodTypeAny>(schema: TSchema): z.ZodArray<TSchema> {
    return z.array(schema);
  },

  object<TShape extends z.ZodRawShape>(shape: TShape): z.ZodObject<TShape> {
    return z.object(shape);
  },
};

export function setSchemaName<TSchema extends z.ZodTypeAny>(schema: TSchema, name: string): TSchema {
  (schema as MetaCarrier)[NAME_SYMBOL] = name;
  return schema;
}

export function getSchemaName(schema: z.ZodTypeAny): string | undefined {
  return (schema as MetaCarrier)[NAME_SYMBOL];
}

export function inspectSchemaMetadata(schema: z.ZodTypeAny): SchemaMetadataSummary {
  const requiredFields: string[] = [];
  const defaults: Record<string, unknown> = {};
  const fieldDescriptions: Record<string, string> = {};

  const objectShape = getObjectShape(schema);
  if (objectShape) {
    for (const [fieldName, fieldSchema] of Object.entries(objectShape)) {
      const { optional } = unwrap(fieldSchema);
      if (!optional) {
        requiredFields.push(fieldName);
      }

      const defaultValue = readDefaultValue(fieldSchema);
      if (defaultValue !== undefined) {
        defaults[fieldName] = defaultValue;
      }

      const description = readSchemaDescription(fieldSchema);
      if (description) {
        fieldDescriptions[fieldName] = description;
      }
    }
  }

  return {
    name: getSchemaName(schema),
    description: readSchemaDescription(schema),
    requiredFields,
    defaults,
    fieldDescriptions,
  };
}

export function inferSchemaExample(schema: z.ZodTypeAny): unknown | null {
  const objectShape = getObjectShape(schema);
  if (!objectShape) {
    const fallback = readDefaultValue(schema);
    if (fallback !== undefined) {
      return fallback;
    }
    return null;
  }

  const out: Record<string, unknown> = {};

  for (const [fieldName, fieldSchema] of Object.entries(objectShape)) {
    const defaultValue = readDefaultValue(fieldSchema);
    if (defaultValue !== undefined) {
      out[fieldName] = defaultValue;
    }
  }

  return Object.keys(out).length > 0 ? out : null;
}

function getObjectShape(schema: z.ZodTypeAny): Record<string, z.ZodTypeAny> | null {
  const unwrapped = unwrap(schema).schema;
  const typeName = unwrapped.def?.type;

  if (typeName !== "object") {
    return null;
  }

  const rawShape = unwrapped.def?.shape;
  if (typeof rawShape === "function") {
    return (rawShape as () => Record<string, z.ZodTypeAny>)();
  }

  return (rawShape as Record<string, z.ZodTypeAny> | undefined) ?? null;
}

function readDefaultValue(schema: z.ZodTypeAny): unknown {
  let current = schema as ZodLike;

  while (current?.def?.type) {
    const typeName = current.def.type;

    if (typeName === "default") {
      try {
        const raw = current.def.defaultValue;
        if (typeof raw === "function") {
          return (raw as () => unknown)();
        }
        return raw;
      } catch {
        return undefined;
      }
    }

    if (
      typeName === "optional" ||
      typeName === "nullable" ||
      typeName === "catch" ||
      typeName === "readonly"
    ) {
      current = (current.def.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "pipe") {
      current = (current.def.in as ZodLike) ?? current;
      continue;
    }

    return undefined;
  }

  return undefined;
}

function readSchemaDescription(schema: z.ZodTypeAny): string | undefined {
  let current = schema as ZodLike;

  while (current?.def?.type) {
    const desc = (current as { description?: unknown }).description;
    if (typeof desc === "string" && desc.trim().length > 0) {
      return desc.trim();
    }

    const typeName = current.def.type;

    if (typeName === "optional" || typeName === "default" || typeName === "nullable") {
      current = (current.def.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "catch" || typeName === "readonly") {
      current = (current.def.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "pipe") {
      current = (current.def.in as ZodLike) ?? current;
      continue;
    }

    break;
  }

  return undefined;
}

function unwrap(schema: z.ZodTypeAny): { schema: ZodLike; optional: boolean } {
  let current = schema as ZodLike;
  let optional = false;

  while (current?.def?.type) {
    const typeName = current.def.type;

    if (typeName === "optional" || typeName === "default") {
      optional = true;
      current = (current.def.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "nullable" || typeName === "catch" || typeName === "readonly") {
      current = (current.def.innerType as ZodLike) ?? current;
      continue;
    }

    if (typeName === "pipe") {
      current = (current.def.in as ZodLike) ?? current;
      continue;
    }

    break;
  }

  return {
    schema: current,
    optional,
  };
}
