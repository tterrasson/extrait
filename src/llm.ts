import type { z } from "zod";
import {
  createModelAdapter,
  createDefaultProviderRegistry,
  type ModelAdapterConfig,
  type ProviderRegistry
} from "./providers/registry";
import { structured } from "./structured";
import type {
  LLMAdapter,
  StructuredCallOptions,
  StructuredPromptBuilder,
  StructuredResult
} from "./types";

export interface CreateLLMOptions extends ModelAdapterConfig {
  defaults?: StructuredCallOptions<z.ZodTypeAny>;
}

export interface LLMClient {
  adapter: LLMAdapter;
  provider?: string;
  model?: string;
  structured<TSchema extends z.ZodTypeAny>(
    schema: TSchema,
    prompt: StructuredPromptBuilder,
    options?: StructuredCallOptions<TSchema>,
  ): Promise<StructuredResult<z.infer<TSchema>>>;
}

export function createLLM(
  config: CreateLLMOptions,
  registry: ProviderRegistry = createDefaultProviderRegistry(),
): LLMClient {
  const adapter = createModelAdapter(config, registry);
  const defaults = config.defaults;

  return {
    adapter,
    provider: adapter.provider,
    model: adapter.model,

    async structured<TSchema extends z.ZodTypeAny>(
      schema: TSchema,
      prompt: StructuredPromptBuilder,
      options?: StructuredCallOptions<TSchema>,
    ): Promise<StructuredResult<z.infer<TSchema>>> {
      const merged = mergeStructuredOptions(defaults, options);
      return structured(adapter, schema, prompt, merged);
    },
  };
}

function mergeStructuredOptions<TSchema extends z.ZodTypeAny>(
  defaults: StructuredCallOptions<z.ZodTypeAny> | undefined,
  overrides: StructuredCallOptions<TSchema> | undefined,
): StructuredCallOptions<TSchema> {
  if (!defaults && !overrides) {
    return {};
  }

  return {
    ...(defaults as StructuredCallOptions<TSchema> | undefined),
    ...overrides,
    parse: {
      ...(defaults?.parse ?? {}),
      ...(overrides?.parse ?? {}),
    },
    request: {
      ...(defaults?.request ?? {}),
      ...(overrides?.request ?? {}),
    },
    stream: mergeObjectLike(defaults?.stream, overrides?.stream),
    selfHeal: mergeObjectLike(defaults?.selfHeal, overrides?.selfHeal),
    debug: mergeObjectLike(defaults?.debug, overrides?.debug),
  };
}

function mergeObjectLike<TValue>(defaults: TValue | undefined, overrides: TValue | undefined): TValue | undefined {
  if (overrides === undefined) {
    return defaults;
  }

  if (defaults === undefined) {
    return overrides;
  }

  if (!isPlainObject(defaults) || !isPlainObject(overrides)) {
    return overrides;
  }

  return {
    ...(defaults as Record<string, unknown>),
    ...(overrides as Record<string, unknown>),
  } as TValue;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
