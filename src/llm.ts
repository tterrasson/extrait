import type { z } from "zod";
import {
  createModelAdapter,
  createDefaultProviderRegistry,
  type BuiltinProviderKind,
  type ModelAdapterConfig,
  type ProviderRegistry
} from "./providers/registry";
import { generate } from "./generate";
import { structured } from "./structured";
import type {
  EmbeddingRequest,
  EmbeddingResult,
  GenerateCallOptions,
  GenerateOptions,
  GenerateResult,
  LLMAdapter,
  LLMRequest,
  ParseLLMOutputOptions,
  StructuredDebugOptions,
  StructuredCallOptions,
  StructuredMode,
  StructuredSelfHealInput,
  StructuredPromptBuilder,
  StructuredResult,
  StructuredTimeoutOptions,
  StructuredTraceEvent,
  GenerateTraceEvent,
} from "./types";

interface LLMClientDefaults {
  mode?: StructuredMode;
  outdent?: boolean;
  parse?: ParseLLMOutputOptions;
  selfHeal?: StructuredSelfHealInput;
  stream?: StructuredCallOptions<z.ZodTypeAny>["stream"] | GenerateCallOptions["stream"];
  debug?: boolean | StructuredDebugOptions;
  observe?: ((event: StructuredTraceEvent | GenerateTraceEvent) => void) | undefined;
  systemPrompt?: string;
  request?: Omit<LLMRequest, "prompt" | "systemPrompt" | "messages">;
  schemaInstruction?: string;
  timeout?: StructuredTimeoutOptions;
}

export type CreateLLMOptions<TProvider extends string = BuiltinProviderKind> = ModelAdapterConfig<TProvider> & {
  defaults?: LLMClientDefaults;
};

export interface LLMClient {
  adapter: LLMAdapter;
  provider?: string;
  model?: string;
  structured<TSchema extends z.ZodTypeAny>(
    schema: TSchema,
    prompt: StructuredPromptBuilder,
    options?: StructuredCallOptions<TSchema>,
  ): Promise<StructuredResult<z.infer<TSchema>>>;
  generate(
    prompt: StructuredPromptBuilder,
    options?: GenerateCallOptions,
  ): Promise<GenerateResult>;
  generate(options: GenerateOptions): Promise<GenerateResult>;
  embed(input: string | string[], options?: Omit<EmbeddingRequest, "input">): Promise<EmbeddingResult>;
}

export function createLLM<TProvider extends string>(
  config: CreateLLMOptions<TProvider>,
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

    async generate(
      promptOrOptions: StructuredPromptBuilder | GenerateOptions,
      options?: GenerateCallOptions,
    ): Promise<GenerateResult> {
      if (isGenerateOptions(promptOrOptions)) {
        const merged = {
          ...mergeGenerateOptions(defaults, promptOrOptions),
          prompt: promptOrOptions.prompt,
        };
        return generate(adapter, merged);
      }

      const merged = mergeGenerateOptions(defaults, options);
      return generate(adapter, promptOrOptions, merged);
    },

    async embed(input: string | string[], options: Omit<EmbeddingRequest, "input"> = {}): Promise<EmbeddingResult> {
      if (!adapter.embed) {
        throw new Error(`Provider "${adapter.provider ?? "unknown"}" does not support embeddings.`);
      }
      return adapter.embed({ ...options, input });
    },
  };
}

function mergeStructuredOptions<TSchema extends z.ZodTypeAny>(
  defaults: LLMClientDefaults | undefined,
  overrides: StructuredCallOptions<TSchema> | undefined,
): StructuredCallOptions<TSchema> {
  if (!defaults && !overrides) {
    return {};
  }

  return {
    ...(defaults as StructuredCallOptions<TSchema> | undefined),
    ...overrides,
    parse: {
      ...defaults?.parse,
      ...overrides?.parse,
    },
    request: {
      ...defaults?.request,
      ...overrides?.request,
    },
    stream: mergeObjectLike(
      defaults?.stream as StructuredCallOptions<TSchema>["stream"],
      overrides?.stream,
    ),
    selfHeal: mergeObjectLike(defaults?.selfHeal, overrides?.selfHeal),
    debug: mergeObjectLike(defaults?.debug, overrides?.debug),
    timeout: mergeObjectLike(defaults?.timeout, overrides?.timeout),
  };
}

function mergeGenerateOptions(
  defaults: LLMClientDefaults | undefined,
  overrides: GenerateCallOptions | GenerateOptions | undefined,
): GenerateCallOptions {
  if (!defaults && !overrides) {
    return {};
  }

  return {
    outdent: overrides?.outdent ?? defaults?.outdent,
    systemPrompt: overrides?.systemPrompt ?? defaults?.systemPrompt,
    request: {
      ...defaults?.request,
      ...overrides?.request,
    },
    stream: mergeObjectLike(defaults?.stream as GenerateCallOptions["stream"], overrides?.stream),
    debug: mergeObjectLike(defaults?.debug, overrides?.debug),
    timeout: mergeObjectLike(defaults?.timeout, overrides?.timeout),
    observe: overrides?.observe ?? (defaults?.observe as GenerateCallOptions["observe"] | undefined),
  };
}

function isGenerateOptions(value: StructuredPromptBuilder | GenerateOptions): value is GenerateOptions {
  return typeof value === "object" && value !== null && "prompt" in value;
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
