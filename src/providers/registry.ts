import type { HTTPHeaders, LLMAdapter } from "../types";
import {
  createOpenAICompatibleAdapter,
  type OpenAICompatibleAdapterOptions,
} from "./openai-compatible";
import {
  createOpenAICompatibleLegacyAdapter,
  type OpenAICompatibleLegacyAdapterOptions,
} from "./openai-compatible-legacy";
import {
  createAnthropicCompatibleAdapter,
  type AnthropicCompatibleAdapterOptions,
} from "./anthropic-compatible";

export type BuiltinProviderKind =
  | "openai-compatible"
  | "openai-compatible-legacy"
  | "anthropic-compatible";

export interface ProviderFactory<TOptions = unknown> {
  (options: TOptions): LLMAdapter;
}

export interface ProviderRegistry {
  register<TOptions>(kind: string, factory: ProviderFactory<TOptions>): void;
  create<TOptions>(kind: string, options: TOptions): LLMAdapter;
  has(kind: string): boolean;
  list(): string[];
}

/** Advanced knobs only — the endpoint and credentials live at the top level of the config. */
export interface ProviderTransportConfig {
  path?: string;
  /** Embedding endpoint override (openai-compatible providers only). */
  embeddingPath?: string;
  headers?: HTTPHeaders;
  defaultBody?: Record<string, unknown>;
  version?: string;
  /** Default `max_tokens` (anthropic-compatible only). */
  defaultMaxTokens?: number;
  /** Default cap on MCP tool-call rounds per request. */
  defaultMaxToolRounds?: number;
  fetcher?: typeof fetch;
}

interface ModelAdapterConfigBase {
  model: string;
  apiKey?: string;
  transport?: ProviderTransportConfig;
  options?: Record<string, unknown>;
}

/**
 * Built-in providers have no default endpoint: `baseURL` is mandatory, so an
 * unconfigured client never ships credentials to a vendor host. Custom providers
 * registered through {@link ProviderRegistry.register} opt out by naming their
 * kind explicitly, e.g. `ModelAdapterConfig<"my-provider">`.
 */
export type ModelAdapterConfig<TProvider extends string = BuiltinProviderKind> = ModelAdapterConfigBase & {
  provider: TProvider;
} & (TProvider extends BuiltinProviderKind ? { baseURL: string } : { baseURL?: string });

/** Structural view used internally, once the generic requirement has been checked. */
type ResolvedModelAdapterConfig = ModelAdapterConfigBase & {
  provider: string;
  baseURL?: string;
};

class InMemoryProviderRegistry implements ProviderRegistry {
  private readonly factories = new Map<string, ProviderFactory<unknown>>();

  register<TOptions>(kind: string, factory: ProviderFactory<TOptions>): void {
    this.factories.set(kind, factory as ProviderFactory<unknown>);
  }

  create<TOptions>(kind: string, options: TOptions): LLMAdapter {
    const factory = this.factories.get(kind);
    if (!factory) {
      throw new Error(`Unknown provider kind: ${kind}`);
    }

    return factory(options);
  }

  has(kind: string): boolean {
    return this.factories.has(kind);
  }

  list(): string[] {
    return [...this.factories.keys()].sort();
  }
}

export function createProviderRegistry(): ProviderRegistry {
  return new InMemoryProviderRegistry();
}

export function registerBuiltinProviders(registry: ProviderRegistry): ProviderRegistry {
  registry.register("openai-compatible", createOpenAICompatibleAdapter);
  registry.register("openai-compatible-legacy", createOpenAICompatibleLegacyAdapter);
  registry.register("anthropic-compatible", createAnthropicCompatibleAdapter);
  return registry;
}

export function createDefaultProviderRegistry(): ProviderRegistry {
  return registerBuiltinProviders(createProviderRegistry());
}

export function createModelAdapter<TProvider extends string>(
  config: ModelAdapterConfig<TProvider>,
  registry: ProviderRegistry = createDefaultProviderRegistry(),
): LLMAdapter {
  const resolved = config as ResolvedModelAdapterConfig;
  const providerOptions = buildProviderOptions(resolved);
  return registry.create(resolved.provider, providerOptions);
}

function requireBaseURL(provider: string, baseURL: string | undefined): string {
  if (!baseURL) {
    throw new Error(
      `Provider "${provider}" requires an explicit baseURL. ` +
        `There is no default endpoint, so credentials are never sent to an unintended host.`,
    );
  }

  return baseURL;
}

function buildProviderOptions(config: ResolvedModelAdapterConfig): unknown {
  const transport: ProviderTransportConfig = config.transport ?? {};

  if (config.provider === "openai-compatible") {
    const options: OpenAICompatibleAdapterOptions = {
      model: config.model,
      baseURL: requireBaseURL(config.provider, config.baseURL),
      apiKey: config.apiKey,
      path: transport.path,
      embeddingPath: transport.embeddingPath,
      headers: transport.headers,
      defaultBody: transport.defaultBody,
      defaultMaxToolRounds: transport.defaultMaxToolRounds,
      fetcher: transport.fetcher,
    };

    return options;
  }

  if (config.provider === "openai-compatible-legacy") {
    const options: OpenAICompatibleLegacyAdapterOptions = {
      model: config.model,
      baseURL: requireBaseURL(config.provider, config.baseURL),
      apiKey: config.apiKey,
      path: transport.path,
      embeddingPath: transport.embeddingPath,
      headers: transport.headers,
      defaultBody: transport.defaultBody,
      defaultMaxToolRounds: transport.defaultMaxToolRounds,
      fetcher: transport.fetcher,
    };

    return options;
  }

  if (config.provider === "anthropic-compatible") {
    const options: AnthropicCompatibleAdapterOptions = {
      model: config.model,
      baseURL: requireBaseURL(config.provider, config.baseURL),
      apiKey: config.apiKey,
      path: transport.path,
      headers: transport.headers,
      version: transport.version,
      defaultBody: transport.defaultBody,
      defaultMaxTokens: transport.defaultMaxTokens,
      defaultMaxToolRounds: transport.defaultMaxToolRounds,
      fetcher: transport.fetcher,
    };

    return options;
  }

  return {
    model: config.model,
    baseURL: config.baseURL,
    apiKey: config.apiKey,
    ...transport,
    ...config.options,
  };
}
