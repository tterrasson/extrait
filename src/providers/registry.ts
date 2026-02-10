import type { HTTPHeaders, LLMAdapter } from "../types";
import {
  createOpenAICompatibleAdapter,
  type OpenAICompatibleAdapterOptions,
} from "./openai-compatible";
import {
  createAnthropicCompatibleAdapter,
  type AnthropicCompatibleAdapterOptions,
} from "./anthropic-compatible";

export type BuiltinProviderKind = "openai-compatible" | "anthropic-compatible";

export interface ProviderFactory<TOptions = unknown> {
  (options: TOptions): LLMAdapter;
}

export interface ProviderRegistry {
  register<TOptions>(kind: string, factory: ProviderFactory<TOptions>): void;
  create<TOptions>(kind: string, options: TOptions): LLMAdapter;
  has(kind: string): boolean;
  list(): string[];
}

export interface ProviderTransportConfig {
  baseURL?: string;
  apiKey?: string;
  path?: string;
  headers?: HTTPHeaders;
  defaultBody?: Record<string, unknown>;
  version?: string;
  fetcher?: typeof fetch;
}

export interface ModelAdapterConfig {
  provider: string;
  model: string;
  baseURL?: string;
  apiKey?: string;
  transport?: ProviderTransportConfig;
  options?: Record<string, unknown>;
}

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
  registry.register("anthropic-compatible", createAnthropicCompatibleAdapter);
  return registry;
}

export function createDefaultProviderRegistry(): ProviderRegistry {
  return registerBuiltinProviders(createProviderRegistry());
}

export function createModelAdapter(
  config: ModelAdapterConfig,
  registry: ProviderRegistry = createDefaultProviderRegistry(),
): LLMAdapter {
  const providerOptions = buildProviderOptions(config);
  return registry.create(config.provider, providerOptions);
}

function buildProviderOptions(config: ModelAdapterConfig): unknown {
  const transport = {
    ...config.transport,
    baseURL: config.transport?.baseURL ?? config.baseURL,
    apiKey: config.transport?.apiKey ?? config.apiKey,
  };

  if (config.provider === "openai-compatible") {
    const options: OpenAICompatibleAdapterOptions = {
      model: config.model,
      baseURL: transport.baseURL ?? "https://api.openai.com",
      apiKey: transport.apiKey,
      path: transport.path,
      headers: transport.headers,
      defaultBody: transport.defaultBody,
      fetcher: transport.fetcher,
    };

    return options;
  }

  if (config.provider === "anthropic-compatible") {
    const options: AnthropicCompatibleAdapterOptions = {
      model: config.model,
      baseURL: transport.baseURL ?? "https://api.anthropic.com",
      apiKey: transport.apiKey,
      path: transport.path,
      headers: transport.headers,
      version: transport.version,
      defaultBody: transport.defaultBody,
      fetcher: transport.fetcher,
    };

    return options;
  }

  return {
    model: config.model,
    ...transport,
    ...(config.options ?? {}),
  };
}
