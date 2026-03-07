import { describe, expect, test } from "bun:test";
import {
  createDefaultProviderRegistry,
  createModelAdapter,
  createProviderRegistry,
  registerBuiltinProviders,
} from "@/providers/registry";

describe("provider registry", () => {
  test("registers built-in providers", () => {
    const registry = createDefaultProviderRegistry();

    expect(registry.has("openai-compatible")).toBe(true);
    expect(registry.has("anthropic-compatible")).toBe(true);
    expect(registry.list()).toEqual(["anthropic-compatible", "openai-compatible"]);
  });

  test("instantiates an adapter from unified provider config", () => {
    const adapter = createModelAdapter({
      provider: "openai-compatible",
      model: "gpt-test",
      transport: {
        baseURL: "https://example.com",
      },
    });

    expect(adapter.provider).toBe("openai-compatible");
    expect(adapter.model).toBe("gpt-test");
  });

  test("accepts a custom provider via register", async () => {
    const registry = createProviderRegistry();
    registerBuiltinProviders(registry);

    registry.register("custom", (options: { text: string }) => ({
      provider: "custom",
      async complete() {
        return { text: options.text };
      },
    }));

    const adapter = createModelAdapter(
      {
        provider: "custom",
        model: "custom-model",
        options: {
          text: "ok",
        },
      },
      registry,
    );

    const out = await adapter.complete({ prompt: "hello" });
    expect(out.text).toBe("ok");
  });
});
