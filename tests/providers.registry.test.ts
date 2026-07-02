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
    expect(registry.has("openai-compatible-legacy")).toBe(true);
    expect(registry.has("anthropic-compatible")).toBe(true);
    expect(registry.list()).toEqual([
      "anthropic-compatible",
      "openai-compatible",
      "openai-compatible-legacy",
    ]);
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

    const legacy = createModelAdapter({
      provider: "openai-compatible-legacy",
      model: "gpt-test",
      transport: { baseURL: "https://example.com" },
    });
    expect(legacy.provider).toBe("openai-compatible-legacy");
  });

  test("forwards unified transport config to the legacy provider", async () => {
    let url = "";
    let headers: RequestInit["headers"];
    let body: Record<string, unknown> = {};
    const fetcher = (async (input, init) => {
      url = String(input);
      headers = init?.headers;
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return new Response(JSON.stringify({
        choices: [{ message: { role: "assistant", content: "ok" } }],
      }));
    }) as typeof fetch;
    const adapter = createModelAdapter({
      provider: "openai-compatible-legacy",
      model: "gpt-test",
      transport: {
        baseURL: "https://legacy.example.com/root/",
        apiKey: "secret",
        path: "/chat",
        headers: { "x-test": "yes" },
        defaultBody: { seed: 42 },
        fetcher,
      },
    });

    await adapter.complete({ prompt: "hello" });

    expect(url).toBe("https://legacy.example.com/root/chat");
    expect(headers).toMatchObject({
      authorization: "Bearer secret",
      "content-type": "application/json",
      "x-test": "yes",
    });
    expect(body).toMatchObject({ model: "gpt-test", seed: 42 });
    expect(body.messages).toEqual([{ role: "user", content: "hello" }]);
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
