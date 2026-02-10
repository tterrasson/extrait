import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { createLLM } from "../src/llm";
import { createProviderRegistry } from "../src/providers/registry";

describe("createLLM", () => {
  test("configures a single client with defaults + override per call", async () => {
    const registry = createProviderRegistry();
    const requests: string[] = [];

    registry.register(
      "mock",
      (options: { text: string }) => ({
        provider: "mock",
        model: "m1",
        async complete(request) {
          requests.push(request.prompt);
          return {
            text: options.text,
            finishReason: "stop",
            usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
          };
        },
      }),
    );

    const llm = createLLM(
      {
        provider: "mock",
        model: "m1",
        options: { text: '{"value": 42}' },
        defaults: {
          mode: "loose",
          selfHeal: 1,
          debug: true,
        },
      },
      registry,
    );

    const schema = z.object({ value: z.number() });
    const result = await llm.structured(schema, "Return a JSON", {
      debug: false,
      selfHeal: false,
    });

    expect(result.data).toEqual({ value: 42 });
    expect(result.finishReason).toBe("stop");
    expect(requests[0]).toContain("Strictly follow this schema:");
  });

  test("merges with no defaults and no overrides", async () => {
    const registry = createProviderRegistry();

    registry.register(
      "mock",
      (options: { text: string }) => ({
        provider: "mock",
        model: "m1",
        async complete(request) {
          return {
            text: options.text,
            finishReason: "stop",
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          };
        },
      }),
    );

    const llm = createLLM(
      {
        provider: "mock",
        model: "m1",
        options: { text: '{"val": 1}' },
      },
      registry,
    );

    const schema = z.object({ val: z.number() });
    const result = await llm.structured(schema, "Return JSON");
    expect(result.data).toEqual({ val: 1 });
  });

  test("defaults only (no per-call overrides)", async () => {
    const registry = createProviderRegistry();

    registry.register(
      "mock",
      (options: { text: string }) => ({
        provider: "mock",
        model: "m1",
        async complete(request) {
          return {
            text: options.text,
            finishReason: "stop",
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          };
        },
      }),
    );

    const llm = createLLM(
      {
        provider: "mock",
        model: "m1",
        options: { text: '{"val": 2}' },
        defaults: {
          mode: "loose",
        },
      },
      registry,
    );

    const schema = z.object({ val: z.number() });
    const result = await llm.structured(schema, "Return JSON");
    expect(result.data).toEqual({ val: 2 });
  });

  test("mergeObjectLike: override is non-plain overrides default object", async () => {
    const registry = createProviderRegistry();

    registry.register(
      "mock",
      (options: { text: string }) => ({
        provider: "mock",
        model: "m1",
        async complete(request) {
          return {
            text: options.text,
            finishReason: "stop",
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          };
        },
      }),
    );

    const llm = createLLM(
      {
        provider: "mock",
        model: "m1",
        options: { text: '{"val": 3}' },
        defaults: {
          selfHeal: { enabled: true, maxAttempts: 3 },
          debug: { enabled: true },
        },
      },
      registry,
    );

    const schema = z.object({ val: z.number() });
    // Override object with boolean (non-plain overrides plain)
    const result = await llm.structured(schema, "Return JSON", {
      selfHeal: false,
      debug: false,
    });
    expect(result.data).toEqual({ val: 3 });
  });
});
