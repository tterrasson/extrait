import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { createLLM } from "@/llm";
import { createProviderRegistry } from "@/providers/registry";
import { DEFAULT_SCHEMA_INSTRUCTION} from "@/format";

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
          requests.push(request.prompt ?? "");
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
    expect(requests[0]).toContain(DEFAULT_SCHEMA_INSTRUCTION);
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

  test("mergeObjectLike merges stream objects and keeps explicit default disable", async () => {
    const registry = createProviderRegistry();
    let completeCalls = 0;
    let streamCalls = 0;

    registry.register(
      "mock",
      (options: { text: string }) => ({
        provider: "mock",
        model: "m1",
        async complete() {
          completeCalls += 1;
          return {
            text: options.text,
            finishReason: "stop",
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          };
        },
        async stream() {
          streamCalls += 1;
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
        options: { text: '{"val": 4}' },
        defaults: {
          stream: { enabled: false },
        },
      },
      registry,
    );

    const schema = z.object({ val: z.number() });
    const streamUpdates: unknown[] = [];
    const result = await llm.structured(schema, "Return JSON", {
      stream: {
        onData(event) {
          streamUpdates.push(event);
        },
      },
    });

    expect(result.data).toEqual({ val: 4 });
    expect(completeCalls).toBe(1);
    expect(streamCalls).toBe(0);
    expect(streamUpdates).toHaveLength(0);
  });

  test("mergeObjectLike lets boolean stream override object defaults", async () => {
    const registry = createProviderRegistry();
    let completeCalls = 0;
    let streamCalls = 0;

    registry.register(
      "mock",
      (options: { text: string }) => ({
        provider: "mock",
        model: "m1",
        async complete() {
          completeCalls += 1;
          return {
            text: options.text,
            finishReason: "stop",
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          };
        },
        async stream() {
          streamCalls += 1;
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
        options: { text: '{"val": 5}' },
        defaults: {
          stream: { enabled: true },
        },
      },
      registry,
    );

    const schema = z.object({ val: z.number() });
    const result = await llm.structured(schema, "Return JSON", {
      stream: false,
    });

    expect(result.data).toEqual({ val: 5 });
    expect(completeCalls).toBe(1);
    expect(streamCalls).toBe(0);
  });

  test("mergeObjectLike keeps default debug logger when override only toggles enabled", async () => {
    const registry = createProviderRegistry();
    const debugLogs: string[] = [];

    registry.register(
      "mock",
      (options: { text: string }) => ({
        provider: "mock",
        model: "m1",
        async complete() {
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
        options: { text: '{"val": 6}' },
        defaults: {
          debug: {
            enabled: false,
            colors: false,
            logger(line) {
              debugLogs.push(line);
            },
          },
        },
      },
      registry,
    );

    const schema = z.object({ val: z.number() });
    const result = await llm.structured(schema, "Return JSON", {
      debug: {
        enabled: true,
      },
    });

    expect(result.data).toEqual({ val: 6 });
    expect(debugLogs.some((line) => line.includes("[structured][request]"))).toBe(true);
    expect(debugLogs.some((line) => line.includes("[structured][response]"))).toBe(true);
    expect(debugLogs.some((line) => line.includes("text:"))).toBe(true);
    expect(debugLogs.some((line) => line.includes("reasoning:"))).toBe(true);
    expect(debugLogs.some((line) => line.includes("parseSource:"))).toBe(false);
  });

  test("structured debug response distinguishes text, reasoning, and parseSource", async () => {
    const registry = createProviderRegistry();
    const debugLogs: string[] = [];

    registry.register(
      "mock",
      () => ({
        provider: "mock",
        model: "m1",
        async complete() {
          return {
            text: '{"val": 7}',
            reasoning: "plan",
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
      },
      registry,
    );

    const schema = z.object({ val: z.number() });
    const result = await llm.structured(schema, "Return JSON", {
      debug: {
        enabled: true,
        colors: false,
        verbose: true,
        logger(line) {
          debugLogs.push(line);
        },
      },
    });

    expect(result.data).toEqual({ val: 7 });
    const responseLog = debugLogs.find((line) => line.includes("[structured][response]")) ?? "";
    expect(responseLog).toContain("text:");
    expect(responseLog).toContain('{"val": 7}');
    expect(responseLog).toContain("reasoning:");
    expect(responseLog).toContain("plan");
    expect(responseLog).toContain("parseSource:");
    expect(responseLog).toContain('<think>plan</think>{"val": 7}');
  });

  test("structured debug response renders empty reasoning explicitly", async () => {
    const registry = createProviderRegistry();
    const debugLogs: string[] = [];

    registry.register(
      "mock",
      (options: { text: string }) => ({
        provider: "mock",
        model: "m1",
        async complete() {
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
        options: { text: '{"val": 8}' },
      },
      registry,
    );

    const schema = z.object({ val: z.number() });
    const result = await llm.structured(schema, "Return JSON", {
      debug: {
        enabled: true,
        colors: false,
        logger(line) {
          debugLogs.push(line);
        },
      },
    });

    expect(result.data).toEqual({ val: 8 });
    const responseLog = debugLogs.find((line) => line.includes("[structured][response]")) ?? "";
    expect(responseLog).toContain("reasoning:\n(none)");
    expect(responseLog).not.toContain("parseSource:");
  });

  test("structured debug response hides parseSource by default", async () => {
    const registry = createProviderRegistry();
    const debugLogs: string[] = [];

    registry.register(
      "mock",
      () => ({
        provider: "mock",
        model: "m1",
        async complete() {
          return {
            text: "<think>plan</think>{\"val\": 9}",
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
      },
      registry,
    );

    const schema = z.object({ val: z.number() });
    const result = await llm.structured(schema, "Return JSON", {
      debug: {
        enabled: true,
        colors: false,
        logger(line) {
          debugLogs.push(line);
        },
      },
    });

    expect(result.data).toEqual({ val: 9 });
    const responseLog = debugLogs.find((line) => line.includes("[structured][response]")) ?? "";
    expect(responseLog).toContain("text:");
    expect(responseLog).toContain("{\"val\": 9}");
    expect(responseLog).toContain("reasoning:");
    expect(responseLog).toContain("plan");
    expect(responseLog).not.toContain("parseSource:");
    expect(responseLog).not.toContain("parseSourceChars=");
  });
});
