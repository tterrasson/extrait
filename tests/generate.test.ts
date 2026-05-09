import { describe, expect, test } from "bun:test";
import { generate } from "@/generate";
import { prompt } from "@/prompt";
import type {
  GenerateStreamEvent,
  LLMAdapter,
  LLMRequest,
  LLMResponse,
  LLMStreamCallbacks,
  MCPToolClient,
} from "@/types";

class MockAdapter implements LLMAdapter {
  public requests: LLMRequest[] = [];
  constructor(private readonly response: LLMResponse) {}

  async complete(request: LLMRequest): Promise<LLMResponse> {
    this.requests.push(request);
    return this.response;
  }
}

describe("generate", () => {
  test("supports generate(adapter, prompt, options)", async () => {
    const model = new MockAdapter({
      text: "Hello world",
      finishReason: "stop",
      usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
    });

    const result = await generate(model, "Say hello", {
      request: {
        temperature: 0.3,
        body: { user: "demo" },
      },
    });

    expect(result).toMatchObject({
      text: "Hello world",
      reasoning: "",
      attempts: [
        {
          attempt: 1,
          via: "complete",
          text: "Hello world",
          reasoning: "",
          usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
          finishReason: "stop",
        },
      ],
      usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
      finishReason: "stop",
    });
    expect(model.requests[0]).toMatchObject({
      prompt: "Say hello",
      temperature: 0.3,
      body: { user: "demo" },
    });
  });

  test("forwards request.reasoningEffort to the adapter request", async () => {
    const model = new MockAdapter({
      text: "Hello world",
    });

    await generate(model, "Say hello", {
      request: {
        reasoningEffort: "max",
      },
    });

    expect(model.requests[0]).toMatchObject({
      prompt: "Say hello",
      reasoningEffort: "max",
    });
  });

  test("supports generate(adapter, { prompt, ...options }) with messages and merged system prompt", async () => {
    const model = new MockAdapter({
      text: "Answer",
      reasoning: "plan",
      finishReason: "stop",
    });

    const result = await generate(model, {
      prompt: prompt()
        .system`You are concise.`
        .user`Question?`,
      systemPrompt: "Be accurate.",
    });

    expect(result.text).toBe("Answer");
    expect(result.reasoning).toBe("plan");
    expect(model.requests[0]?.messages).toEqual([
      { role: "system", content: "Be accurate." },
      { role: "system", content: "You are concise." },
      { role: "user", content: "Question?" },
    ]);
  });

  test("outdents multiline prompt strings by default", async () => {
    const model = new MockAdapter({ text: "ok" });

    await generate(
      model,
      `
        First line.
        Second line.
      `,
    );

    expect(model.requests[0]?.prompt).toBe("First line.\nSecond line.");
  });

  test("can disable outdent", async () => {
    const model = new MockAdapter({ text: "ok" });

    await generate(
      model,
      `
        First line.
        Second line.
      `,
      { outdent: false },
    );

    expect(model.requests[0]?.prompt).toContain("First line.\n        Second line.");
  });

  test("streaming emits text/reasoning snapshots and final done=true event", async () => {
    const events: GenerateStreamEvent[] = [];

    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: "Hello", reasoning: "plan", finishReason: "stop" };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onStart?.();
        callbacks.onChunk?.({ textDelta: "", reasoningDelta: "plan" });
        callbacks.onChunk?.({ textDelta: "Hel" });
        callbacks.onChunk?.({ textDelta: "lo", finishReason: "stop" });
        const out = {
          text: "Hello",
          reasoning: "plan",
          finishReason: "stop",
          usage: { inputTokens: 7, outputTokens: 4, totalTokens: 11 },
        };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    const result = await generate(model, "Say hello", {
      stream: {
        enabled: true,
        onData: (event) => events.push(event),
      },
    });

    expect(result.text).toBe("Hello");
    expect(result.reasoning).toBe("plan");
    expect(result.attempts).toEqual([
      {
        attempt: 1,
        via: "stream",
        text: "Hello",
        reasoning: "plan",
        usage: { inputTokens: 7, outputTokens: 4, totalTokens: 11 },
        finishReason: "stop",
      },
    ]);
    expect(events).toEqual([
      {
        delta: { text: "", reasoning: "plan" },
        snapshot: { text: "", reasoning: "plan" },
        done: false,
        usage: undefined,
        finishReason: undefined,
      },
      {
        delta: { text: "Hel", reasoning: "" },
        snapshot: { text: "Hel", reasoning: "plan" },
        done: false,
        usage: undefined,
        finishReason: undefined,
      },
      {
        delta: { text: "lo", reasoning: "" },
        snapshot: { text: "Hello", reasoning: "plan" },
        done: false,
        usage: undefined,
        finishReason: undefined,
      },
      {
        delta: { text: "", reasoning: "" },
        snapshot: { text: "Hello", reasoning: "plan" },
        done: true,
        usage: { inputTokens: 7, outputTokens: 4, totalTokens: 11 },
        finishReason: "stop",
      },
    ]);
  });

  test("streaming exposes turn metadata and cumulative reasoning blocks", async () => {
    const events: GenerateStreamEvent[] = [];
    const transitions: string[] = [];

    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: "done" };
      },
      async stream(request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onStart?.();
        callbacks.onChunk?.({ textDelta: "", reasoningDelta: "first", turnIndex: 1 });
        request.onTurnTransition?.({ turnIndex: 1, kind: "reasoningComplete", reasoningText: "first" });
        request.onTurnTransition?.({
          turnIndex: 1,
          kind: "toolCallsEmit",
          toolCalls: [{ id: "call_1", type: "function", name: "add" }],
        });
        callbacks.onChunk?.({
          textDelta: "",
          turnIndex: 1,
          toolCalls: [{ id: "call_1", type: "function", name: "add" }],
        });
        request.onTurnTransition?.({ turnIndex: 1, kind: "toolResultsReceived" });
        callbacks.onChunk?.({ textDelta: "", reasoningDelta: "second", turnIndex: 2 });
        callbacks.onChunk?.({ textDelta: "done", turnIndex: 2, finishReason: "stop" });
        request.onTurnTransition?.({ turnIndex: 2, kind: "reasoningComplete", reasoningText: "second" });
        request.onTurnTransition?.({ turnIndex: 2, kind: "streamEnd" });
        const out: LLMResponse = {
          text: "done",
          reasoning: "first\n\nsecond",
          reasoningBlocks: [
            { turnIndex: 1, text: "first" },
            { turnIndex: 2, text: "second" },
          ],
          finishReason: "stop",
        };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    const result = await generate(model, "Say hello", {
      stream: {
        enabled: true,
        onTurnTransition: (transition) => transitions.push(`${transition.turnIndex}:${transition.kind}`),
        onData: (event) => events.push(event),
      },
    });

    expect(result.reasoningBlocks).toEqual([
      { turnIndex: 1, text: "first" },
      { turnIndex: 2, text: "second" },
    ]);
    expect(events.some((event) => event.turnIndex === 1)).toBe(true);
    expect(events.some((event) => event.toolCalls?.[0]?.id === "call_1")).toBe(true);
    expect(events.find((event) => event.toolCalls)?.snapshot.reasoningBlocks).toEqual([
      { turnIndex: 1, text: "first" },
    ]);
    expect(events.at(-1)?.snapshot.reasoningBlocks).toEqual(result.reasoningBlocks);
    expect(transitions).toEqual([
      "1:reasoningComplete",
      "1:toolCallsEmit",
      "1:toolResultsReceived",
      "2:reasoningComplete",
      "2:streamEnd",
    ]);
  });

  test("streaming suppresses duplicate snapshots but still emits final done event", async () => {
    const events: GenerateStreamEvent[] = [];

    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: "ok" };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onStart?.();
        callbacks.onChunk?.({ textDelta: "<think>" });
        callbacks.onChunk?.({ textDelta: "</think>" });
        callbacks.onChunk?.({ textDelta: "ok", finishReason: "stop" });
        const out = { text: "<think></think>ok", finishReason: "stop" };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    const result = await generate(model, "Say hello", {
      stream: {
        enabled: true,
        onData: (event) => events.push(event),
      },
    });

    expect(result.text).toBe("ok");
    expect(events).toEqual([
      {
        delta: { text: "", reasoning: "" },
        snapshot: { text: "", reasoning: "" },
        done: false,
        usage: undefined,
        finishReason: undefined,
      },
      {
        delta: { text: "ok", reasoning: "" },
        snapshot: { text: "ok", reasoning: "" },
        done: false,
        usage: undefined,
        finishReason: undefined,
      },
      {
        delta: { text: "", reasoning: "" },
        snapshot: { text: "ok", reasoning: "" },
        done: true,
        usage: undefined,
        finishReason: "stop",
      },
    ]);
  });

  test("stream.to stdout writes only visible text", async () => {
    const writes: string[] = [];
    const originalWrite = process.stdout.write.bind(process.stdout);

    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: "Hello", reasoning: "plan" };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onStart?.();
        callbacks.onChunk?.({ textDelta: "", reasoningDelta: "plan" });
        callbacks.onChunk?.({ textDelta: "Hello", finishReason: "stop" });
        const out = {
          text: "Hello",
          reasoning: "plan",
          finishReason: "stop",
        };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    process.stdout.write = ((chunk: string | Uint8Array) => {
      writes.push(typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf8"));
      return true;
    }) as typeof process.stdout.write;

    try {
      const result = await generate(model, "Say hello", {
        stream: {
          enabled: true,
          to: "stdout",
        },
      });

      expect(result.reasoning).toBe("plan");
      expect(writes.join("")).toBe("Hello");
    } finally {
      process.stdout.write = originalWrite as typeof process.stdout.write;
    }
  });

  test("normalizes inline think blocks and dedicated reasoning", async () => {
    const model = new MockAdapter({
      text: '<think>draft</think>Hello',
      reasoning: "plan",
      finishReason: "stop",
    });

    const result = await generate(model, "Say hello");

    expect(result.text).toBe("Hello");
    expect(result.reasoning).toBe("plan\n\ndraft");
  });

  test("forwards request.signal to complete and stream calls", async () => {
    const controller = new AbortController();
    const signals: AbortSignal[] = [];

    const model: LLMAdapter = {
      async complete(request: LLMRequest): Promise<LLMResponse> {
        signals.push(request.signal as AbortSignal);
        return { text: "done", finishReason: "stop" };
      },
      async stream(request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        signals.push(request.signal as AbortSignal);
        callbacks.onStart?.();
        callbacks.onChunk?.({ textDelta: "done", finishReason: "stop" });
        const out = { text: "done", finishReason: "stop" };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    await generate(model, "one", {
      request: { signal: controller.signal },
    });
    await generate(model, "two", {
      request: { signal: controller.signal },
      stream: true,
    });

    expect(signals).toEqual([controller.signal, controller.signal]);
  });

  test("uses timeout.request signal when no request signal is provided", async () => {
    const signals: AbortSignal[] = [];

    const model: LLMAdapter = {
      async complete(request: LLMRequest): Promise<LLMResponse> {
        signals.push(request.signal as AbortSignal);
        return { text: "done", finishReason: "stop" };
      },
      async stream(request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        signals.push(request.signal as AbortSignal);
        callbacks.onStart?.();
        callbacks.onChunk?.({ textDelta: "done", finishReason: "stop" });
        const out = { text: "done", finishReason: "stop" };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    await generate(model, "one", {
      timeout: { request: 1_000 },
    });
    await generate(model, "two", {
      timeout: { request: 1_000 },
      stream: true,
    });

    expect(signals).toHaveLength(2);
    expect(signals.every((signal) => signal instanceof AbortSignal)).toBe(true);
  });

  test("request.signal takes precedence over timeout.request", async () => {
    const controller = new AbortController();
    let receivedSignal: AbortSignal | undefined;

    const model: LLMAdapter = {
      async complete(request: LLMRequest): Promise<LLMResponse> {
        receivedSignal = request.signal;
        return { text: "done", finishReason: "stop" };
      },
    };

    await generate(model, "one", {
      request: { signal: controller.signal },
      timeout: { request: 1_000 },
    });

    expect(receivedSignal).toBe(controller.signal);
  });

  test("applies timeout.tool to MCP clients passed through generate", async () => {
    const slowClient: MCPToolClient = {
      id: "slow",
      async listTools() {
        return { tools: [{ name: "wait", inputSchema: { type: "object", properties: {} } }] };
      },
      async callTool() {
        return new Promise(() => undefined);
      },
    };

    const model: LLMAdapter = {
      async complete(request: LLMRequest): Promise<LLMResponse> {
        await request.mcpClients?.[0]?.callTool({ name: "wait", arguments: {} });
        return { text: "unreachable" };
      },
    };

    await expect(
      generate(model, "Use the tool", {
        request: {
          mcpClients: [slowClient],
        },
        timeout: { tool: 5 },
      }),
    ).rejects.toThrow("Tool call timed out after 5ms");
  });
});
