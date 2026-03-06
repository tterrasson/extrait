import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { prompt } from "../src/prompt";
import {
  buildDefaultStructuredPrompt,
  buildSelfHealPrompt,
  DEFAULT_SELF_HEAL_NO_ISSUES_MESSAGE,
  DEFAULT_STRUCTURED_OBJECT_INSTRUCTION,
  DEFAULT_STRUCTURED_STYLE_INSTRUCTION,
  structured,
  StructuredParseError,
} from "../src/structured";
import type { LLMAdapter, LLMRequest, LLMResponse, LLMStreamCallbacks } from "../src/types";
import { DEFAULT_SCHEMA_INSTRUCTION} from "../src/format";

class MockAdapter implements LLMAdapter {
  private readonly outputs: string[];
  public calls = 0;
  public requests: LLMRequest[] = [];

  constructor(outputs: string[]) {
    this.outputs = outputs;
  }

  async complete(request: LLMRequest): Promise<LLMResponse> {
    this.requests.push(request);
    const text = this.outputs[this.calls] ?? this.outputs[this.outputs.length - 1] ?? "{}";
    this.calls += 1;
    return {
      text,
      usage: {
        inputTokens: 10,
        outputTokens: 5,
        totalTokens: 15,
      },
      finishReason: "stop",
    };
  }
}

class StreamingMockAdapter implements LLMAdapter {
  public streamCalls = 0;

  async complete(): Promise<LLMResponse> {
    return { text: '{"value": 1}' };
  }

  async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
    this.streamCalls += 1;
    callbacks.onStart?.();

    const chunks = ["{", '"value"', ":", " 123", "}"];
    for (const chunk of chunks) {
      callbacks.onToken?.(chunk);
      callbacks.onChunk?.({
        textDelta: chunk,
      });
    }

    const out = {
      text: chunks.join(""),
      usage: {
        inputTokens: 20,
        outputTokens: 7,
        totalTokens: 27,
      },
      finishReason: "stop",
    };

    callbacks.onComplete?.(out);
    return out;
  }
}

describe("structured", () => {
  test("supports overload structured(adapter, schema, prompt, options) + selfHeal sugar", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(["{ value: 'oops' }", '{"value": 12}']);

    const result = await structured(model, schema, "Return a JSON with a numeric value field.", {
      mode: "loose",
      selfHeal: 1,
    });

    expect(model.calls).toBe(2);
    expect(result.data).toEqual({ value: 12 });
    expect(result.attempts.length).toBe(2);
    expect(result.attempts[0]?.success).toBe(false);
    expect(result.attempts[1]?.success).toBe(true);
    expect(result.usage?.totalTokens).toBe(30);
  });

  test("throws StructuredParseError on failure", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(["not json"]);

    let captured: unknown;

    try {
      await structured(model, schema, "Return JSON", {
        selfHeal: false,
      });
    } catch (error) {
      captured = error;
    }

    expect(captured).toBeInstanceOf(StructuredParseError);

    const parseError = captured as StructuredParseError;
    expect(parseError.name).toBe("StructuredParseError");
    expect(parseError.attempt).toBe(1);
    expect(parseError.raw).toContain("not json");
  });

  test("streaming emits progressive structured snapshots", async () => {
    const schema = z.object({ value: z.number() });
    const model = new StreamingMockAdapter();
    const snapshots: Array<{ data: unknown; done: boolean }> = [];

    const result = await structured(model, schema, "Return JSON", {
      stream: {
        enabled: true,
        onData: (event) => {
          snapshots.push({
            data: event.data,
            done: event.done,
          });
        },
      },
      selfHeal: false,
    });

    expect(model.streamCalls).toBe(1);
    expect(snapshots).toEqual([
      { data: {}, done: false },
      { data: { value: null }, done: false },
      { data: { value: 123 }, done: false },
      { data: { value: 123 }, done: true },
    ]);
    expect(result.data).toEqual({ value: 123 });
    expect(result.finishReason).toBe("stop");
    expect(result.usage?.totalTokens).toBe(27);
  });

  test("streaming exposes partial string values before full completion", async () => {
    const schema = z.object({
      sentiment: z.enum(["POSITIVE", "NEGATIVE", "NEUTRAL"]),
      confidence: z.number(),
    });

    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: '{"sentiment":"POSITIVE","confidence":0.8}' };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onStart?.();
        const deltas = ['{"sentiment":"POS', 'ITIVE","confidence":', "0.8", "}"];
        for (const delta of deltas) {
          callbacks.onToken?.(delta);
          callbacks.onChunk?.({ textDelta: delta });
        }
        const out = {
          text: deltas.join(""),
          finishReason: "stop",
        };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    const snapshots: Array<{ data: unknown; done: boolean }> = [];

    const result = await structured(model, schema, "Return JSON", {
      stream: {
        enabled: true,
        onData: (event) => {
          snapshots.push({
            data: event.data,
            done: event.done,
          });
        },
      },
      selfHeal: false,
    });

    expect(result.data).toEqual({ sentiment: "POSITIVE", confidence: 0.8 });
    expect(snapshots).toEqual([
      { data: { sentiment: "POS" }, done: false },
      { data: { sentiment: "POSITIVE", confidence: null }, done: false },
      { data: { sentiment: "POSITIVE", confidence: 0.8 }, done: false },
      { data: { sentiment: "POSITIVE", confidence: 0.8 }, done: true },
    ]);
  });

  test("streaming handles prose prefixes with apostrophes before JSON", async () => {
    const schema = z.object({
      sentiment: z.string(),
      confidence: z.number(),
    });

    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: `Voici l'analyse: {"sentiment":"POSITIVE","confidence":0.8}` };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onStart?.();
        const deltas = ["Voici l'analyse: ", '{"sentiment":"POS', 'ITIVE","confidence":0.8}'];
        for (const delta of deltas) {
          callbacks.onToken?.(delta);
          callbacks.onChunk?.({ textDelta: delta });
        }
        const out = {
          text: deltas.join(""),
          finishReason: "stop",
        };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    const snapshots: Array<{ data: unknown; done: boolean }> = [];
    const result = await structured(model, schema, "Return JSON", {
      stream: {
        enabled: true,
        onData: (event) => snapshots.push({ data: event.data, done: event.done }),
      },
      selfHeal: false,
    });

    expect(result.data).toEqual({ sentiment: "POSITIVE", confidence: 0.8 });
    expect(snapshots).toEqual([
      { data: { sentiment: "POS" }, done: false },
      { data: { sentiment: "POSITIVE", confidence: 0.8 }, done: false },
      { data: { sentiment: "POSITIVE", confidence: 0.8 }, done: true },
    ]);
  });

  test("streaming emits a final done event even without parsable partial data", async () => {
    const schema = z.object({ value: z.number() });

    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: "not json at all" };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onStart?.();
        callbacks.onToken?.("not ");
        callbacks.onChunk?.({ textDelta: "not " });
        callbacks.onToken?.("json");
        callbacks.onChunk?.({ textDelta: "json" });
        const out = {
          text: "not json",
          finishReason: "stop",
        };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    const events: Array<{ data: unknown; done: boolean }> = [];

    await expect(
      structured(model, schema, "Return JSON", {
        stream: {
          enabled: true,
          onData: (event) => events.push({ data: event.data, done: event.done }),
        },
        selfHeal: false,
      }),
    ).rejects.toBeInstanceOf(StructuredParseError);

    expect(events).toEqual([{ data: null, done: true }]);
  });

  test("automatically injects the enriched schema format", async () => {
    const schema = z
      .object({
        summary: z.string().describe("summary text"),
        tags: z.array(z.string()).default([]).describe("tag list"),
      })
      .describe("Summary object");


    const model = new MockAdapter(['{"summary":"ok","tags":[]}']);

    const result = await structured(model, schema, "Create a summary", {
      selfHeal: false,
    });

    expect(result.data.summary).toBe("ok");

    const sentPrompt = model.requests[0]?.prompt ?? "";
    expect(sentPrompt).toContain(DEFAULT_SCHEMA_INSTRUCTION);
    expect(sentPrompt).toContain("summary: string,  // summary text");
    expect(sentPrompt).toContain("tags: string[],  // tag list");
    expect(sentPrompt).not.toContain("Required fields:");
    expect(sentPrompt).not.toContain("Defaults:");
  });

  test("mode 'strict' disables repair (valid JSON OK, JSON-ish KO)", async () => {
    const schema = z.object({ value: z.number() });

    const valid = new MockAdapter(['{"value": 42}']);
    const result = await structured(valid, schema, "Return JSON", {
      mode: "strict",
      selfHeal: false,
    });
    expect(result.data).toEqual({ value: 42 });

    const malformed = new MockAdapter(["{ value: 42 }"]);
    let caught: unknown;
    try {
      await structured(malformed, schema, "Return JSON", { mode: "strict", selfHeal: false });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(StructuredParseError);
  });

  test("buildDefaultStructuredPrompt includes the task and instructions", () => {
    const result = buildDefaultStructuredPrompt("Analyze this text");
    expect(result).toContain("Analyze this text");
    expect(result).toContain(DEFAULT_STRUCTURED_OBJECT_INSTRUCTION);
    expect(result).toContain(DEFAULT_STRUCTURED_STYLE_INSTRUCTION);
  });

  test("buildDefaultStructuredPrompt accepts custom instruction lines", () => {
    const result = buildDefaultStructuredPrompt("Analyze this text", {
      objectInstruction: "Return one JSON object only.",
      styleInstruction: "No markdown.",
    });

    expect(result).toContain("Return one JSON object only.");
    expect(result).toContain("No markdown.");
  });

  test("buildSelfHealPrompt includes the schema, errors and raw output", () => {
    const schema = z.object({ name: z.string() });
    const result = buildSelfHealPrompt({
      rawOutput: '{"name": 42}',
      issues: [
        { path: ["name"], message: "Expected string, received number", code: "invalid_type", expected: "string", received: "number" } as z.ZodIssue,
      ],
      schema,
    });
    expect(result).toContain("Fix the following output");
    expect(result).toContain(DEFAULT_SCHEMA_INSTRUCTION);
    expect(result).toContain("name: string");
    expect(result).toContain("name: Expected string, received number");
    expect(result).toContain('{"name": 42}');
  });

  test("buildSelfHealPrompt accepts custom static labels", () => {
    const schema = z.object({ name: z.string() });
    const result = buildSelfHealPrompt({
      rawOutput: '{"name": "ok"}',
      issues: [],
      schema,
      text: {
        noIssuesMessage: "Nothing to fix.",
        validationErrorsLabel: "Errors:",
      },
    });

    expect(result).toContain("Errors:");
    expect(result).toContain("Nothing to fix.");
    expect(result).not.toContain(DEFAULT_SELF_HEAL_NO_ISSUES_MESSAGE);
  });

  test("buildSelfHealPrompt includes structured self-heal context payload", () => {
    const schema = z.object({ name: z.string() });
    const result = buildSelfHealPrompt({
      rawOutput: '{"name": 42}',
      issues: [
        { path: ["name"], message: "Expected string, received number", code: "invalid_type", expected: "string", received: "number" } as z.ZodIssue,
      ],
      schema,
      selectedInput: "candidate",
      selectedOutput: '{"name": 42}',
      parserErrors: [{ stage: "validate", message: "name invalid" }],
    });

    expect(result).toContain('"protocol": "extrait.self-heal.v2"');
    expect(result).toContain('"selectedInput": "candidate"');
    expect(result).toContain('"validationIssues"');
    expect(result).toContain('"parserErrors"');
  });

  test("self-heal stops early when there is no progress", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(['{"value":"oops"}', '{"value":"oops"}', '{"value":"oops"}']);

    await expect(
      structured(model, schema, "Return JSON", {
        mode: "loose",
        selfHeal: { enabled: true, maxAttempts: 3, stopOnNoProgress: true },
      }),
    ).rejects.toBeInstanceOf(StructuredParseError);

    expect(model.calls).toBe(2);
  });

  test("prompt as function receives context with mode", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(['{"value": 1}']);
    let receivedContext: { mode: string } | undefined;

    await structured(
      model,
      schema,
      (ctx) => {
        receivedContext = ctx;
        return "Return JSON";
      },
      { mode: "strict", selfHeal: false },
    );

    expect(receivedContext).toBeDefined();
    expect(receivedContext!.mode).toBe("strict");
  });

  test("accepts prompt builder with system + user messages", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(['{"value": 7}']);

    const result = await structured(
      model,
      schema,
      prompt()
        .system`You are a strict JSON assistant.`
        .user`
          Return a value.
        `,
      { selfHeal: false },
    );

    expect(result.data).toEqual({ value: 7 });
    expect(model.requests[0]?.messages).toEqual([
      { role: "system", content: "You are a strict JSON assistant." },
      {
        role: "user",
        content: expect.stringContaining(DEFAULT_SCHEMA_INSTRUCTION),
      },
    ]);
    expect((model.requests[0]?.messages?.[1] as { content?: string } | undefined)?.content).toContain("Return a value.");
  });

  test("injects format into last user message in multi-turn conversation", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(['{"value": 9}']);

    const result = await structured(
      model,
      schema,
      prompt()
        .system`You are helpful.`
        .user`What is 4+5?`
        .assistant`The answer is 9.`
        .user`Confirm as JSON.`,
      { selfHeal: false },
    );

    expect(result.data).toEqual({ value: 9 });
    const messages = model.requests[0]?.messages ?? [];
    // Format injected into LAST user message (index 3), not first user message (index 1)
    expect((messages[3] as { content?: string }).content).toContain(DEFAULT_SCHEMA_INSTRUCTION);
    expect((messages[1] as { content?: string }).content).not.toContain(DEFAULT_SCHEMA_INSTRUCTION);
  });

  test("prepends systemPrompt option as system message when using messages-based prompt", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(['{"value": 5}']);

    const result = await structured(
      model,
      schema,
      prompt().user`Return a value.`,
      { selfHeal: false, systemPrompt: "You are a strict JSON assistant." },
    );

    expect(result.data).toEqual({ value: 5 });
    const messages = model.requests[0]?.messages ?? [];
    expect(messages[0]).toEqual({ role: "system", content: "You are a strict JSON assistant." });
    expect((messages[1] as { content?: string }).content).toContain(DEFAULT_SCHEMA_INSTRUCTION);
  });

  test("forwards request.signal to adapter complete calls", async () => {
    const schema = z.object({ value: z.number() });
    const controller = new AbortController();
    let receivedSignal: AbortSignal | undefined;

    const model: LLMAdapter = {
      async complete(request: LLMRequest): Promise<LLMResponse> {
        receivedSignal = request.signal;
        return {
          text: '{"value": 11}',
          finishReason: "stop",
        };
      },
    };

    const result = await structured(model, schema, "Return JSON", {
      selfHeal: false,
      request: {
        signal: controller.signal,
      },
    });

    expect(result.data).toEqual({ value: 11 });
    expect(receivedSignal).toBe(controller.signal);
  });

  test("forwards request.signal to adapter stream calls", async () => {
    const schema = z.object({ value: z.number() });
    const controller = new AbortController();
    let receivedSignal: AbortSignal | undefined;

    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: '{"value": 22}' };
      },
      async stream(request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        receivedSignal = request.signal;
        callbacks.onStart?.();
        callbacks.onToken?.('{"value":22}');
        callbacks.onChunk?.({ textDelta: '{"value":22}' });
        const out = {
          text: '{"value":22}',
          finishReason: "stop",
        };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    const result = await structured(model, schema, "Return JSON", {
      selfHeal: false,
      request: {
        signal: controller.signal,
      },
      stream: {
        enabled: true,
      },
    });

    expect(result.data).toEqual({ value: 22 });
    expect(receivedSignal).toBe(controller.signal);
  });

  test("outdents multiline prompt strings by default", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(['{"value": 1}']);

    await structured(
      model,
      schema,
      `
        First line.
        Second line.
      `,
      { selfHeal: false },
    );

    const sentPrompt = model.requests[0]?.prompt ?? "";
    expect(sentPrompt).toContain("First line.\nSecond line.");
    expect(sentPrompt).not.toContain("First line.\n        Second line.");
  });

  test("can disable outdent in structured options", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(['{"value": 1}']);

    await structured(
      model,
      schema,
      `
        First line.
        Second line.
      `,
      { selfHeal: false, outdent: false },
    );

    const sentPrompt = model.requests[0]?.prompt ?? "";
    expect(sentPrompt).toContain("First line.\n        Second line.");
  });

  test("missing prompt throws a clear error", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter(["{}"])

    await expect(structured(model, schema, undefined as any)).rejects.toThrow("Missing prompt");
  });

  test("keeps think blocks in result while ignoring them for parsing", async () => {
    const schema = z.object({ value: z.number() });
    const model = new MockAdapter([
      ["<think>", '{"value": 0}', "</think>", '{"value": 7}'].join("\n"),
    ]);

    const result = await structured(model, schema, "Return JSON", {
      selfHeal: false,
    });

    expect(result.data).toEqual({ value: 7 });
    expect(result.thinkBlocks.length).toBe(1);
    expect(result.thinkBlocks[0]?.content).toContain('{"value": 0}');
    expect(result.attempts[0]?.thinkBlocks.length).toBe(1);
  });
});
