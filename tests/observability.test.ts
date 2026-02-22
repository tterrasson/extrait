import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { parseLLMOutput } from "../src/parse";
import { structured } from "../src/structured";
import type {
  LLMAdapter,
  LLMRequest,
  LLMResponse,
  LLMStreamCallbacks,
  ParseTraceEvent,
  StructuredTraceEvent,
} from "../src/types";

class StreamingMockAdapter implements LLMAdapter {
  provider = "mock";
  model = "mock-model";
  streamCalls = 0;

  async complete(_request: LLMRequest): Promise<LLMResponse> {
    return { text: '{"value": 1}' };
  }

  async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
    this.streamCalls += 1;
    callbacks.onStart?.();

    const tokens = ["{", '"value"', ":", " 123", "}"];
    for (const token of tokens) {
      callbacks.onToken?.(token);
      callbacks.onChunk?.({ textDelta: token, done: false });
    }

    const text = tokens.join("");
    const out = {
      text,
      usage: { inputTokens: 9, outputTokens: 3, totalTokens: 12 },
      finishReason: "stop",
    };
    callbacks.onComplete?.(out);
    return out;
  }
}

describe("observability", () => {
  test("parseLLMOutput returns traces and diagnostics per candidate", () => {
    const schema = z.object({ value: z.number() });
    const traces: ParseTraceEvent[] = [];

    const result = parseLLMOutput("Result: { value: 'oops' }", schema, {
      repair: true,
      onTrace: (event) => traces.push(event),
    });

    expect(result.success).toBe(false);
    expect(result.diagnostics.length).toBeGreaterThan(0);
    expect(result.diagnostics[0]?.stage).toBe("validate");
    expect(traces.some((event) => event.stage === "extract")).toBe(true);
    expect(traces.some((event) => event.stage === "validate")).toBe(true);
  });

  test("structured uses streaming and emits delta/data/parse", async () => {
    const schema = z.object({ value: z.number() });
    const adapter = new StreamingMockAdapter();
    const events: StructuredTraceEvent[] = [];

    const result = await structured(adapter, schema, "Return JSON", {
      stream: {
        enabled: true,
      },
      selfHeal: false,
      observe: (event) => events.push(event),
    });

    expect(adapter.streamCalls).toBe(1);
    expect(result.data).toEqual({ value: 123 });
    expect(events.some((event) => event.stage === "llm.stream.delta")).toBe(true);
    expect(events.some((event) => event.stage === "llm.stream.data")).toBe(true);
    expect(events.some((event) => event.stage === "parse")).toBe(true);
  });
});
