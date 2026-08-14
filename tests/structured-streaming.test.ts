import { describe, expect, test } from "bun:test";
import { jsonrepair } from "jsonrepair";
import { z } from "zod";
import { parseStreamingStructuredData } from "@/structured-streaming";
import { normalizeModelOutput } from "@/generate-output";
import { sanitizeThink } from "@/think";
import { structured } from "@/structured";
import type {
  LLMAdapter,
  LLMRequest,
  LLMResponse,
  LLMStreamCallbacks,
  StructuredStreamEvent,
} from "@/types";

// Frozen copy of the pre-optimization implementation (STREAM.md étape 1b):
// it received `parseSource` (`<think>${reasoning}</think>${text}`) and
// re-sanitized it before searching for a JSON root. Kept verbatim as the
// reference for the differential test below.
function referenceParseStreamingStructuredData(parseSource: string): unknown | null {
  const sanitized = sanitizeThink(parseSource);
  const start = referenceFindFirstJsonRootStart(sanitized.visibleText);
  if (start < 0) {
    return null;
  }

  const candidate = sanitized.visibleText.slice(start).trim();
  if (!candidate) {
    return null;
  }

  try {
    const repaired = jsonrepair(candidate);
    const parsed = JSON.parse(repaired);
    if (typeof parsed !== "object" || parsed === null) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function referenceFindFirstJsonRootStart(input: string): number {
  let inString = false;
  let escaped = false;

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    if (!char) {
      continue;
    }

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\") {
        escaped = true;
        continue;
      }
      if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === "{" || char === "[") {
      return index;
    }
  }

  const objectStart = input.indexOf("{");
  const arrayStart = input.indexOf("[");
  if (objectStart < 0) {
    return arrayStart;
  }
  if (arrayStart < 0) {
    return objectStart;
  }
  return Math.min(objectStart, arrayStart);
}

describe("parseStreamingStructuredData differential (étape 1b)", () => {
  const cases: Array<{ name: string; text: string; reasoning?: string }> = [
    { name: "plain JSON", text: '{"value": 42}' },
    { name: "JSON with prose prefix", text: 'Here you go: {"value": 42}' },
    { name: "empty text", text: "" },
    { name: "no JSON at all", text: "just words" },
    { name: "incomplete JSON string value", text: '{"value": "unfinished' },
    { name: "incomplete nested array", text: '{"items": [1, 2,' },
    {
      name: "think block inline in text",
      text: '<think>internal notes {"decoy": 1}</think>{"value": 7}',
    },
    { name: "unterminated think block", text: '<think>still thinking {"decoy": 1}' },
    { name: "orphan </think> in text", text: 'prefix </think> {"value": 3}' },
    { name: "orphan </think> inside JSON string", text: '{"value": "a </think> b"}' },
    { name: "JSON with dedicated reasoning", text: '{"value": 5}', reasoning: "chain of thought" },
    {
      name: "JSON only in reasoning",
      text: "no structured data here",
      reasoning: 'the answer is {"value": 9}',
    },
    {
      name: "reasoning containing think tags",
      text: '{"value": 11}',
      reasoning: "<think>nested</think> more",
    },
    {
      name: "reasoning containing attribute think tags",
      text: '{"value": 13}',
      reasoning: "<think foo>secret</think>",
    },
    {
      name: "nested attribute think tag in text",
      text: '<think><think foo>secret</think></think>{"value": 14}',
    },
    {
      name: "quoted brace before real root",
      text: 'The token "{" is special. {"value": 12}',
    },
    { name: "reasoning only, no text", text: "", reasoning: "thinking out loud" },
  ];

  for (const { name, text, reasoning } of cases) {
    test(name, () => {
      const normalized = normalizeModelOutput(text, reasoning);
      expect(parseStreamingStructuredData(normalized.text)).toEqual(
        referenceParseStreamingStructuredData(normalized.parseSource),
      );
    });
  }

  test("stream.dataInterval coalesces data recomputation; done always reparses", async () => {
    const events: Array<StructuredStreamEvent<{ a: number; b: number }>> = [];
    const finalText = '{"a": 1, "b": 2}';
    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: finalText };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onStart?.();
        callbacks.onChunk?.({ textDelta: '{"a": 1' });
        callbacks.onChunk?.({ textDelta: ', "b": 2' });
        callbacks.onChunk?.({ textDelta: "}", finishReason: "stop" });
        const out: LLMResponse = { text: finalText, finishReason: "stop" };
        callbacks.onComplete?.(out);
        return out;
      },
    };

    const result = await structured(model, z.object({ a: z.number(), b: z.number() }), "Return JSON", {
      stream: {
        enabled: true,
        dataInterval: 60_000,
        onData: (event) => events.push(event),
      },
    });

    expect(result.data).toEqual({ a: 1, b: 2 });
    expect(events).toHaveLength(4);
    // First event parses; the next two reuse the coalesced value even though
    // the accumulated text would parse further.
    expect(events[0]?.snapshot.data).toEqual({ a: 1 } as never);
    expect(events[1]?.snapshot.data).toEqual({ a: 1 } as never);
    expect(events[2]?.snapshot.data).toEqual({ a: 1 } as never);
    // The terminal event always reparses.
    expect(events[3]?.done).toBe(true);
    expect(events[3]?.snapshot.data).toEqual({ a: 1, b: 2 });
  });

  test("default dataInterval stays exact on small outputs and coalesces large ones", async () => {
    // Small accumulation (≤ 2 ko): every event reparses, current behavior.
    const smallEvents: Array<StructuredStreamEvent<{ a: number; b: number }>> = [];
    const smallText = '{"a": 1, "b": 2}';
    const smallModel: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: smallText };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onChunk?.({ textDelta: '{"a": 1' });
        callbacks.onChunk?.({ textDelta: ', "b": 2}' });
        return { text: smallText, finishReason: "stop" };
      },
    };
    await structured(smallModel, z.object({ a: z.number(), b: z.number() }), "Return JSON", {
      stream: { enabled: true, onData: (event) => smallEvents.push(event) },
    });
    expect(smallEvents[0]?.snapshot.data).toEqual({ a: 1 } as never);
    expect(smallEvents[1]?.snapshot.data).toEqual({ a: 1, b: 2 });

    // Large accumulation (> 2 ko): synchronous chunks land inside the 25 ms
    // window, so intermediate events reuse the coalesced value; done reparses.
    const pad = "x".repeat(3000);
    const largeText = `{"pad": "${pad}", "a": 1, "b": 2}`;
    const largeEvents: Array<StructuredStreamEvent<{ pad: string; a: number; b: number }>> = [];
    const largeModel: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: largeText };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onChunk?.({ textDelta: `{"pad": "${pad}", "a": 1` });
        callbacks.onChunk?.({ textDelta: ', "b": 2' });
        callbacks.onChunk?.({ textDelta: "}", finishReason: "stop" });
        return { text: largeText, finishReason: "stop" };
      },
    };
    await structured(
      largeModel,
      z.object({ pad: z.string(), a: z.number(), b: z.number() }),
      "Return JSON",
      { stream: { enabled: true, onData: (event) => largeEvents.push(event) } },
    );
    expect(largeEvents).toHaveLength(4);
    expect(largeEvents[0]?.snapshot.data).toEqual({ pad, a: 1 } as never);
    // Reused by reference: no reparse happened for the intermediate events.
    expect(largeEvents[1]?.snapshot.data).toBe(largeEvents[0]?.snapshot.data as never);
    expect(largeEvents[2]?.snapshot.data).toBe(largeEvents[0]?.snapshot.data as never);
    expect(largeEvents[3]?.done).toBe(true);
    expect(largeEvents[3]?.snapshot.data).toEqual({ pad, a: 1, b: 2 });
  });

  test("attribute think tags cannot make the done snapshot diverge from the final parse", async () => {
    // Regression: `<think foo>` is recognized by the scanner but was not
    // neutralized by the literal-tag regex when reasoning is re-wrapped into
    // `<think>...</think>` for the final parse. The unbalanced nesting made
    // sanitization swallow the visible JSON: the done event announced valid
    // data, then structured() threw StructuredParseError on the same output.
    const text = '<think><think foo>secret</think></think>{"value": 1}';
    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text };
      },
      async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
        callbacks.onChunk?.({ textDelta: text });
        return { text, finishReason: "stop" };
      },
    };

    let doneData: unknown;
    const result = await structured(model, z.object({ value: z.number() }), "Return JSON", {
      selfHeal: false,
      stream: {
        enabled: true,
        onData: (event) => {
          if (event.done) {
            doneData = event.snapshot.data;
          }
        },
      },
    });

    expect(result.data).toEqual({ value: 1 });
    expect(doneData).toEqual({ value: 1 });
  });

  test("attribute think tags in dedicated reasoning do not poison the final parse", async () => {
    const model: LLMAdapter = {
      async complete(): Promise<LLMResponse> {
        return { text: '{"value": 1}', reasoning: "<think foo>secret</think>" };
      },
    };

    const result = await structured(model, z.object({ value: z.number() }), "Return JSON", {
      selfHeal: false,
    });

    expect(result.data).toEqual({ value: 1 });
    expect(result.reasoning).toBe("secret");
  });

  test("random chunk splits render identically at every prefix", () => {
    const payload = '<think>plan {"decoy": true}</think>{"items": [{"id": 1, "label": "a"}, {"id": 2}]}';
    for (let cut = 0; cut <= payload.length; cut += 1) {
      const normalized = normalizeModelOutput(payload.slice(0, cut), "reasoning so far");
      expect(parseStreamingStructuredData(normalized.text)).toEqual(
        referenceParseStreamingStructuredData(normalized.parseSource),
      );
    }
  });
});
