import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { createStreamingStructuredParser } from "@/structured-streaming";
import { normalizeModelOutput } from "@/generate-output";
import { structured } from "@/structured";
import type {
  LLMAdapter,
  LLMRequest,
  LLMResponse,
  LLMStreamCallbacks,
  StructuredStreamEvent,
} from "@/types";

/** Replays `raw` one character at a time and returns the preview at each step. */
function previewAtEveryPrefix(raw: string, reasoning?: string): unknown[] {
  const parser = createStreamingStructuredParser();
  const previews: unknown[] = [];
  for (let cut = 0; cut <= raw.length; cut += 1) {
    const normalized = normalizeModelOutput(raw.slice(0, cut), reasoning);
    previews.push(parser.update(normalized.text));
  }
  return previews;
}

/** Feeds `raw` in randomly sized chunks and returns the final preview. */
function previewWithRandomChunks(raw: string, random: () => number): unknown {
  const parser = createStreamingStructuredParser();
  let sent = 0;
  let last: unknown = null;
  while (sent < raw.length) {
    sent = Math.min(raw.length, sent + 1 + Math.floor(random() * 12));
    last = parser.update(normalizeModelOutput(raw.slice(0, sent)).text);
  }
  return last;
}

/** Deterministic PRNG so a failing seed can be replayed. */
function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function randomJsonValue(random: () => number, depth: number): unknown {
  const roll = random();
  if (depth >= 4 || roll < 0.3) {
    const leaf = random();
    if (leaf < 0.2) return null;
    if (leaf < 0.4) return random() < 0.5;
    if (leaf < 0.7) return Math.round((random() - 0.5) * 20_000) / 100;
    return randomString(random);
  }
  if (roll < 0.65) {
    const size = Math.floor(random() * 4);
    const out: Record<string, unknown> = {};
    for (let index = 0; index < size; index += 1) {
      out[randomString(random)] = randomJsonValue(random, depth + 1);
    }
    return out;
  }
  const size = Math.floor(random() * 4);
  return Array.from({ length: size }, () => randomJsonValue(random, depth + 1));
}

/** Includes the characters that stress escaping and the root scanner. */
function randomString(random: () => number): string {
  const alphabet = ['a', 'z', ' ', '"', "\\", "{", "}", "[", "]", ":", ",", "é", "\n", "\t", "0"];
  const size = Math.floor(random() * 8);
  let out = "";
  for (let index = 0; index < size; index += 1) {
    out += alphabet[Math.floor(random() * alphabet.length)];
  }
  return out;
}

describe("streaming preview parser", () => {
  test("renders values as they arrive", () => {
    const previews = previewAtEveryPrefix('{"sentiment": "POSITIVE", "score": 0.8}');
    expect(previews[0]).toBeNull();
    expect(previews[1]).toEqual({});
    // The key is committed but its value has not started.
    expect(previews[13]).toEqual({ sentiment: null });
    // Strings stream through character by character.
    expect(previews[18]).toEqual({ sentiment: "POS" });
    expect(previews.at(-1)).toEqual({ sentiment: "POSITIVE", score: 0.8 });
  });

  test("types partial literals instead of exposing them as text", () => {
    const previews = previewAtEveryPrefix('{"ok": true}');
    expect(previews[9]).toEqual({ ok: true }); // "tr"
    expect(previews.at(-1)).toEqual({ ok: true });
  });

  test("skips a decoy brace quoted in the prose prefix", () => {
    const previews = previewAtEveryPrefix('The token "{" is special. {"value": 12}');
    expect(previews.at(-1)).toEqual({ value: 12 });
  });

  test("never previews a brace that only appears inside quoted prose", () => {
    const raw = '"draft: {not the payload}" {"value": 1}';
    const rootAt = raw.indexOf('{"value"');
    const previews = previewAtEveryPrefix(raw);
    // Nothing at all while the decoy brace is the only one seen.
    for (const preview of previews.slice(0, rootAt + 1)) {
      expect(preview).toBeNull();
    }
    expect(previews.at(-1)).toEqual({ value: 1 });
  });

  test("ignores prose trailing the closed root", () => {
    const previews = previewAtEveryPrefix('{"value": 1} and that is the answer.');
    expect(previews.at(-1)).toEqual({ value: 1 });
  });

  test("infers a comma the model dropped", () => {
    const previews = previewAtEveryPrefix('{"a": 1 "b": 2, "c": [1 2]}');
    expect(previews.at(-1)).toEqual({ a: 1, b: 2, c: [1, 2] });
  });

  test("holds the last good preview on syntax it cannot advance past", () => {
    const previews = previewAtEveryPrefix('{"a": 1, }}}: nonsense');
    expect(previews.at(-1)).toEqual({ a: 1 });
  });

  test("resets when think sanitization rewrites the prefix", () => {
    const previews = previewAtEveryPrefix('prefix </think> {"value": 3}');
    expect(previews.at(-1)).toEqual({ value: 3 });
  });

  test("ignores a decoy root inside a think block", () => {
    const previews = previewAtEveryPrefix('<think>plan {"decoy": 1}</think>{"real": 2}');
    expect(previews.at(-1)).toEqual({ real: 2 });
  });

  test("finds the root in the reasoning channel only through the visible text", () => {
    const previews = previewAtEveryPrefix('{"value": 5}', "chain of thought");
    expect(previews.at(-1)).toEqual({ value: 5 });
  });

  test("decodes escapes and unicode", () => {
    const raw = '{"s": "a \\"q\\" b \\u00e9", "path": "C:\\\\tmp"}';
    expect(previewAtEveryPrefix(raw).at(-1)).toEqual({ s: 'a "q" b é', path: "C:\\tmp" });
  });

  test("never exposes an escape half written", () => {
    // Cutting inside `\u00e9` must not leak a `\u00` fragment into the preview.
    for (const preview of previewAtEveryPrefix('{"s": "x\\u00e9y"}')) {
      const value = (preview as { s?: string } | null)?.s;
      if (typeof value === "string") {
        expect(value).not.toContain("\\");
      }
    }
  });

  test("does not mistake an escaped backslash for a partial unicode escape", () => {
    // The tail `\\u12` is an escaped backslash followed by text, not a
    // half-written `\uXXXX`; stripping it used to leak raw escapes.
    const raw = '{"s": "x\\\\u12y"}';
    const previews = previewAtEveryPrefix(raw);
    expect(previews[13]).toEqual({ s: "x\\u12" }); // cut right after `x\\u12`
    expect(previews.at(-1)).toEqual({ s: "x\\u12y" });
  });

  test("renders the longest numeric prefix of a growing number", () => {
    const previews = previewAtEveryPrefix('{"n": -12.5e2}');
    expect(previews[10]).toEqual({ n: -12 }); // `-12.`
    expect(previews[12]).toEqual({ n: -12.5 }); // `-12.5e`
    expect(previews.at(-1)).toEqual({ n: -1250 });
  });

  test("resets when a rewrite lands inside the string open at the tail", () => {
    // Only the prefix up to the parse cursor used to be verified; a rewrite
    // in the already-scanned body of an open string kept stale scan state.
    const parser = createStreamingStructuredParser();
    parser.update('{"a": "hello');
    expect(parser.update('{"a": "hi", "b": 1}')).toEqual({ a: "hi", b: 1 });
  });

  test("snapshots are immutable: an earlier preview is not mutated by later ones", () => {
    const parser = createStreamingStructuredParser();
    parser.update('{"items": [1');
    const early = parser.update('{"items": [1, 2') as { items: number[] };
    expect(early).toEqual({ items: [1, 2] });
    parser.update('{"items": [1, 2, 3, 4], "done": true}');
    expect(early).toEqual({ items: [1, 2] });
  });

  test("completed subtrees keep their identity across events", () => {
    const parser = createStreamingStructuredParser();
    parser.update('{"a": {"x": 1}, "b": [1');
    const first = parser.update('{"a": {"x": 1}, "b": [1, 2') as { a: unknown };
    const second = parser.update('{"a": {"x": 1}, "b": [1, 2, 3') as { a: unknown };
    expect(second.a).toBe(first.a);
  });
});

describe("streaming preview fuzz", () => {
  test("every prefix of a random document yields a prefix-consistent preview", () => {
    const random = makeRandom(0x5eed);
    for (let iteration = 0; iteration < 200; iteration += 1) {
      const document = randomJsonValue(random, 0);
      if (typeof document !== "object" || document === null) {
        continue;
      }
      const raw = JSON.stringify(document);
      const previews = previewAtEveryPrefix(raw);
      // The preview must converge on the real value, never throw, and only ever
      // be null or an object along the way.
      expect(previews.at(-1)).toEqual(document);
      for (const preview of previews) {
        expect(preview === null || typeof preview === "object").toBe(true);
      }
    }
  });

  test("chunk boundaries do not change the result", () => {
    const random = makeRandom(0xc0ffee);
    for (let iteration = 0; iteration < 200; iteration += 1) {
      const document = randomJsonValue(random, 0);
      if (typeof document !== "object" || document === null) {
        continue;
      }
      const raw = JSON.stringify(document);
      expect(previewWithRandomChunks(raw, random)).toEqual(document);
    }
  });

  test("pretty-printed documents parse the same as compact ones", () => {
    const random = makeRandom(0xbeef);
    for (let iteration = 0; iteration < 100; iteration += 1) {
      const document = randomJsonValue(random, 0);
      if (typeof document !== "object" || document === null) {
        continue;
      }
      expect(previewAtEveryPrefix(JSON.stringify(document, null, 2)).at(-1)).toEqual(document);
      expect(previewAtEveryPrefix(JSON.stringify(document)).at(-1)).toEqual(document);
    }
  });

  test("prose around a random document does not break the root search", () => {
    const random = makeRandom(0x1234);
    for (let iteration = 0; iteration < 100; iteration += 1) {
      const document = randomJsonValue(random, 0);
      if (typeof document !== "object" || document === null) {
        continue;
      }
      const raw = `Voici la réponse: ${JSON.stringify(document)}\nVoilà.`;
      expect(previewAtEveryPrefix(raw).at(-1)).toEqual(document);
    }
  });

  test("corrupted documents never throw and never stop making progress", () => {
    const random = makeRandom(0xdead);
    const noise = ['{', '}', '[', ']', '"', ':', ',', "\\", "e", "-", " ", " "];
    for (let iteration = 0; iteration < 500; iteration += 1) {
      const document = randomJsonValue(random, 0);
      if (typeof document !== "object" || document === null) {
        continue;
      }
      let raw = JSON.stringify(document);
      // Apply a handful of random corruptions.
      const edits = 1 + Math.floor(random() * 5);
      for (let edit = 0; edit < edits && raw.length > 1; edit += 1) {
        const at = Math.floor(random() * raw.length);
        const roll = random();
        if (roll < 0.4) {
          raw = raw.slice(0, at) + raw.slice(at + 1);
        } else if (roll < 0.8) {
          raw = raw.slice(0, at) + noise[Math.floor(random() * noise.length)] + raw.slice(at);
        } else {
          raw = raw.slice(0, at) + raw.slice(at, at + 3) + raw.slice(at);
        }
      }
      const parser = createStreamingStructuredParser();
      for (let cut = 0; cut <= raw.length; cut += 1) {
        // A hang inside resume() would surface as a test timeout.
        const preview = parser.update(raw.slice(0, cut));
        expect(preview === null || typeof preview === "object").toBe(true);
      }
    }
  });

  test("deep nesting does not overflow the stack", () => {
    const depth = 5_000;
    const raw = `${'{"n":'.repeat(depth)}1${"}".repeat(depth)}`;
    const parser = createStreamingStructuredParser();
    expect(() => parser.update(raw)).not.toThrow();
    let cursor = parser.update(raw) as Record<string, unknown>;
    for (let level = 0; level < depth - 1; level += 1) {
      cursor = cursor.n as Record<string, unknown>;
    }
    expect(cursor.n).toBe(1);
  });

  test("a value never regresses once its container has closed", () => {
    const random = makeRandom(0xabcd);
    for (let iteration = 0; iteration < 100; iteration += 1) {
      const document = randomJsonValue(random, 0);
      if (typeof document !== "object" || document === null) {
        continue;
      }
      const raw = JSON.stringify(document);
      const parser = createStreamingStructuredParser();
      let previousKeys = 0;
      for (let cut = 0; cut <= raw.length; cut += 1) {
        const preview = parser.update(raw.slice(0, cut));
        if (preview && !Array.isArray(preview)) {
          const keys = Object.keys(preview).length;
          // The top-level object only ever gains fields as text arrives.
          expect(keys).toBeGreaterThanOrEqual(previousKeys);
          previousKeys = keys;
        }
      }
    }
  });

  test("truncating anywhere never throws and stays a prefix of the whole", () => {
    const random = makeRandom(0xfeed);
    for (let iteration = 0; iteration < 100; iteration += 1) {
      const document = randomJsonValue(random, 0);
      if (typeof document !== "object" || document === null) {
        continue;
      }
      const raw = JSON.stringify(document);
      const cut = Math.floor(random() * raw.length);
      const parser = createStreamingStructuredParser();
      expect(() => parser.update(raw.slice(0, cut))).not.toThrow();
    }
  });
});

describe("streaming preview integration", () => {
  test("stream.dataInterval coalesces recomputation; done always recomputes", async () => {
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
    expect(events[0]?.snapshot.data).toEqual({ a: 1 } as never);
    expect(events[1]?.snapshot.data).toEqual({ a: 1 } as never);
    expect(events[2]?.snapshot.data).toEqual({ a: 1 } as never);
    expect(events[3]?.done).toBe(true);
    expect(events[3]?.snapshot.data).toEqual({ a: 1, b: 2 });
  });

  test("default dataInterval stays exact on small outputs and coalesces large ones", async () => {
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

    // Longer than AUTO_DATA_INTERVAL_EXACT_MAX_CHARS, to land in coalesced mode.
    const pad = "x".repeat(9000);
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
    // Reused by reference: no recomputation happened for the intermediate events.
    expect(largeEvents[1]?.snapshot.data).toBe(largeEvents[0]?.snapshot.data as never);
    expect(largeEvents[2]?.snapshot.data).toBe(largeEvents[0]?.snapshot.data as never);
    expect(largeEvents[3]?.done).toBe(true);
    expect(largeEvents[3]?.snapshot.data).toEqual({ pad, a: 1, b: 2 });
  });

  test("attribute think tags cannot make the done snapshot diverge from the final parse", async () => {
    // `<think foo>` is recognized by the scanner; unbalanced nesting used to
    // make sanitization swallow the visible JSON, so the done event announced
    // data that the final parse then rejected.
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
});
