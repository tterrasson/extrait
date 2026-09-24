import { describe, expect, test } from "bun:test";
import { generate } from "@/generate";
import { normalizeModelOutput, withoutTrailingThinkTagPrefix } from "@/generate-output";
import { createStreamNormalizer } from "@/stream-normalizer";
import type { LLMAdapter, LLMStreamCallbacks } from "@/types";

interface Chunk {
  text?: string;
  reasoning?: string;
}

interface ObservedEvent {
  text: string;
  reasoning: string;
  deltaText: string;
  deltaReasoning: string;
  resync?: { text: boolean; reasoning: boolean };
}

// The pre-optimization algorithm: renormalize everything on each chunk and
// diff the stable snapshots by string comparison.
function reference(chunks: Chunk[]): ObservedEvent[] {
  const events: ObservedEvent[] = [];
  let text = "";
  let reasoning = "";
  let lastText: string | undefined;
  let lastReasoning: string | undefined;
  let previousText = "";
  let previousReasoning = "";

  const emit = (done: boolean): void => {
    const normalized = normalizeModelOutput(text, reasoning);
    if (!done && lastText !== undefined && normalized.text === lastText && normalized.reasoning === lastReasoning) {
      return;
    }
    const stableText = done ? normalized.text : withoutTrailingThinkTagPrefix(normalized.text);
    const stableReasoning = done ? normalized.reasoning : withoutTrailingThinkTagPrefix(normalized.reasoning);
    // A retraction reports the whole stable value again, flagged as a resync.
    const resync = { text: !stableText.startsWith(previousText), reasoning: !stableReasoning.startsWith(previousReasoning) };
    events.push({
      text: normalized.text,
      reasoning: normalized.reasoning,
      deltaText: resync.text ? stableText : stableText.slice(previousText.length),
      deltaReasoning: resync.reasoning ? stableReasoning : stableReasoning.slice(previousReasoning.length),
      ...(resync.text || resync.reasoning ? { resync } : {}),
    });
    lastText = normalized.text;
    lastReasoning = normalized.reasoning;
    previousText = stableText;
    previousReasoning = stableReasoning;
  };

  for (const chunk of chunks) {
    if (chunk.text) {
      text += chunk.text;
      emit(false);
    }
    if (chunk.reasoning) {
      reasoning += chunk.reasoning;
      emit(false);
    }
  }
  emit(true);
  return events;
}

async function actual(chunks: Chunk[]): Promise<ObservedEvent[]> {
  const events: ObservedEvent[] = [];
  const adapter: LLMAdapter = {
    async complete() {
      throw new Error("unused");
    },
    async stream(_request, callbacks: LLMStreamCallbacks = {}) {
      let text = "";
      let reasoning = "";
      for (const chunk of chunks) {
        text += chunk.text ?? "";
        reasoning += chunk.reasoning ?? "";
        callbacks.onChunk?.({ textDelta: chunk.text ?? "", reasoningDelta: chunk.reasoning });
      }
      return { text, reasoning };
    },
  };
  await generate(adapter, "go", {
    stream: {
      onData: (event) => {
        events.push({
          text: event.snapshot.text,
          reasoning: event.snapshot.reasoning,
          deltaText: event.delta.text,
          deltaReasoning: event.delta.reasoning,
          ...(event.resync ? { resync: event.resync } : {}),
        });
      },
    },
  });
  return events;
}

function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

const PIECES = [
  "hello ", "world", "<think>", "</think>", "<THINK>", "</ Think >", "<think a=\">\">", "<", "/", "t", "h", "i",
  "n", "k", ">", " ", "\n", "{\"a\":1}", "<b>", "a < b", "</think", "<thinking>", "x",
];

function randomChunks(next: () => number): Chunk[] {
  const source = Array.from({ length: 4 + Math.floor(next() * 30) }, () => PIECES[Math.floor(next() * PIECES.length)]).join("");
  const reasoningSource = next() < 0.4
    ? Array.from({ length: Math.floor(next() * 12) }, () => PIECES[Math.floor(next() * PIECES.length)]).join("")
    : "";
  const chunks: Chunk[] = [];
  let textCursor = 0;
  let reasoningCursor = 0;
  while (textCursor < source.length || reasoningCursor < reasoningSource.length) {
    if (reasoningCursor < reasoningSource.length && (next() < 0.5 || textCursor >= source.length)) {
      const size = 1 + Math.floor(next() * 5);
      chunks.push({ reasoning: reasoningSource.slice(reasoningCursor, reasoningCursor + size) });
      reasoningCursor += size;
    } else {
      const size = 1 + Math.floor(next() * 6);
      chunks.push({ text: source.slice(textCursor, textCursor + size) });
      textCursor += size;
    }
  }
  return chunks;
}

describe("streaming normalizer", () => {
  test("emits the same snapshots and deltas as a full renormalization", async () => {
    const next = random(42);
    for (let run = 0; run < 1_500; run += 1) {
      const chunks = randomChunks(next);
      expect({ chunks, events: await actual(chunks) }).toEqual({ chunks, events: reference(chunks) });
    }
  });

  test("long outputs, where the delta window slides, match too", async () => {
    const next = random(3);
    for (let run = 0; run < 150; run += 1) {
      const chunks = [...randomChunks(next), ...randomChunks(next), ...randomChunks(next), ...randomChunks(next)];
      expect({ chunks, events: await actual(chunks) }).toEqual({ chunks, events: reference(chunks) });
    }
  });

  test("every update renders what normalizeModelOutput renders", () => {
    const next = random(7);
    for (let run = 0; run < 3_000; run += 1) {
      const normalizer = createStreamNormalizer();
      let text = "";
      let reasoning = "";
      let previous = { text: "", reasoning: "" };
      for (const chunk of randomChunks(next)) {
        text += chunk.text ?? "";
        reasoning += chunk.reasoning ?? "";
        const update = normalizer.push(chunk);
        const expected = normalizeModelOutput(text, reasoning);
        expect([update.text, update.reasoning]).toEqual([expected.text, expected.reasoning]);
        expect(update.changed).toEqual({
          text: expected.text !== previous.text,
          reasoning: expected.reasoning !== previous.reasoning,
        });
        expect(update.textExtends).toBe(expected.text.startsWith(previous.text));
        previous = { text: expected.text, reasoning: expected.reasoning };
      }
      const final = normalizer.finish();
      expect([final.text, final.reasoning]).toEqual([
        normalizeModelOutput(text, reasoning).text,
        normalizeModelOutput(text, reasoning).reasoning,
      ]);
    }
  });

  test("the retraction of streamed text by a late think tag is a resync", () => {
    const normalizer = createStreamNormalizer();
    expect(normalizer.push({ text: "abc<T" }).delta.text).toBe("abc<T");
    const update = normalizer.push({ text: "HINK>hidden</think>and more" });
    expect(update.text).toBe("abcand more");
    expect(update.textExtends).toBe(false);
    expect(update.resync).toEqual({ text: true, reasoning: false });
    expect(update.delta.text).toBe("abcand more");
  });

  test("folding the deltas always rebuilds the stable output", () => {
    const next = random(11);
    let resyncs = 0;
    for (let run = 0; run < 3_000; run += 1) {
      const normalizer = createStreamNormalizer();
      const view = { text: "", reasoning: "" };
      const fold = (update: ReturnType<typeof normalizer.push>): void => {
        view.text = update.resync.text ? update.delta.text : view.text + update.delta.text;
        view.reasoning = update.resync.reasoning ? update.delta.reasoning : view.reasoning + update.delta.reasoning;
        resyncs += update.resync.text || update.resync.reasoning ? 1 : 0;
      };
      const chunks = [...randomChunks(next), ...randomChunks(next)];
      for (const chunk of chunks) {
        const update = normalizer.push(chunk);
        fold(update);
        expect(view).toEqual({
          text: withoutTrailingThinkTagPrefix(update.text),
          reasoning: withoutTrailingThinkTagPrefix(update.reasoning),
        });
      }
      const final = normalizer.finish();
      fold(final);
      expect(view).toEqual({ text: final.text, reasoning: final.reasoning });
    }
    // Retractions must actually occur, or this proves nothing about them.
    expect(resyncs).toBeGreaterThan(50);
  });

  test("the final text given to finish() is authoritative", () => {
    const normalizer = createStreamNormalizer();
    normalizer.push({ text: "Hel" });
    const final = normalizer.finish({ text: "Hello!" });
    expect(final.text).toBe("Hello!");
    expect(final.delta.text).toBe("lo!");
  });

  test("a consumer of deltas alone stays linear in the output size", () => {
    const run = (chunks: number): number => {
      const normalizer = createStreamNormalizer();
      const piece = "lorem ipsum <b>x</b> ";
      let total = 0;
      const started = performance.now();
      normalizer.push({ reasoning: "planning" });
      normalizer.push({ text: "<think>inline plan</think>" });
      for (let index = 0; index < chunks; index += 1) {
        total += normalizer.push({ text: piece }).delta.text.length;
      }
      total += normalizer.finish().delta.text.length;
      expect(total).toBe(chunks * piece.length);
      return performance.now() - started;
    };
    run(20_000);
    const small = run(40_000);
    const large = run(160_000);
    // 4x the output: linear is ~4x the time, quadratic ~16x.
    expect(large / small).toBeLessThan(8);
  });
});
