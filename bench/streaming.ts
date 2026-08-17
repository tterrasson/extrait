/**
 * Streaming cost harness.
 *
 * Fake adapter emitting a synthetic structured output in N fixed-size chunks,
 * in both generate() and structured() modes. Reports wall time, peak heapUsed
 * and the number of onData events (the non-regression oracle: it must never
 * change across optimization steps).
 *
 * Run: bun run bench:stream
 */
import { z } from "zod";
import { generate } from "../src/generate";
import { structured } from "../src/structured";
import type { LLMAdapter, LLMRequest, LLMResponse, LLMStreamCallbacks } from "../src/types";

const CHUNK_SIZE = 20;
const CHUNK_COUNTS = [500, 2_000, 10_000];

function buildPayload(totalChars: number): string {
  let out = '<think>laying out the plan for the answer</think>{"items":[';
  let id = 0;
  while (out.length < totalChars) {
    out += `${id > 0 ? "," : ""}{"id":${id},"label":"item number ${id} with some padding text"}`;
    id += 1;
  }
  return `${out}]}`;
}

// A single long text field: the preview must walk the open string once overall,
// not re-scan it on every chunk.
function buildLongStringPayload(totalChars: number): string {
  const body = "lorem ipsum dolor sit amet ".repeat(Math.ceil(totalChars / 27));
  return `{"content":"${body.slice(0, totalChars)}","done":true}`;
}

// One flat array of scalars: the worst case for the snapshot copy, which
// duplicates every element of the still-open array on each recomputation.
function buildFlatArrayPayload(totalChars: number): string {
  let out = '{"values":[';
  let id = 0;
  while (out.length < totalChars) {
    out += `${id > 0 ? "," : ""}${id}`;
    id += 1;
  }
  return `${out}]}`;
}

interface BenchResult {
  wallMs: number;
  peakHeapBytes: number;
  events: number;
}

function makeAdapter(payload: string, sampleHeap: () => void): LLMAdapter {
  return {
    async complete(): Promise<LLMResponse> {
      return { text: payload };
    },
    async stream(_request: LLMRequest, callbacks: LLMStreamCallbacks = {}): Promise<LLMResponse> {
      callbacks.onStart?.();
      for (let offset = 0; offset < payload.length; offset += CHUNK_SIZE) {
        callbacks.onChunk?.({ textDelta: payload.slice(offset, offset + CHUNK_SIZE) });
        sampleHeap();
      }
      const response: LLMResponse = { text: payload, finishReason: "stop" };
      callbacks.onComplete?.(response);
      return response;
    },
  };
}

const SCHEMA = z.object({
  items: z.array(z.object({ id: z.number(), label: z.string() })),
});

const LONG_STRING_SCHEMA = z.object({ content: z.string(), done: z.boolean() });

const FLAT_ARRAY_SCHEMA = z.object({ values: z.array(z.number()) });

type Mode =
  | "generate"
  | "structured"
  | "structured+dataInterval0"
  | "structured+longString"
  | "structured+flatArray0";

async function runScenario(mode: Mode, chunks: number): Promise<BenchResult> {
  const longString = mode === "structured+longString";
  const flatArray = mode === "structured+flatArray0";
  const payload = longString
    ? buildLongStringPayload(chunks * CHUNK_SIZE)
    : flatArray
      ? buildFlatArrayPayload(chunks * CHUNK_SIZE)
      : buildPayload(chunks * CHUNK_SIZE);
  let peakHeapBytes = 0;
  let events = 0;
  const sampleHeap = (): void => {
    const used = process.memoryUsage().heapUsed;
    if (used > peakHeapBytes) {
      peakHeapBytes = used;
    }
  };
  const adapter = makeAdapter(payload, sampleHeap);
  const stream = {
    enabled: true,
    ...(mode === "structured+dataInterval0" || longString || flatArray ? { dataInterval: 0 } : {}),
    onData: () => {
      events += 1;
    },
  };

  Bun.gc(true);
  const startedAt = performance.now();
  if (mode === "generate") {
    await generate(adapter, "bench", { stream });
  } else if (longString) {
    await structured(adapter, LONG_STRING_SCHEMA, "bench", { stream });
  } else if (flatArray) {
    await structured(adapter, FLAT_ARRAY_SCHEMA, "bench", { stream });
  } else {
    await structured(adapter, SCHEMA, "bench", { stream });
  }
  const wallMs = performance.now() - startedAt;
  sampleHeap();
  return { wallMs, peakHeapBytes, events };
}

const rows: string[] = [];
for (const mode of [
  "generate",
  "structured",
  "structured+dataInterval0",
  "structured+longString",
  "structured+flatArray0",
] as const) {
  for (const chunks of CHUNK_COUNTS) {
    const result = await runScenario(mode, chunks);
    const row = `| ${mode} | ${chunks} | ${(chunks * CHUNK_SIZE / 1000).toFixed(0)} ko | ${result.wallMs.toFixed(0)} ms | ${(result.peakHeapBytes / 1024 / 1024).toFixed(1)} Mo | ${result.events} |`;
    rows.push(row);
    console.log(row);
  }
}

console.log("\n| Mode | Chunks | Sortie | Temps mur | Heap crête | Événements |");
console.log("|------|--------|--------|-----------|------------|------------|");
for (const row of rows) {
  console.log(row);
}
