import { expect, test } from "bun:test";
import { createOpenAICompatibleAdapter } from "../src/providers/openai-compatible";
import { createOpenAICompatibleLegacyAdapter } from "../src/providers/openai-compatible-legacy";

const NETWORK_CHUNK_SIZE = 97;
const ARGUMENT_DELTA_SIZE = 333;

/** Wraps a fixed byte payload into a `fetch` that streams it back in small chunks. */
function createChunkedFetcher(bytes: Uint8Array): typeof fetch {
  const fetcher = async () =>
    new Response(
      new ReadableStream({
        start(controller) {
          for (let i = 0; i < bytes.length; i += NETWORK_CHUNK_SIZE) {
            controller.enqueue(bytes.slice(i, i + NETWORK_CHUNK_SIZE));
          }
          controller.close();
        },
      }),
    );

  return fetcher as unknown as typeof fetch;
}

/** Wraps a fixed body into a `fetch` that returns it in one piece. */
function createStaticFetcher(body: string): typeof fetch {
  const fetcher = async () => new Response(body);

  return fetcher as unknown as typeof fetch;
}

function encodeSSE(events: unknown[]): Uint8Array {
  const payload = events.map((event) => `data: ${JSON.stringify(event)}\r\n\r\n`).join("");

  return new TextEncoder().encode(payload);
}

for (const api of ["responses", "chat"] as const) {
  test(`${api}: empty arguments stay empty, then a large write grows without lost bytes`, async () => {
    const args = JSON.stringify({
      path: "large.ts",
      content: ' é🙂\\"\n'.repeat(20_000),
    });

    const item = {
      type: "function_call",
      id: "fc_1",
      call_id: "call_1",
      name: "write",
      arguments: "",
    };

    const events: unknown[] =
      api === "responses"
        ? [{ type: "response.output_item.added", item }]
        : [
            {
              choices: [
                {
                  delta: {
                    tool_calls: [
                      {
                        index: 0,
                        id: "call_1",
                        type: "function",
                        function: { name: "write", arguments: "" },
                      },
                    ],
                  },
                },
              ],
            },
          ];

    for (let i = 0; i < args.length; i += ARGUMENT_DELTA_SIZE) {
      const delta = args.slice(i, i + ARGUMENT_DELTA_SIZE);

      events.push(
        api === "responses"
          ? { type: "response.function_call_arguments.delta", item_id: "fc_1", delta }
          : {
              choices: [
                { delta: { tool_calls: [{ index: 0, function: { arguments: delta } }] } },
              ],
            },
      );
    }

    events.push(
      api === "responses"
        ? {
            type: "response.completed",
            response: { status: "completed", output: [{ ...item, arguments: args }] },
          }
        : { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
    );

    const factory =
      api === "responses" ? createOpenAICompatibleAdapter : createOpenAICompatibleLegacyAdapter;

    const adapter = factory({
      baseURL: "http://test.invalid",
      model: "test",
      fetcher: createChunkedFetcher(encodeSSE(events)),
    });

    const snapshots: unknown[] = [];

    const result = await adapter.stream!(
      { prompt: "write" },
      {
        onChunk(chunk) {
          if (chunk.toolCalls) snapshots.push(chunk.toolCalls[0]?.arguments);
        },
      },
    );

    expect(snapshots[0]).toBe("");
    expect(snapshots.length).toBeGreaterThan(100);

    for (const snapshot of snapshots) {
      expect(args.startsWith(String(snapshot))).toBe(true);
    }

    expect(result.toolCalls?.[0]?.arguments).toBe(args);
  });
}

const INCOMPLETE_REASONS = [
  ["max_output_tokens", "length"],
  ["content_filter", "content_filter"],
] as const;

for (const [reason, expected] of INCOMPLETE_REASONS) {
  test(`Responses preserves incomplete reason ${reason} in stream and complete`, async () => {
    const payload = {
      status: "incomplete",
      incomplete_details: { reason },
      output: [
        {
          type: "function_call",
          id: "fc_1",
          call_id: "call_1",
          name: "write",
          arguments: '{"content":"partial',
        },
      ],
    };

    for (const streaming of [false, true]) {
      const body = streaming
        ? `data: ${JSON.stringify({ type: "response.incomplete", response: payload })}\n\n`
        : JSON.stringify(payload);

      const adapter = createOpenAICompatibleAdapter({
        baseURL: "http://test.invalid",
        model: "test",
        fetcher: createStaticFetcher(body),
      });

      const result = await (streaming
        ? adapter.stream!({ prompt: "write" })
        : adapter.complete({ prompt: "write" }));

      expect(result.finishReason).toBe(expected);
      expect(result.toolCalls?.[0]?.arguments).toBe('{"content":"partial');
    }
  });
}
