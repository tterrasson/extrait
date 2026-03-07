import { describe, expect, test } from "bun:test";
import { createOpenAICompatibleAdapter } from "@/providers/openai-compatible";
import type { MCPToolClient, LLMStreamChunk } from "@/types";

function sseResponse(events: string[]): Response {
  const body = events.map((e) => `data: ${e}\n\n`).join("");
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function jsonResponse(payload: Record<string, unknown>): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function createSimpleMCP(): MCPToolClient {
  return {
    id: "calc",
    async listTools() {
      return {
        tools: [
          {
            name: "add",
            description: "Add two numbers",
            inputSchema: { type: "object", properties: { a: { type: "number" }, b: { type: "number" } } },
          },
        ],
      };
    },
    async callTool(params) {
      const args = params.arguments ?? {};
      return { result: (args.a as number) + (args.b as number) };
    },
  };
}

describe("openai-compatible streaming", () => {
  test("streams SSE chunks with token and chunk callbacks", async () => {
    const tokens: string[] = [];
    const chunks: LLMStreamChunk[] = [];
    let started = false;
    let completed = false;

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "Hello" } }] }),
        JSON.stringify({ choices: [{ delta: { content: " world" } }] }),
        JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "Say hello" },
      {
        onStart: () => (started = true),
        onToken: (t) => tokens.push(t),
        onChunk: (c) => chunks.push(c),
        onComplete: () => (completed = true),
      },
    );

    expect(started).toBe(true);
    expect(completed).toBe(true);
    expect(tokens).toEqual(["Hello", " world"]);
    expect(result.text).toBe("Hello world");
    expect(result.finishReason).toBe("stop");
    expect(result.usage?.inputTokens).toBe(5);
    expect(result.usage?.outputTokens).toBe(2);
    expect(chunks.length).toBe(3);
  });

  test("ignores [DONE] sentinel in text", async () => {
    const tokens: string[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "ok" } }] }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onToken: (t) => tokens.push(t) },
    );

    expect(result.text).toBe("ok");
    expect(tokens).toEqual(["ok"]);
  });

  test("throws on HTTP error during streaming", async () => {
    const fetcher = (async () =>
      new Response("Server Error", { status: 500 })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("HTTP 500");
  });

  test("streams through MCP rounds and keeps result.text as final assistant text", async () => {
    let startedCount = 0;
    let completed = false;
    const tokens: string[] = [];
    const chunks: LLMStreamChunk[] = [];
    let round = 0;

    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      const bodyParsed = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      round += 1;

      if (round === 1) {
        return sseResponse([
          JSON.stringify({ choices: [{ delta: { content: "I will calculate. " } }] }),
          JSON.stringify({
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      id: "call_add",
                      type: "function",
                      function: { name: "add", arguments: "{\"a\":2" },
                    },
                  ],
                },
              },
            ],
          }),
          JSON.stringify({
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      function: { arguments: ",\"b\":3}" },
                    },
                  ],
                },
                finish_reason: "tool_calls",
              },
            ],
            usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
          }),
          "[DONE]",
        ]);
      }

      const messages = Array.isArray(bodyParsed.messages) ? bodyParsed.messages : [];
      expect(messages.some((entry) => (entry as { role?: string }).role === "tool")).toBe(true);
      return sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "The result is " } }] }),
        JSON.stringify({ choices: [{ delta: { content: "5" } }] }),
        JSON.stringify({
          choices: [{ delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
        }),
        "[DONE]",
      ]);
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        mcpClients: [createSimpleMCP()],
      },
      {
        onStart: () => (startedCount += 1),
        onToken: (t) => tokens.push(t),
        onChunk: (chunk) => chunks.push(chunk),
        onComplete: () => (completed = true),
      },
    );

    expect(startedCount).toBe(1);
    expect(completed).toBe(true);
    expect(tokens).toEqual(["I will calculate. ", "The result is ", "5"]);
    expect(result.text).toBe("The result is 5");
    expect(result.toolCalls?.[0]).toMatchObject({
      id: "call_add",
      name: "add",
      output: { result: 5 },
    });
    expect(result.toolExecutions?.[0]).toMatchObject({
      callId: "call_add",
      name: "add",
      clientId: "calc",
    });
    expect(chunks.some((chunk) => chunk.finishReason === "tool_calls")).toBe(true);
  });

  test("responses API stream does not call onToken when no text delta is emitted", async () => {
    const tokens: string[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          type: "response.completed",
          response: {
            id: "resp_empty",
            status: "completed",
            output_text: "",
          },
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    await adapter.stream!({ prompt: "test" }, { onToken: (t) => tokens.push(t) });

    expect(tokens).toEqual([]);
  });

  test("streams text deltas for responses API path", async () => {
    const tokens: string[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ type: "response.output_text.delta", delta: "from " }),
        JSON.stringify({ type: "response.output_text.delta", delta: "responses api" }),
        JSON.stringify({
          type: "response.completed",
          response: {
            id: "resp_1",
            status: "completed",
            output_text: "from responses api",
            usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5 },
          },
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onToken: (t) => tokens.push(t) },
    );

    expect(result.text).toBe("from responses api");
    expect(tokens).toEqual(["from ", "responses api"]);
  });

  test("streams responses API MCP rounds and keeps result.text as final text", async () => {
    let round = 0;
    const tokens: string[] = [];
    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      const bodyParsed = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      round += 1;

      if (round === 1) {
        return sseResponse([
          JSON.stringify({ type: "response.output_text.delta", delta: "Let me compute. " }),
          JSON.stringify({
            type: "response.output_item.added",
            item: {
              type: "function_call",
              call_id: "call_sum",
              name: "add",
              arguments: "{\"a\":7",
            },
          }),
          JSON.stringify({
            type: "response.function_call_arguments.delta",
            call_id: "call_sum",
            delta: ",\"b\":9}",
          }),
          JSON.stringify({
            type: "response.completed",
            response: {
              id: "resp_1",
              status: "requires_action",
              usage: { input_tokens: 4, output_tokens: 2, total_tokens: 6 },
            },
          }),
          "[DONE]",
        ]);
      }

      expect(bodyParsed.previous_response_id).toBe("resp_1");
      const inputItems = Array.isArray(bodyParsed.input) ? bodyParsed.input : [];
      expect((inputItems[0] as { type?: string }).type).toBe("function_call_output");

      return sseResponse([
        JSON.stringify({ type: "response.output_text.delta", delta: "16" }),
        JSON.stringify({
          type: "response.completed",
          response: {
            id: "resp_2",
            status: "completed",
            output_text: "16",
            usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
          },
        }),
        "[DONE]",
      ]);
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        mcpClients: [createSimpleMCP()],
      },
      {
        onToken: (t) => tokens.push(t),
      },
    );

    expect(tokens).toEqual(["Let me compute. ", "16"]);
    expect(result.text).toBe("16");
    expect(result.toolCalls?.[0]).toMatchObject({
      id: "call_sum",
      name: "add",
      output: { result: 16 },
    });
    expect(result.toolExecutions?.[0]).toMatchObject({
      callId: "call_sum",
      name: "add",
      clientId: "calc",
    });
  });
});

describe("openai-compatible text extraction", () => {
  test("pickResponsesText from output_text", async () => {
    const fetcher = (async () =>
      jsonResponse({
        output_text: "direct output",
        status: "completed",
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.text).toBe("direct output");
  });

  test("pickResponsesText from output content array", async () => {
    const fetcher = (async () =>
      jsonResponse({
        output: [
          {
            type: "message",
            content: [
              { type: "output_text", text: "hello " },
              { type: "output_text", text: "world" },
            ],
          },
        ],
        status: "completed",
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.text).toBe("hello world");
  });

  test("pickResponsesText from output item text", async () => {
    const fetcher = (async () =>
      jsonResponse({
        output: [
          { type: "message", text: "from text field" },
        ],
        status: "completed",
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.text).toBe("from text field");
  });

  test("pickResponsesText with output_text in content parts", async () => {
    const fetcher = (async () =>
      jsonResponse({
        output: [
          {
            type: "message",
            content: [
              { type: "output_text", output_text: "via output_text part" },
            ],
          },
        ],
        status: "completed",
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.text).toBe("via output_text part");
  });

  test("pickAssistantText with array content", async () => {
    const fetcher = (async () =>
      jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: [
                { type: "text", text: "part1" },
                "part2",
                { type: "image" },
              ],
            },
          },
        ],
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.text).toBe("part1part2");
  });

  test("pickAssistantText fallback to legacy text field", async () => {
    const fetcher = (async () =>
      jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            text: "legacy text",
            message: { role: "assistant" },
          },
        ],
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.text).toBe("legacy text");
  });

  test("pickResponsesFinishReason from status", async () => {
    const fetcher = (async () =>
      jsonResponse({
        output_text: "done",
        status: "completed",
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.finishReason).toBe("completed");
  });

  test("HTTP error in passthrough mode", async () => {
    const fetcher = (async () =>
      new Response("Bad Request", { status: 400 })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    await expect(adapter.complete({ prompt: "test" })).rejects.toThrow("HTTP 400");
  });

  test("streaming SSE with array delta content (multimodal)", async () => {
    const tokens: string[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          choices: [{
            delta: {
              content: [
                { type: "text", text: "hello " },
                { type: "text", text: "there" },
              ],
            },
          }],
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onToken: (t) => tokens.push(t) },
    );

    expect(result.text).toBe("hello there");
    expect(tokens).toEqual(["hello there"]);
  });
});
