import { describe, expect, test } from "bun:test";
import { createOpenAICompatibleAdapter } from "../src/providers/openai-compatible";
import type { MCPToolClient, LLMStreamChunk } from "../src/types";

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

  test("delegates to streamViaComplete when MCP clients are present", async () => {
    let started = false;
    let completed = false;
    const tokens: string[] = [];

    const fetcher = (async () =>
      jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: { role: "assistant", content: "result" },
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      })) as unknown as typeof fetch;

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
        onStart: () => (started = true),
        onToken: (t) => tokens.push(t),
        onComplete: () => (completed = true),
      },
    );

    expect(started).toBe(true);
    expect(completed).toBe(true);
    expect(tokens).toEqual(["result"]);
    expect(result.text).toBe("result");
  });

  test("streamViaComplete does not call onToken for empty text", async () => {
    const tokens: string[] = [];

    const fetcher = (async () =>
      jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: { role: "assistant", content: "" },
          },
        ],
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    await adapter.stream!(
      {
        prompt: "test",
        mcpClients: [createSimpleMCP()],
      },
      { onToken: (t) => tokens.push(t) },
    );

    expect(tokens).toEqual([]);
  });

  test("delegates to streamViaComplete for responses API path", async () => {
    const tokens: string[] = [];

    const fetcher = (async () =>
      jsonResponse({
        output_text: "from responses api",
        status: "completed",
        usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5 },
      })) as unknown as typeof fetch;

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
    expect(tokens).toEqual(["from responses api"]);
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
