import { describe, expect, test } from "bun:test";
import { createOpenAICompatibleAdapter as createResponsesAdapter } from "@/providers/openai-compatible";
import {
  createOpenAICompatibleLegacyAdapter as createOpenAICompatibleAdapter,
} from "@/providers/openai-compatible-legacy";
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
    const requests: Record<string, unknown>[] = [];
    const tokens: string[] = [];
    const chunks: LLMStreamChunk[] = [];
    let started = false;
    let completed = false;

    const fetcher = (async (_input: Request | string | URL, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>);
      return sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "Hello" } }] }),
        JSON.stringify({ choices: [{ delta: { content: " world" } }] }),
        JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } }),
        "[DONE]",
      ]);
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "Say hello", reasoningEffort: "max" },
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
    expect(requests[0]?.reasoning_effort).toBe("xhigh");
    expect(requests[0]?.stream_options).toEqual({ include_usage: true });
  });

  test("normalizes and accumulates streamed content and refusal logprobs", async () => {
    const chunks: LLMStreamChunk[] = [];
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          choices: [{
            delta: { content: "yes" },
            logprobs: {
              content: [{
                token: "yes",
                logprob: -0.1,
                bytes: [121, 101, 115],
                top_logprobs: [
                  { token: "yes", logprob: -0.1, bytes: [121, 101, 115] },
                  { token: "no", logprob: -2.4, bytes: null },
                ],
              }],
            },
          }],
        }),
        JSON.stringify({
          choices: [{
            delta: {},
            finish_reason: "content_filter",
            logprobs: {
              refusal: [{ token: " refusal", logprob: -0.3, bytes: [999] }],
            },
          }],
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });
    const result = await adapter.stream!(
      { prompt: "test", body: { logprobs: true, top_logprobs: 2 } },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    expect(chunks[0]?.logprobs?.content?.[0]).toEqual({
      token: "yes",
      logprob: -0.1,
      bytes: [121, 101, 115],
      top_logprobs: [
        { token: "yes", logprob: -0.1, bytes: [121, 101, 115] },
        { token: "no", logprob: -2.4, bytes: null },
      ],
    });
    expect(result.logprobs).toEqual({
      content: [{
        token: "yes",
        logprob: -0.1,
        bytes: [121, 101, 115],
        top_logprobs: [
          { token: "yes", logprob: -0.1, bytes: [121, 101, 115] },
          { token: "no", logprob: -2.4, bytes: null },
        ],
      }],
      refusal: [{ token: " refusal", logprob: -0.3 }],
    });
  });

  test("requests usage reporting on streamed chat completions and respects caller overrides", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input: Request | string | URL, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>);
      return sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "hi" }, finish_reason: "stop" }] }),
        "[DONE]",
      ]);
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await adapter.stream!({ prompt: "Say hi" });
    expect(requests[0]?.stream_options).toEqual({ include_usage: true });

    await adapter.stream!({
      prompt: "Say hi",
      body: { stream_options: { include_usage: false, continuous_usage_stats: true } },
    });
    expect(requests[1]?.stream_options).toEqual({
      include_usage: false,
      continuous_usage_stats: true,
    });
  });

  test("keeps the latest stream usage snapshot instead of summing chunk usage", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "Hello" } }], usage: { prompt_tokens: 5, total_tokens: 5 } }),
        JSON.stringify({
          choices: [{ delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "Say hello" });

    expect(result.usage).toEqual({
      inputTokens: 5,
      outputTokens: 2,
      totalTokens: 7,
    });
  });

  test("streams reasoning_content separately from visible text", async () => {
    const tokens: string[] = [];
    const chunks: LLMStreamChunk[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { reasoning_content: "Thinking" } }] }),
        JSON.stringify({ choices: [{ delta: { reasoning_content: "..." } }] }),
        JSON.stringify({ choices: [{ delta: { content: "{\"value\":" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "7}" }, finish_reason: "stop" }] }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      {
        onToken: (token) => tokens.push(token),
        onChunk: (chunk) => chunks.push(chunk),
      },
    );

    expect(tokens).toEqual(['{"value":', "7}"]);
    expect(result.text).toBe('{"value":7}');
    expect(result.reasoning).toBe("Thinking...");
    expect(chunks.map((chunk) => chunk.reasoningDelta).filter(Boolean)).toEqual(["Thinking", "..."]);
    expect(chunks.map((chunk) => chunk.textDelta).filter(Boolean)).toEqual(['{"value":', "7}"]);
  });

  test("streams reasoning separately when provider uses reasoning field", async () => {
    const chunks: LLMStreamChunk[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { reasoning: "step 1" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "done" }, finish_reason: "stop" }] }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    expect(result.text).toBe("done");
    expect(result.reasoning).toBe("step 1");
    expect(chunks[0]?.reasoningDelta).toBe("step 1");
    expect(chunks[1]?.textDelta).toBe("done");
  });

  test("keeps reasoning when a stream finishes without visible text", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { reasoning: "silent chain" } }] }),
        JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "test" });

    expect(result.text).toBe("");
    expect(result.reasoning).toBe("silent chain");
    expect(result.finishReason).toBe("stop");
  });

  test("streams chat completion tool calls in pass-through mode", async () => {
    const chunks: LLMStreamChunk[] = [];
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call_lookup",
                    type: "function",
                    function: { name: "lookup", arguments: "{\"q\"" },
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
                    function: { arguments: ":\"x\"}" },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        body: {
          tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object" } } }],
        },
      },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    expect(result.text).toBe("");
    expect(result.finishReason).toBe("tool_calls");
    expect(result.toolCalls).toEqual([
      { id: "call_lookup", type: "function", name: "lookup", arguments: "{\"q\":\"x\"}" },
    ]);
    expect(chunks.some((chunk) => chunk.toolCalls?.[0]?.id === "call_lookup")).toBe(true);
  });

  test("streams chat completion tool calls from final assistant message chunks", async () => {
    const chunks: LLMStreamChunk[] = [];
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          choices: [
            {
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    id: "call_lookup",
                    type: "function",
                    function: { name: "lookup", arguments: "{\"q\":\"x\"}" },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        body: {
          tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object" } } }],
        },
      },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    expect(result.text).toBe("");
    expect(result.finishReason).toBe("tool_calls");
    expect(result.toolCalls).toEqual([
      { id: "call_lookup", type: "function", name: "lookup", arguments: "{\"q\":\"x\"}" },
    ]);
    expect(chunks.some((chunk) => chunk.toolCalls?.[0]?.id === "call_lookup")).toBe(true);
  });

  test("converts native XML tool-call text into toolCalls without streaming raw markup", async () => {
    const chunks: LLMStreamChunk[] = [];
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "Let me check.\n<tool" } }] }),
        JSON.stringify({
          choices: [
            {
              delta: {
                content:
                  "_call>\n<function=bash>\n<parameter=command>\nfind ./src -type f\n</parameter>\n</function>\n",
              },
            },
          ],
        }),
        JSON.stringify({
          choices: [
            {
              delta: { content: "</tool_call>" },
              finish_reason: "tool_calls",
            },
          ],
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        body: {
          tools: [{ type: "function", function: { name: "bash", parameters: { type: "object" } } }],
        },
      },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    expect(result.text).toBe("Let me check.\n");
    expect(result.finishReason).toBe("tool_calls");
    expect(result.toolCalls).toEqual([
      { id: "call_native_0", type: "function", name: "bash", arguments: '{"command":"find ./src -type f"}' },
    ]);
    expect(chunks.map((chunk) => chunk.textDelta).join("")).toBe("Let me check.\n");
    expect(chunks.map((chunk) => chunk.textDelta).join("")).not.toContain("<tool_call>");
    expect(chunks.some((chunk) => chunk.toolCalls?.[0]?.name === "bash")).toBe(true);
  });

  test("parses native JSON <tool_call> blocks emitted as content", async () => {
    const tokens: string[] = [];
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "before <tool_" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "call> after" }, finish_reason: "stop" }] }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        body: { tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object" } } }] },
      },
      { onToken: (token) => tokens.push(token) },
    );

    // The tool-call block is stripped from the visible text and surfaced as a tool call.
    expect(result.text).toBe("before  after");
    expect(tokens.join("")).toBe("before  after");
    expect(result.finishReason).toBe("stop");
    expect(result.toolCalls).toEqual([
      { id: "call_native_0", type: "function", name: "lookup", arguments: "{\"q\":\"x\"}" },
    ]);
  });

  test("parses native XML <tool_call> blocks and preserves parameter types", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          choices: [
            {
              delta: {
                content:
                  "<tool_call><function=search><parameter=query>hello</parameter><parameter=limit>5</parameter><parameter=fuzzy>true</parameter></function></tool_call>",
              },
            },
          ],
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        body: { tools: [{ type: "function", function: { name: "search", parameters: { type: "object" } } }] },
      },
      {},
    );

    expect(result.text).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls?.[0]?.name).toBe("search");
    // String stays a string; number/boolean keep their JSON type.
    expect(JSON.parse(String(result.toolCalls?.[0]?.arguments))).toEqual({
      query: "hello",
      limit: 5,
      fuzzy: true,
    });
  });

  test("preserves incomplete native tool-call markup as text at the end of a completed stream", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          choices: [{
            delta: { content: "Before <tool_call>{\"name\":\"lookup\"" },
            finish_reason: "stop",
          }],
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!({
      prompt: "test",
      body: { tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object" } } }] },
    });

    expect(result.text).toBe("Before <tool_call>{\"name\":\"lookup\"");
    expect(result.toolCalls).toBeUndefined();
  });

  test("does not intercept <tool_call> markup when the request declares no tools", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "Use the <tool_call>{\"name\":\"x\"}</tool_call> format." } }] }),
        JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "test" }, {});

    // No tools declared: the literal markup stays in the text, untouched.
    expect(result.text).toBe("Use the <tool_call>{\"name\":\"x\"}</tool_call> format.");
    expect(result.toolCalls).toBeUndefined();
    expect(result.finishReason).toBe("stop");
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
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onToken: (t) => tokens.push(t) },
    );

    expect(result.text).toBe("ok");
    expect(tokens).toEqual(["ok"]);
  });

  test("streams refusal deltas as response text", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { refusal: "I cannot" } }] }),
        JSON.stringify({ choices: [{ delta: { refusal: " help." }, finish_reason: "content_filter" }] }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "test" });

    expect(result.text).toBe("I cannot help.");
    expect(result.finishReason).toBe("content_filter");
  });

  test("throws when a Chat Completions stream ends before a terminal event", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "partial" } }] }),
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("before a terminal event");
  });

  test("throws for an in-band Chat Completions stream error", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ error: { type: "server_error", message: "Upstream failed" } }),
      ])) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("Upstream failed");
  });

  test("throws for a malformed Chat Completions stream event", async () => {
    const fetcher = (async () => sseResponse(["not-json", "[DONE]"])) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("Invalid JSON event");
  });

  test("throws on HTTP error during streaming", async () => {
    const fetcher = (async () =>
      new Response("Server Error", { status: 500 })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
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
      model: "test-model",
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
    expect(result.usage).toEqual({
      inputTokens: 7,
      outputTokens: 3,
      totalTokens: 10,
    });
    expect(chunks.some((chunk) => chunk.finishReason === "tool_calls")).toBe(true);
  });

  test("streams partial tool-call arguments incrementally in MCP mode", async () => {
    let round = 0;
    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      round += 1;
      if (round === 1) {
        return sseResponse([
          JSON.stringify({
            choices: [
              {
                delta: {
                  tool_calls: [
                    { index: 0, id: "call_add", type: "function", function: { name: "add", arguments: "{\"a\":2" } },
                  ],
                },
              },
            ],
          }),
          JSON.stringify({
            choices: [
              { delta: { tool_calls: [{ index: 0, function: { arguments: ",\"b\":3}" } }] }, finish_reason: "tool_calls" },
            ],
          }),
          "[DONE]",
        ]);
      }
      return sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "5" } }] }),
        JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "[DONE]",
      ]);
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({ baseURL: "https://example.com", model: "test-model", fetcher });

    const chunks: LLMStreamChunk[] = [];
    await adapter.stream!(
      { prompt: "test", mcpClients: [createSimpleMCP()] },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    // The accumulated argument string must surface progressively, not only once
    // the full call is assembled at round end.
    const argSnapshots = chunks
      .map((chunk) => chunk.toolCalls?.[0]?.arguments)
      .filter((value): value is string => typeof value === "string");
    expect(argSnapshots).toContain("{\"a\":2");
    expect(argSnapshots).toContain("{\"a\":2,\"b\":3}");
    // Partial snapshot must appear before the complete one.
    expect(argSnapshots.indexOf("{\"a\":2")).toBeLessThan(argSnapshots.indexOf("{\"a\":2,\"b\":3}"));
  });

  test("streams reasoning in MCP mode before tool calls", async () => {
    const chunks: LLMStreamChunk[] = [];
    const transitions: string[] = [];
    let round = 0;

    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      const bodyParsed = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      round += 1;

      if (round === 1) {
        return sseResponse([
          JSON.stringify({ choices: [{ delta: { reasoning: "Need addition. " } }] }),
          JSON.stringify({
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      id: "call_add_reasoning",
                      type: "function",
                      function: { name: "add", arguments: "{\"a\":1,\"b\":4}" },
                    },
                  ],
                },
                finish_reason: "tool_calls",
              },
            ],
          }),
          "[DONE]",
        ]);
      }

      const messages = Array.isArray(bodyParsed.messages) ? bodyParsed.messages : [];
      const assistantMessage = messages.find((entry) => (entry as { role?: string }).role === "assistant") as
        | { reasoning?: string }
        | undefined;
      expect(assistantMessage?.reasoning).toBe("Need addition. ");

      return sseResponse([
        JSON.stringify({ choices: [{ delta: { reasoning: "Final answer. " } }] }),
        JSON.stringify({ choices: [{ delta: { content: "5" }, finish_reason: "stop" }] }),
        "[DONE]",
      ]);
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        mcpClients: [createSimpleMCP()],
        onTurnTransition: (transition) => transitions.push(`${transition.turnIndex}:${transition.kind}`),
      },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    expect(chunks[0]?.reasoningDelta).toBe("Need addition. ");
    expect(chunks[0]?.turnIndex).toBe(1);
    expect(chunks.some((chunk) => chunk.toolCalls?.[0]?.id === "call_add_reasoning")).toBe(true);
    expect(result.text).toBe("5");
    expect(result.reasoning).toBe("Need addition.\n\nFinal answer.");
    expect(result.reasoningBlocks).toEqual([
      { turnIndex: 1, text: "Need addition." },
      { turnIndex: 2, text: "Final answer." },
    ]);
    expect(transitions).toEqual([
      "1:reasoningComplete",
      "1:toolCallsEmit",
      "1:toolResultsReceived",
      "2:reasoningComplete",
      "2:streamEnd",
    ]);
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

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
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

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onToken: (t) => tokens.push(t) },
    );

    expect(result.text).toBe("from responses api");
    expect(tokens).toEqual(["from ", "responses api"]);
  });

  test("streams responses API tool calls in pass-through mode", async () => {
    const chunks: LLMStreamChunk[] = [];
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          type: "response.output_item.added",
          output_index: 0,
          item: {
            type: "function_call",
            call_id: "call_lookup",
            name: "lookup",
            arguments: "",
          },
        }),
        JSON.stringify({
          type: "response.function_call_arguments.delta",
          output_index: 0,
          item_id: "call_lookup",
          delta: "{\"q\"",
        }),
        JSON.stringify({
          type: "response.function_call_arguments.done",
          output_index: 0,
          item_id: "call_lookup",
          arguments: "{\"q\":\"x\"}",
        }),
        JSON.stringify({
          type: "response.completed",
          response: {
            id: "resp_1",
            status: "completed",
            output: [
              {
                type: "function_call",
                call_id: "call_lookup",
                name: "lookup",
                arguments: "{\"q\":\"x\"}",
              },
            ],
          },
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        body: {
          tools: [{ type: "function", name: "lookup", parameters: { type: "object" } }],
        },
      },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    expect(result.text).toBe("");
    expect(result.toolCalls).toEqual([
      { id: "call_lookup", type: "function", name: "lookup", arguments: "{\"q\":\"x\"}" },
    ]);
    expect(chunks.some((chunk) => chunk.toolCalls?.[0]?.id === "call_lookup")).toBe(true);
  });

  test("prefers final responses API usage over interim stream usage", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          type: "response.in_progress",
          response: {
            id: "resp_1",
            status: "in_progress",
            usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
          },
        }),
        JSON.stringify({
          type: "response.completed",
          response: {
            id: "resp_1",
            status: "completed",
            output_text: "done",
            usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5 },
          },
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "test" });

    expect(result.text).toBe("done");
    expect(result.usage).toEqual({
      inputTokens: 2,
      outputTokens: 3,
      totalTokens: 5,
    });
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

      expect(bodyParsed.previous_response_id).toBeUndefined();
      const inputItems = Array.isArray(bodyParsed.input) ? bodyParsed.input : [];
      expect(inputItems.at(-2)).toMatchObject({ type: "function_call", call_id: "call_sum" });
      expect(inputItems.at(-1)).toMatchObject({ type: "function_call_output", call_id: "call_sum" });

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

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
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
    expect(result.usage).toEqual({
      inputTokens: 5,
      outputTokens: 3,
      totalTokens: 8,
    });
  });

  test("uses distinct fallback IDs for native tool calls across MCP rounds", async () => {
    const requests: Record<string, unknown>[] = [];
    let round = 0;
    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      requests.push(JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>);
      round += 1;

      if (round <= 2) {
        return sseResponse([
          JSON.stringify({
            choices: [{
              delta: {
                content: `<tool_call>{"name":"add","arguments":{"a":${round},"b":1}}</tool_call>`,
              },
              finish_reason: "tool_calls",
            }],
          }),
          "[DONE]",
        ]);
      }

      return sseResponse([
        JSON.stringify({ choices: [{ delta: { content: "done" }, finish_reason: "stop" }] }),
        "[DONE]",
      ]);
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!({
      prompt: "test",
      mcpClients: [createSimpleMCP()],
      maxToolRounds: 2,
    });

    expect(result.text).toBe("done");
    expect(result.toolCalls?.map((call) => call.id)).toEqual([
      "call_native_round_1_0",
      "call_native_round_2_0",
    ]);
    const finalMessages = requests[2]?.messages as Array<Record<string, unknown>>;
    const assistantMessages = finalMessages.filter((message) => message.role === "assistant");
    expect(assistantMessages.map((message) => {
      const calls = message.tool_calls as Array<Record<string, unknown>>;
      return calls[0]?.id;
    })).toEqual(["call_native_round_1_0", "call_native_round_2_0"]);
  });
});

describe("openai-compatible text extraction", () => {
  test("normalizes chat completion logprobs", async () => {
    const fetcher = (async () =>
      jsonResponse({
        choices: [{
          finish_reason: "stop",
          message: { role: "assistant", content: "A" },
          logprobs: {
            content: [{
              token: "A",
              logprob: -0.25,
              bytes: [65],
              top_logprobs: [{ token: "B", logprob: -1.5, bytes: [66] }],
            }],
          },
        }],
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });
    const result = await adapter.complete({ prompt: "test" });

    expect(result.logprobs?.content).toEqual([{
      token: "A",
      logprob: -0.25,
      bytes: [65],
      top_logprobs: [{ token: "B", logprob: -1.5, bytes: [66] }],
    }]);
  });

  test("pickResponsesText from output_text", async () => {
    const fetcher = (async () =>
      jsonResponse({
        output_text: "direct output",
        status: "completed",
      })) as unknown as typeof fetch;

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.text).toBe("direct output");
  });

  test("extracts reasoning_content from chat completions messages", async () => {
    const fetcher = (async () =>
      jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: '{"value":7}',
              reasoning_content: "legacy reasoning",
            },
          },
        ],
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.text).toBe('{"value":7}');
    expect(result.reasoning).toBe("legacy reasoning");
  });

  test("extracts reasoning from chat completions messages", async () => {
    const fetcher = (async () =>
      jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: "done",
              reasoning: "new reasoning",
            },
          },
        ],
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.text).toBe("done");
    expect(result.reasoning).toBe("new reasoning");
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

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
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

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
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

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
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
      model: "test-model",
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
      model: "test-model",
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

    const adapter = createResponsesAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "test-model",
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
      model: "test-model",
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
      model: "test-model",
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
