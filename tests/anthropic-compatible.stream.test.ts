import { describe, expect, test } from "bun:test";
import { createAnthropicCompatibleAdapter } from "@/providers/anthropic-compatible";
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
            description: "Add",
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

describe("anthropic-compatible streaming", () => {
  test("streams SSE chunks with token and chunk callbacks", async () => {
    const requests: Record<string, unknown>[] = [];
    const tokens: string[] = [];
    const chunks: LLMStreamChunk[] = [];
    let started = false;
    let completed = false;

    const fetcher = (async (_input: string | URL | Request, init: RequestInit) => {
      requests.push(JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>);
      return sseResponse([
        JSON.stringify({ type: "content_block_delta", delta: { text: "Hello" } }),
        JSON.stringify({ type: "content_block_delta", delta: { text: " world" } }),
        JSON.stringify({
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: 2 },
        }),
        "[DONE]",
      ]);
    }) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
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
    expect(result.finishReason).toBe("end_turn");
    expect(result.usage?.outputTokens).toBe(2);
    expect(requests[0]?.output_config).toEqual({
      effort: "max",
    });
    expect(requests[0]?.thinking).toEqual({
      type: "adaptive",
    });
  });

  test("keeps the latest cumulative usage snapshot instead of summing chunk usage", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          type: "message_start",
          message: { usage: { input_tokens: 10, output_tokens: 0 } },
        }),
        JSON.stringify({
          type: "message_delta",
          delta: { stop_reason: "end_turn", usage: { input_tokens: 10, output_tokens: 2 } },
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "test" });

    expect(result.usage).toEqual({
      inputTokens: 10,
      outputTokens: 2,
    });
  });

  test("ignores [DONE] sentinel", async () => {
    const tokens: string[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ type: "content_block_delta", delta: { text: "ok" } }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
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
      new Response("Internal Error", { status: 500 })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("HTTP 500");
  });

  test("streams through MCP rounds and keeps result.text as final assistant text", async () => {
    let startedCount = 0;
    let completed = false;
    const tokens: string[] = [];
    const chunks: LLMStreamChunk[] = [];
    const transitions: string[] = [];
    const requests: Record<string, unknown>[] = [];
    let round = 0;
    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      const bodyParsed = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(bodyParsed);
      round += 1;

      if (round === 1) {
        return sseResponse([
          JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } }),
          JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Need math. " } }),
          JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "signed-thinking" } }),
          JSON.stringify({ type: "content_block_start", index: 1, content_block: { type: "redacted_thinking", data: "redacted-data" } }),
          JSON.stringify({ type: "content_block_start", index: 2, content_block: { type: "text", text: "Let me check. " } }),
          JSON.stringify({
            type: "content_block_start",
            index: 3,
            content_block: {
              type: "tool_use",
              id: "toolu_add",
              name: "add",
              input: {},
            },
          }),
          JSON.stringify({
            type: "content_block_delta",
            index: 3,
            delta: { type: "input_json_delta", partial_json: "{\"a\":2" },
          }),
          JSON.stringify({
            type: "content_block_delta",
            index: 3,
            delta: { type: "input_json_delta", partial_json: ",\"b\":3}" },
          }),
          JSON.stringify({
            type: "message_delta",
            delta: { stop_reason: "tool_use" },
            usage: { input_tokens: 4, output_tokens: 2 },
          }),
          "[DONE]",
        ]);
      }

      const messages = Array.isArray(bodyParsed.messages) ? bodyParsed.messages : [];
      const hasToolResultMessage = messages.some((entry) => {
        const record = entry as { role?: string; content?: unknown };
        if (record.role !== "user" || !Array.isArray(record.content)) {
          return false;
        }
        return record.content.some((part) => (part as { type?: string }).type === "tool_result");
      });
      expect(hasToolResultMessage).toBe(true);

      return sseResponse([
        JSON.stringify({ type: "content_block_delta", delta: { type: "thinking_delta", thinking: "Got result. " } }),
        JSON.stringify({ type: "content_block_delta", delta: { text: "Result: " } }),
        JSON.stringify({ type: "content_block_delta", delta: { text: "5" } }),
        JSON.stringify({
          type: "message_delta",
          delta: { stop_reason: "end_turn", usage: { output_tokens: 1 } },
        }),
        "[DONE]",
      ]);
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.stream!(
      {
        prompt: "test",
        reasoningEffort: "low",
        mcpClients: [createSimpleMCP()],
        onTurnTransition: (transition) => transitions.push(`${transition.turnIndex}:${transition.kind}`),
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
    expect(tokens).toEqual(["Let me check. ", "Result: ", "5"]);
    expect(result.text).toBe("Result: 5");
    expect(result.reasoning).toBe("Need math.\n\nGot result.");
    expect(result.reasoningBlocks).toEqual([
      { turnIndex: 1, text: "Need math." },
      { turnIndex: 2, text: "Got result." },
    ]);
    expect(result.toolCalls?.[0]).toMatchObject({
      id: "toolu_add",
      name: "add",
      output: { result: 5 },
    });
    expect(result.toolExecutions?.[0]).toMatchObject({
      callId: "toolu_add",
      name: "add",
      clientId: "calc",
    });
    expect(result.usage).toEqual({
      inputTokens: 4,
      outputTokens: 3,
    });
    expect(chunks.some((chunk) => chunk.finishReason === "tool_use")).toBe(true);
    expect(chunks.some((chunk) => chunk.reasoningDelta === "Need math. " && chunk.turnIndex === 1)).toBe(true);
    expect(chunks.some((chunk) => chunk.toolCalls?.[0]?.id === "toolu_add")).toBe(true);
    expect(transitions).toEqual([
      "1:reasoningComplete",
      "1:toolCallsEmit",
      "1:toolResultsReceived",
      "2:reasoningComplete",
      "2:streamEnd",
    ]);
    expect(requests[0]?.output_config).toEqual({ effort: "low" });
    expect(requests[0]?.thinking).toEqual({ type: "adaptive" });
    expect(requests[1]?.output_config).toEqual({ effort: "low" });
    expect(requests[1]?.thinking).toEqual({ type: "adaptive" });
    const secondRoundMessages = requests[1]?.messages as Array<Record<string, unknown>>;
    expect(secondRoundMessages.at(-2)).toEqual({
      role: "assistant",
      content: [
        { type: "thinking", thinking: "Need math. ", signature: "signed-thinking" },
        { type: "redacted_thinking", data: "redacted-data" },
        { type: "text", text: "Let me check. " },
        { type: "tool_use", id: "toolu_add", name: "add", input: { a: 2, b: 3 } },
      ],
    });
  });

  test("streams partial tool-call arguments incrementally in MCP mode", async () => {
    let round = 0;
    const fetcher = (async () => {
      round += 1;
      if (round === 1) {
        return sseResponse([
          JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "toolu_add", name: "add", input: {} } }),
          JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: "{\"a\":2" } }),
          JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: ",\"b\":3}" } }),
          JSON.stringify({ type: "message_delta", delta: { stop_reason: "tool_use" } }),
          "[DONE]",
        ]);
      }
      return sseResponse([
        JSON.stringify({ type: "content_block_delta", delta: { text: "5" } }),
        JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" } }),
        "[DONE]",
      ]);
    }) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({ baseURL: "https://example.com", model: "claude-test", fetcher });

    const chunks: LLMStreamChunk[] = [];
    await adapter.stream!(
      { prompt: "test", mcpClients: [createSimpleMCP()] },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    // Arguments (built from `input_json_delta` fragments) surface progressively.
    const argSnapshots = chunks
      .map((chunk) => chunk.toolCalls?.[0]?.arguments)
      .filter((value): value is string => typeof value === "string");
    expect(argSnapshots).toContain("{\"a\":2");
    expect(argSnapshots).toContain("{\"a\":2,\"b\":3}");
    expect(argSnapshots.indexOf("{\"a\":2")).toBeLessThan(argSnapshots.indexOf("{\"a\":2,\"b\":3}"));
  });

  test("streams and returns tool calls in pass-through mode", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "toolu_lookup", name: "lookup", input: {} } }),
        JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: "{\"q\":" } }),
        JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: "\"x\"}" } }),
        JSON.stringify({ type: "message_delta", delta: { stop_reason: "tool_use" } }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({ baseURL: "https://example.com", model: "claude-test", fetcher });

    const chunks: LLMStreamChunk[] = [];
    const result = await adapter.stream!(
      { prompt: "test", body: { tools: [{ name: "lookup", input_schema: { type: "object" } }] } },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    // The tool call is surfaced in the final result (previously dropped entirely).
    expect(result.toolCalls?.[0]).toMatchObject({ id: "toolu_lookup", name: "lookup", arguments: "{\"q\":\"x\"}" });
    expect(result.finishReason).toBe("tool_use");
    // ...and its arguments streamed incrementally across chunks.
    const argSnapshots = chunks
      .map((chunk) => chunk.toolCalls?.[0]?.arguments)
      .filter((value): value is string => typeof value === "string");
    expect(argSnapshots).toContain("{\"q\":");
    expect(argSnapshots).toContain("{\"q\":\"x\"}");
    expect(argSnapshots.indexOf("{\"q\":")).toBeLessThan(argSnapshots.indexOf("{\"q\":\"x\"}"));
  });

  test("extracts delta from content_block.text", async () => {
    const tokens: string[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ type: "content_block_start", content_block: { type: "text", text: "block" } }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onToken: (t) => tokens.push(t) },
    );

    expect(result.text).toBe("block");
    expect(tokens).toEqual(["block"]);
  });

  test("returns empty delta when neither delta.text nor content_block.text", async () => {
    const tokens: string[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ type: "message_start", message: { usage: { input_tokens: 10 } } }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onToken: (t) => tokens.push(t) },
    );

    expect(result.text).toBe("");
    expect(tokens).toEqual([]);
  });
});

describe("anthropic-compatible pickUsage", () => {
  test("extracts usage from direct usage field", async () => {
    const fetcher = (async () =>
      jsonResponse({
        content: [{ type: "text", text: "hi" }],
        usage: { input_tokens: 5, output_tokens: 3 },
        stop_reason: "end_turn",
      })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.usage?.inputTokens).toBe(5);
    expect(result.usage?.outputTokens).toBe(3);
  });

  test("extracts usage from nested message.usage in stream", async () => {
    const chunks: LLMStreamChunk[] = [];

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          type: "message_start",
          message: { usage: { input_tokens: 10, output_tokens: 0 } },
        }),
        JSON.stringify({ type: "content_block_delta", delta: { text: "hi" } }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onChunk: (c) => chunks.push(c) },
    );

    expect(result.usage?.inputTokens).toBe(10);
  });

  test("extracts usage from delta.usage in stream", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({
          type: "message_delta",
          delta: { stop_reason: "end_turn", usage: { output_tokens: 5 } },
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "test" });
    expect(result.usage?.outputTokens).toBe(5);
    expect(result.finishReason).toBe("end_turn");
  });
});

describe("anthropic-compatible pickFinishReason", () => {
  test("extracts from direct stop_reason", async () => {
    const fetcher = (async () =>
      jsonResponse({
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
      })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.finishReason).toBe("end_turn");
  });

  test("extracts from delta.stop_reason in stream", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ type: "message_delta", delta: { stop_reason: "max_tokens" } }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "test" });
    expect(result.finishReason).toBe("max_tokens");
  });

  test("extracts from message.stop_reason in stream", async () => {
    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ type: "message_start", message: { stop_reason: "end_turn" } }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "test" });
    expect(result.finishReason).toBe("end_turn");
  });

  test("returns undefined for empty stop_reason", async () => {
    const fetcher = (async () =>
      jsonResponse({
        content: [{ type: "text", text: "hi" }],
        stop_reason: "",
      })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });
    expect(result.finishReason).toBeUndefined();
  });
});

describe("anthropic-compatible toAnthropicToolChoice", () => {
  test("passes through undefined", async () => {
    let bodyParsed: Record<string, unknown> | undefined;

    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      bodyParsed = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      return jsonResponse({
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "test",
      mcpClients: [createSimpleMCP()],
    });

    expect(bodyParsed?.tool_choice).toBeUndefined();
  });

  test("converts 'required' to { type: 'any' }", async () => {
    let bodyParsed: Record<string, unknown> | undefined;

    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      bodyParsed = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      return jsonResponse({
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "test",
      mcpClients: [createSimpleMCP()],
      toolChoice: "required",
    });

    expect(bodyParsed?.tool_choice).toEqual({ type: "any" });
  });

  test("converts function with name to { type: 'tool', name }", async () => {
    let bodyParsed: Record<string, unknown> | undefined;

    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      bodyParsed = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      return jsonResponse({
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "test",
      mcpClients: [createSimpleMCP()],
      toolChoice: { type: "function", function: { name: "add" } },
    });

    expect(bodyParsed?.tool_choice).toEqual({ type: "tool", name: "add" });
  });

  test("converts 'auto' to { type: 'auto' }", async () => {
    let bodyParsed: Record<string, unknown> | undefined;

    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      bodyParsed = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      return jsonResponse({
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "test",
      mcpClients: [createSimpleMCP()],
      toolChoice: "auto",
    });

    expect(bodyParsed?.tool_choice).toEqual({ type: "auto" });
  });

  test("converts 'none' and maps parallelToolCalls to disable_parallel_tool_use", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_url: unknown, init: RequestInit | undefined) => {
      requests.push(JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>);
      return jsonResponse({
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "test",
      mcpClients: [createSimpleMCP()],
      toolChoice: "none",
      parallelToolCalls: false,
    });
    await adapter.complete({
      prompt: "test",
      mcpClients: [createSimpleMCP()],
      parallelToolCalls: true,
    });

    expect(requests[0]?.tool_choice).toEqual({
      type: "none",
    });
    expect(requests[1]?.tool_choice).toEqual({
      type: "auto",
      disable_parallel_tool_use: false,
    });
  });

  test("rejects forced tool choice while Anthropic thinking is enabled", async () => {
    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher: (async () => jsonResponse({})) as unknown as typeof fetch,
    });

    await expect(adapter.complete({
      prompt: "test",
      reasoningEffort: "high",
      mcpClients: [createSimpleMCP()],
      toolChoice: "required",
    })).rejects.toThrow('only supports toolChoice "auto" or "none"');
  });
});

describe("anthropic-compatible error paths", () => {
  test("throws for a malformed JSON stream event", async () => {
    const fetcher = (async () => sseResponse(["not-json", "[DONE]"])) as unknown as typeof fetch;
    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("Invalid JSON event");
  });

  test("throws for an in-band stream error", async () => {
    const fetcher = (async () => sseResponse([
      JSON.stringify({
        type: "error",
        error: { type: "overloaded_error", message: "Service overloaded" },
      }),
    ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("Service overloaded");
  });

  test("throws when the stream ends without a terminal event or stop reason", async () => {
    const fetcher = (async () => sseResponse([
      JSON.stringify({ type: "content_block_delta", delta: { type: "text_delta", text: "partial" } }),
    ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("before a terminal event");
  });

  test("HTTP error in passthrough mode", async () => {
    const fetcher = (async () =>
      new Response("Bad Request", { status: 400 })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await expect(adapter.complete({ prompt: "test" })).rejects.toThrow("HTTP 400");
  });

  test("HTTP error in MCP tool loop", async () => {
    const fetcher = (async () =>
      new Response("Unauthorized", { status: 401 })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await expect(
      adapter.complete({
        prompt: "test",
        mcpClients: [createSimpleMCP()],
      }),
    ).rejects.toThrow("HTTP 401");
  });

  test("maxToolRounds exceeded throws error", async () => {
    let round = 0;

    const fetcher = (async () => {
      round += 1;
      return jsonResponse({
        content: [
          { type: "text", text: "" },
          {
            type: "tool_use",
            id: `call_${round}`,
            name: "add",
            input: { a: 1, b: 2 },
          },
        ],
        stop_reason: "tool_use",
      });
    }) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await expect(
      adapter.complete({
        prompt: "test",
        mcpClients: [createSimpleMCP()],
        maxToolRounds: 1,
      }),
    ).rejects.toThrow("exceeded maxToolRounds");
  });
});

describe("anthropic-compatible pass-through reasoning streaming", () => {
  test("accumulates thinking deltas and exposes reasoning and raw", async () => {
    const events = [
      JSON.stringify({ type: "message_start", message: { usage: { input_tokens: 7 } } }),
      JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } }),
      JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Let me think. " } }),
      JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Done." } }),
      JSON.stringify({ type: "content_block_start", index: 1, content_block: { type: "text", text: "" } }),
      JSON.stringify({ type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "Hello" } }),
      JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 3 } }),
    ];
    const fetcher = (async () => sseResponse(events)) as unknown as typeof fetch;
    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const chunks: LLMStreamChunk[] = [];
    const result = await adapter.stream!(
      { prompt: "hello", reasoningEffort: "medium" },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    expect(result.text).toBe("Hello");
    expect(result.reasoning).toBe("Let me think. Done.");
    expect(result.raw).toBeDefined();
    expect(result.finishReason).toBe("end_turn");
    expect(chunks.some((chunk) => chunk.reasoningDelta === "Let me think. ")).toBe(true);
    expect(chunks.some((chunk) => chunk.textDelta === "Hello")).toBe(true);
  });
});
