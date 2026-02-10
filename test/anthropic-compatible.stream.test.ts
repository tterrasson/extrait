import { describe, expect, test } from "bun:test";
import { createAnthropicCompatibleAdapter } from "../src/providers/anthropic-compatible";
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
    const tokens: string[] = [];
    const chunks: LLMStreamChunk[] = [];
    let started = false;
    let completed = false;

    const fetcher = (async () =>
      sseResponse([
        JSON.stringify({ type: "content_block_delta", delta: { text: "Hello" } }),
        JSON.stringify({ type: "content_block_delta", delta: { text: " world" } }),
        JSON.stringify({
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: 2 },
        }),
        "[DONE]",
      ])) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
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
    expect(result.finishReason).toBe("end_turn");
    expect(result.usage?.outputTokens).toBe(2);
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

  test("delegates to streamViaComplete when MCP clients are present", async () => {
    let started = false;
    let completed = false;
    const tokens: string[] = [];

    const fetcher = (async () =>
      jsonResponse({
        content: [{ type: "text", text: "result" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 3, output_tokens: 1 },
      })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
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

  test("passes through 'auto' as-is", async () => {
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

    expect(bodyParsed?.tool_choice).toBe("auto");
  });
});

describe("anthropic-compatible error paths", () => {
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
