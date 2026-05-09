import { describe, expect, test } from "bun:test";
import {
  createAnthropicCompatibleAdapter,
  DEFAULT_ANTHROPIC_MAX_TOKENS,
} from "@/providers/anthropic-compatible";
import type { MCPToolClient } from "@/types";

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

function createSumMCP(onCall?: (args: Record<string, unknown>) => void): MCPToolClient {
  return {
    id: "calculator",
    async listTools() {
      return {
        tools: [
          {
            name: "sum",
            inputSchema: {
              type: "object",
              properties: {
                a: { type: "number" },
                b: { type: "number" },
              },
              required: ["a", "b"],
            },
          },
        ],
      };
    },
    async callTool(params) {
      const args = params.arguments ?? {};
      onCall?.(args);
      return (args.a as number) + (args.b as number);
    },
  };
}

describe("anthropic-compatible MCP tools", () => {
  test("returns thinking blocks as reasoning blocks for non-streaming MCP rounds", async () => {
    let round = 0;
    const fetcher = (async () => {
      round += 1;

      if (round === 1) {
        return jsonResponse({
          content: [
            { type: "thinking", thinking: "Need sum." },
            { type: "tool_use", id: "toolu_sum", name: "sum", input: { a: 2, b: 3 } },
          ],
          stop_reason: "tool_use",
        });
      }

      return jsonResponse({
        content: [
          { type: "thinking", thinking: "Answer from tool." },
          { type: "text", text: "5" },
        ],
        stop_reason: "end_turn",
      });
    }) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const out = await adapter.complete({
      prompt: "2+3?",
      mcpClients: [createSumMCP()],
    });

    expect(out.text).toBe("5");
    expect(out.reasoning).toBe("Need sum.\n\nAnswer from tool.");
    expect(out.reasoningBlocks).toEqual([
      { turnIndex: 1, text: "Need sum." },
      { turnIndex: 2, text: "Answer from tool." },
    ]);
  });

  test("uses request.messages and extracts leading system turns", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const out = await adapter.complete({
      messages: [
        { role: "system", content: "System one" },
        { role: "system", content: "System two" },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi" },
        { role: "user", content: "Need help" },
      ],
    });

    expect(out.text).toBe("ok");
    expect(requests[0]?.system).toBe("System one\n\nSystem two");
    expect(requests[0]?.messages).toEqual([
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
      { role: "user", content: "Need help" },
    ]);
  });

  test("rejects system turns after non-system messages", async () => {
    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher: (async () => jsonResponse({})) as unknown as typeof fetch,
    });

    await expect(
      adapter.complete({
        messages: [
          { role: "user", content: "Hello" },
          { role: "system", content: "Late system" },
        ],
      }),
    ).rejects.toThrow('Anthropic-compatible messages only support "system" turns at the beginning.');
  });

  test("throws when response has no assistant text or tool calls", async () => {
    const fetcher = (async () =>
      jsonResponse({
        content: [],
        stop_reason: "end_turn",
      })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await expect(adapter.complete({ prompt: "hello" })).rejects.toThrow(
      "No assistant text in Anthropic-compatible response.",
    );
  });

  test("executes MCP tools with local handler loop", async () => {
    const requests: Record<string, unknown>[] = [];
    let round = 0;
    let argsSeen: Record<string, unknown> | undefined;

    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      round += 1;

      if (round === 1) {
        return jsonResponse({
          content: [
            {
              type: "tool_use",
              id: "toolu_sum",
              name: "sum",
              input: { a: 4, b: 6 },
            },
          ],
          stop_reason: "tool_use",
          usage: { input_tokens: 8, output_tokens: 3, total_tokens: 11 },
        });
      }

      return jsonResponse({
        content: [{ type: "text", text: "10" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const out = await adapter.complete({
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Bonjour" },
        { role: "assistant", content: "Salut" },
        { role: "user", content: "Calcule 4 + 6" },
      ],
      mcpClients: [createSumMCP((args) => (argsSeen = args))],
    });

    expect(argsSeen).toEqual({ a: 4, b: 6 });
    expect(out.text).toBe("10");
    expect(out.toolCalls?.[0]).toMatchObject({ id: "toolu_sum", name: "sum", output: 10 });
    expect(out.toolExecutions?.[0]).toMatchObject({ callId: "toolu_sum", clientId: "calculator", handledLocally: true });
    expect(out.usage).toEqual({ inputTokens: 11, outputTokens: 4, totalTokens: 15 });

    const second = requests[1];
    expect(requests[0]?.system).toBe("You are helpful.");
    expect((requests[0]?.messages as Array<{ role?: string }>)[0]?.role).toBe("user");
    const messages = Array.isArray(second?.messages) ? second.messages : [];
    expect((messages[3] as { role?: string }).role).toBe("assistant");
    expect((messages[4] as { role?: string }).role).toBe("user");
  });

  test("returns tool calls in pass-through mode when no MCP clients are provided", async () => {
    const fetcher = (async () =>
      jsonResponse({
        content: [
          {
            type: "tool_use",
            id: "toolu_lookup",
            name: "lookup",
            input: { q: "bun" },
          },
        ],
        stop_reason: "tool_use",
        usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
      })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const out = await adapter.complete({
      prompt: "lookup",
    });

    expect(out.text).toBe("");
    expect(out.toolCalls).toEqual([
      {
        id: "toolu_lookup",
        type: "function",
        name: "lookup",
        arguments: { q: "bun" },
      },
    ]);
  });

  test("stream() executes MCP tool loop and streams only assistant text deltas", async () => {
    let round = 0;
    const fetcher = (async () => {
      round += 1;
      if (round === 1) {
        return sseResponse([
          JSON.stringify({
            type: "content_block_start",
            index: 0,
            content_block: { type: "tool_use", id: "toolu_sum", name: "sum", input: { a: 1, b: 2 } },
          }),
          JSON.stringify({
            type: "message_delta",
            delta: { stop_reason: "tool_use" },
          }),
          "[DONE]",
        ]);
      }

      return sseResponse([
        JSON.stringify({ type: "content_block_delta", delta: { text: "3" } }),
        JSON.stringify({
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
        }),
        "[DONE]",
      ]);
    }) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    const tokens: string[] = [];
    const out = await adapter.stream!(
      {
        prompt: "hello",
        mcpClients: [createSumMCP()],
      },
      {
        onToken: (token) => tokens.push(token),
      },
    );

    expect(tokens).toEqual(["3"]);
    expect(out.text).toBe("3");
  });

  test("uses adapter defaultMaxTokens when request maxTokens is not provided", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      defaultMaxTokens: 321,
      fetcher,
    });

    const out = await adapter.complete({
      prompt: "hello",
    });

    expect(out.text).toBe("ok");
    expect(requests[0]?.max_tokens).toBe(321);
  });

  test("serializes output_config.effort and adaptive thinking for reasoningEffort in pass-through mode", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "hello",
      reasoningEffort: "medium",
    });

    expect(requests[0]?.output_config).toEqual({
      effort: "medium",
    });
    expect(requests[0]?.thinking).toEqual({
      type: "adaptive",
    });
    expect(requests[0]?.max_tokens).toBe(DEFAULT_ANTHROPIC_MAX_TOKENS);
  });

  test("forwards minimal and none output_config.effort values as-is", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "hello",
      reasoningEffort: "minimal",
    });
    await adapter.complete({
      prompt: "hello again",
      reasoningEffort: "none",
    });

    expect(requests[0]?.output_config).toEqual({
      effort: "minimal",
    });
    expect(requests[0]?.thinking).toEqual({
      type: "adaptive",
    });
    expect(requests[1]?.output_config).toEqual({
      effort: "none",
    });
    expect(requests[1]?.thinking).toEqual({
      type: "adaptive",
    });
  });

  test("falls back to library default max tokens when adapter default is invalid", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      defaultMaxTokens: -10,
      fetcher,
    });

    await adapter.complete({ prompt: "hello" });
    expect(requests[0]?.max_tokens).toBe(DEFAULT_ANTHROPIC_MAX_TOKENS);
  });

  test("uses adapter defaultMaxToolRounds when request maxToolRounds is not provided", async () => {
    const fetcher = (async () =>
      jsonResponse({
        content: [
          {
            type: "tool_use",
            id: "toolu_sum",
            name: "sum",
            input: { a: 1, b: 2 },
          },
        ],
        stop_reason: "tool_use",
      })) as unknown as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      defaultMaxToolRounds: 0,
      fetcher,
    });

    await expect(
      adapter.complete({
        prompt: "Calcule 1 + 2",
        mcpClients: [createSumMCP()],
      }),
    ).rejects.toThrow("Tool call loop exceeded maxToolRounds (0).");
  });

  test("serializes output_config.effort and adaptive thinking for MCP requests", async () => {
    const requests: Record<string, unknown>[] = [];
    let round = 0;
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      round += 1;

      if (round === 1) {
        return jsonResponse({
          content: [
            {
              type: "tool_use",
              id: "toolu_sum",
              name: "sum",
              input: { a: 1, b: 2 },
            },
          ],
          stop_reason: "tool_use",
        });
      }

      return jsonResponse({
        content: [{ type: "text", text: "3" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "Calcule 1 + 2",
      reasoningEffort: "low",
      mcpClients: [createSumMCP()],
    });

    expect(requests[0]?.thinking).toEqual({
      type: "adaptive",
    });
    expect(requests[0]?.output_config).toEqual({
      effort: "low",
    });
    expect(requests[0]?.max_tokens).toBe(DEFAULT_ANTHROPIC_MAX_TOKENS);
    expect(requests[1]?.output_config).toEqual({
      effort: "low",
    });
  });

  test("preserves explicit body.thinking while still injecting output_config.effort", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "hello",
      reasoningEffort: "high",
      body: {
        thinking: {
          type: "adaptive",
          custom: true,
        },
        output_config: {
          foo: "bar",
          effort: "low",
        },
      },
    });

    expect(requests[0]?.thinking).toEqual({
      type: "adaptive",
      custom: true,
    });
    expect(requests[0]?.output_config).toEqual({
      foo: "bar",
      effort: "high",
    });
  });

  test("preserves explicit null body.thinking while still injecting output_config.effort", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
      });
    }) as typeof fetch;

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "hello",
      reasoningEffort: "medium",
      body: {
        thinking: null,
      },
    });

    expect(requests[0]).toHaveProperty("thinking", null);
    expect(requests[0]?.output_config).toEqual({
      effort: "medium",
    });
  });
});
