import { describe, expect, test } from "bun:test";
import { createOpenAICompatibleAdapter } from "@/providers/openai-compatible";
import type { MCPToolClient } from "@/types";

function jsonResponse(payload: Record<string, unknown>): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function createCalculatorMCP(onCall?: (args: Record<string, unknown>) => void): MCPToolClient {
  return {
    id: "calculator",
    async listTools() {
      return {
        tools: [
          {
            name: "add",
            description: "Add two numbers",
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
      const a = typeof args.a === "number" ? args.a : 0;
      const b = typeof args.b === "number" ? args.b : 0;
      return { result: a + b };
    },
  };
}

describe("openai-compatible MCP tools", () => {
  test("uses request.messages for chat completions pass-through", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: "ok",
            },
          },
        ],
      });
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const out = await adapter.complete({
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi" },
        { role: "user", content: "Need help" },
      ],
    });

    expect(out.text).toBe("ok");
    expect(requests[0]?.messages).toEqual([
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
      { role: "user", content: "Need help" },
    ]);
  });

  test("serializes reasoning_effort for chat completions pass-through", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: "ok",
            },
          },
        ],
      });
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "Need help",
      reasoningEffort: "max",
    });

    expect(requests[0]?.reasoning_effort).toBe("xhigh");
  });

  test("executes MCP tools with chat completions", async () => {
    const requests: Record<string, unknown>[] = [];
    let round = 0;
    let argsSeen: Record<string, unknown> | undefined;
    const executions: Array<{ callId: string; name?: string; clientId?: string }> = [];

    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      round += 1;

      if (round === 1) {
        return jsonResponse({
          choices: [
            {
              finish_reason: "tool_calls",
              message: {
                role: "assistant",
                content: "",
                tool_calls: [
                  {
                    id: "call_add",
                    type: "function",
                    function: {
                      name: "add",
                      arguments: JSON.stringify({ a: 2, b: 3 }),
                    },
                  },
                ],
              },
            },
          ],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
        });
      }

      return jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: "5",
            },
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      });
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const out = await adapter.complete({
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi" },
        { role: "user", content: "Compute 2+3" },
      ],
      mcpClients: [createCalculatorMCP((args) => (argsSeen = args))],
      onToolExecution: (execution) => {
        executions.push({
          callId: execution.callId,
          name: execution.name,
          clientId: execution.clientId,
        });
      },
    });

    expect(argsSeen).toEqual({ a: 2, b: 3 });
    expect(out.text).toBe("5");
    expect(out.toolCalls?.[0]).toMatchObject({ id: "call_add", name: "add", output: { result: 5 } });
    expect(out.toolExecutions?.[0]).toMatchObject({ callId: "call_add", name: "add", clientId: "calculator" });
    expect(executions).toEqual([{ callId: "call_add", name: "add", clientId: "calculator" }]);

    const first = requests[0];
    expect(first?.reasoning_effort).toBeUndefined();
    expect(first?.messages).toEqual([
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
      { role: "user", content: "Compute 2+3" },
    ]);
    const tools = Array.isArray(first?.tools) ? first.tools : [];
    expect(((tools[0] as { function?: { name?: string } }).function?.name)).toBe("add");

    const second = requests[1];
    const messages = Array.isArray(second?.messages) ? second.messages : [];
    expect(messages.some((entry) => (entry as { role?: string }).role === "tool")).toBe(true);
  });

  test("serializes reasoning_effort for chat completions MCP requests", async () => {
    const requests: Record<string, unknown>[] = [];
    let round = 0;
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      round += 1;

      if (round === 1) {
        return jsonResponse({
          choices: [
            {
              finish_reason: "tool_calls",
              message: {
                role: "assistant",
                content: "",
                tool_calls: [
                  {
                    id: "call_add",
                    type: "function",
                    function: {
                      name: "add",
                      arguments: JSON.stringify({ a: 2, b: 3 }),
                    },
                  },
                ],
              },
            },
          ],
        });
      }

      return jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: "5",
            },
          },
        ],
      });
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "Compute 2+3",
      reasoningEffort: "medium",
      mcpClients: [createCalculatorMCP()],
    });

    expect(requests[0]?.reasoning_effort).toBe("medium");
    expect(requests[1]?.reasoning_effort).toBe("medium");
  });

  test("serializes reasoning_effort for responses API requests", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      return jsonResponse({
        output_text: "ok",
        status: "completed",
      });
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    await adapter.complete({
      prompt: "Need help",
      reasoningEffort: "max",
    });

    expect(requests[0]?.reasoning_effort).toBe("xhigh");
  });

  test("surfaces unknown MCP tool as tool error and continues", async () => {
    let round = 0;
    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      round += 1;

      if (round === 1) {
        return jsonResponse({
          choices: [
            {
              finish_reason: "tool_calls",
              message: {
                role: "assistant",
                content: "",
                tool_calls: [
                  {
                    id: "call_lookup",
                    type: "function",
                    function: {
                      name: "lookup",
                      arguments: JSON.stringify({ q: "x" }),
                    },
                  },
                ],
              },
            },
          ],
        });
      }

      const messages = Array.isArray(body.messages) ? body.messages : [];
      const toolMessage = messages.find((entry) => (entry as { role?: string }).role === "tool") as
        | { content?: string }
        | undefined;
      expect(toolMessage?.content).toBe('{"error":"Tool \\"lookup\\" is not registered in the current toolset."}');

      return jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: "lookup unavailable",
            },
          },
        ],
      });
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const out = await adapter.complete({
      prompt: "lookup",
      mcpClients: [createCalculatorMCP()],
    });

    expect(out.text).toBe("lookup unavailable");
    expect(out.toolCalls?.[0]).toMatchObject({
      id: "call_lookup",
      name: "lookup",
      error: 'Tool "lookup" is not registered in the current toolset.',
    });
    expect(out.toolExecutions?.[0]).toMatchObject({
      callId: "call_lookup",
      name: "lookup",
      clientId: "__unregistered__",
      error: 'Tool "lookup" is not registered in the current toolset.',
    });
  });

  test("executes MCP tools in responses API mode", async () => {
    let round = 0;
    let argsSeen: Record<string, unknown> | undefined;
    const requests: Record<string, unknown>[] = [];

    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      requests.push(body);
      round += 1;

      if (round === 1) {
        return jsonResponse({
          id: "resp_1",
          output: [
            {
              type: "function_call",
              call_id: "call_sum",
              name: "add",
              arguments: JSON.stringify({ a: 7, b: 9 }),
            },
          ],
          status: "requires_action",
          usage: { input_tokens: 4, output_tokens: 2, total_tokens: 6 },
        });
      }

      return jsonResponse({
        id: "resp_2",
        output_text: "16",
        status: "completed",
        usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
      });
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      path: "/v1/responses",
      model: "gpt-test",
      fetcher,
    });

    const out = await adapter.complete({
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "First turn" },
        { role: "assistant", content: "Seen" },
        { role: "user", content: "Add 7 and 9" },
      ],
      mcpClients: [createCalculatorMCP((args) => (argsSeen = args))],
    });

    expect(argsSeen).toEqual({ a: 7, b: 9 });
    expect(out.text).toBe("16");
    expect(out.toolCalls?.some((call) => call.name === "add" && (call.output as { result?: number })?.result === 16)).toBe(
      true,
    );

    const second = requests[1];
    expect(second?.previous_response_id).toBe("resp_1");
    const inputItems = Array.isArray(second?.input) ? second.input : [];
    expect((inputItems[0] as { type?: string }).type).toBe("function_call_output");

    const firstInput = Array.isArray(requests[0]?.input) ? requests[0]?.input : [];
    expect(firstInput).toEqual([
      { role: "system", content: "You are helpful." },
      { role: "user", content: "First turn" },
      { role: "assistant", content: "Seen" },
      { role: "user", content: "Add 7 and 9" },
    ]);
  });

  test("toolDebug logs request and result payloads for each MCP call", async () => {
    let round = 0;
    const logs: string[] = [];

    const fetcher = (async (_input, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      round += 1;

      if (round === 1) {
        return jsonResponse({
          choices: [
            {
              finish_reason: "tool_calls",
              message: {
                role: "assistant",
                content: "",
                tool_calls: [
                  {
                    id: "call_add_debug",
                    type: "function",
                    function: {
                      name: "add",
                      arguments: JSON.stringify({ a: 10, b: 5 }),
                    },
                  },
                ],
              },
            },
          ],
          usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
        });
      }

      const messages = Array.isArray(body.messages) ? body.messages : [];
      expect(messages.some((entry) => (entry as { role?: string }).role === "tool")).toBe(true);

      return jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: "15",
            },
          },
        ],
      });
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const out = await adapter.complete({
      prompt: "Add 10 and 5",
      mcpClients: [createCalculatorMCP()],
      toolDebug: {
        enabled: true,
        logger: (line) => logs.push(line),
      },
    });

    expect(out.text).toBe("15");
    expect(logs.some((line) => line.includes("[tool:mcp:ok]"))).toBe(true);
    expect(
      logs.some((line) => line.includes("[tool:mcp:request]") && line.includes('arguments={"a":10,"b":5}')),
    ).toBe(true);
    expect(
      logs.some((line) => line.includes("[tool:mcp:result:ok]") && line.includes('output={"result":15}')),
    ).toBe(true);
  });

  test("keeps final assistant reasoning in chat completions MCP mode", async () => {
    let round = 0;

    const fetcher = (async () => {
      round += 1;

      if (round === 1) {
        return jsonResponse({
          choices: [
            {
              finish_reason: "tool_calls",
              message: {
                role: "assistant",
                content: "",
                reasoning: "Need a calculator.",
                tool_calls: [
                  {
                    id: "call_add_reasoning_complete",
                    type: "function",
                    function: {
                      name: "add",
                      arguments: JSON.stringify({ a: 4, b: 6 }),
                    },
                  },
                ],
              },
            },
          ],
        });
      }

      return jsonResponse({
        choices: [
          {
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: "10",
              reasoning: "Computed from tool output.",
            },
          },
        ],
      });
    }) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });

    const out = await adapter.complete({
      prompt: "Add 4 and 6",
      mcpClients: [createCalculatorMCP()],
    });

    expect(out.text).toBe("10");
    expect(out.reasoning).toBe("Computed from tool output.");
    expect(out.toolCalls?.[0]).toMatchObject({
      id: "call_add_reasoning_complete",
      name: "add",
      output: { result: 10 },
    });
  });

  test("returns tool calls in pass-through mode when no MCP clients are provided", async () => {
    const fetcher = (async () =>
      jsonResponse({
        choices: [
          {
            finish_reason: "tool_calls",
            message: {
              role: "assistant",
              content: "",
              tool_calls: [
                {
                  id: "call_lookup",
                  type: "function",
                  function: {
                    name: "lookup",
                    arguments: JSON.stringify({ q: "bun" }),
                  },
                },
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

    const out = await adapter.complete({
      prompt: "lookup",
    });

    expect(out.text).toBe("");
    expect(out.toolCalls).toEqual([
      {
        id: "call_lookup",
        type: "function",
        name: "lookup",
        arguments: "{\"q\":\"bun\"}",
      },
    ]);
    expect(out.toolExecutions).toBeUndefined();
  });

  test("uses adapter defaultMaxToolRounds when request does not set maxToolRounds", async () => {
    const fetcher = (async () =>
      jsonResponse({
        choices: [
          {
            finish_reason: "tool_calls",
            message: {
              role: "assistant",
              content: "",
              tool_calls: [
                {
                  id: "call_add",
                  type: "function",
                  function: {
                    name: "add",
                    arguments: JSON.stringify({ a: 1, b: 2 }),
                  },
                },
              ],
            },
          },
        ],
      })) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      defaultMaxToolRounds: 0,
      fetcher,
    });

    await expect(
      adapter.complete({
        prompt: "Add 1 and 2",
        mcpClients: [createCalculatorMCP()],
      }),
    ).rejects.toThrow("Tool call loop exceeded maxToolRounds (0).");
  });
});
