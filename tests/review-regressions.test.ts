import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { createOpenAICompatibleAdapter as createResponsesAdapter } from "@/providers/openai-compatible";
import { createOpenAICompatibleLegacyAdapter } from "@/providers/openai-compatible-legacy";
import { createAnthropicCompatibleAdapter } from "@/providers/anthropic-compatible";
import { consumeSSE } from "@/providers/stream-utils";
import { executeMCPToolCalls, resolveMCPToolset } from "@/providers/mcp-runtime";
import { parseLLMOutput } from "@/parse";
import { conversation } from "@/conversation";
import { createStreamingStructuredParser } from "@/structured-streaming";
import type { MCPToolClient } from "@/types";

function streamOf(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    }),
  );
}

async function collectSSE(chunks: string[]): Promise<string[]> {
  const events: string[] = [];
  await consumeSSE(streamOf(chunks), (data) => events.push(data));
  return events;
}

function sseResponse(events: unknown[]): Response {
  return new Response(events.map((event) => `data: ${typeof event === "string" ? event : JSON.stringify(event)}\n\n`).join(""), {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function captureFetch(respond: (body: Record<string, unknown>) => Response) {
  const bodies: Record<string, unknown>[] = [];
  const fetcher = (async (_input: unknown, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
    bodies.push(body);
    return respond(body);
  }) as typeof fetch;
  return { bodies, fetcher };
}

describe("SSE parsing", () => {
  test("accepts every line terminator the spec allows, including mixed ones", async () => {
    expect(await collectSSE(["data: a\n\r\ndata: b\r\n\ndata: c\r\rdata: d\n\n"])).toEqual(["a", "b", "c", "d"]);
  });

  test("a CRLF split across chunks is one terminator", async () => {
    expect(await collectSSE(["data: a\r", "\n\r", "\ndata: b\r\n\r\n"])).toEqual(["a", "b"]);
  });

  test("an event with an empty data buffer is not dispatched", async () => {
    expect(await collectSSE(["data:\n\n: keep-alive\n\ndata: x\n\n"])).toEqual(["x"]);
  });

  test("joins multi-line data and ignores other fields", async () => {
    expect(await collectSSE(["event: message\nid: 1\ndata: a\ndata:b\nretry: 5\n\n"])).toEqual(["a\nb"]);
  });

  test("stays linear on a long event delivered in tiny chunks", async () => {
    const payload = "x".repeat(2_000_000);
    const chunks: string[] = ["data: "];
    for (let index = 0; index < payload.length; index += 64) {
      chunks.push(payload.slice(index, index + 64));
    }
    chunks.push("\n\n");
    const started = performance.now();
    const events = await collectSSE(chunks);
    expect(events[0]?.length).toBe(payload.length);
    expect(performance.now() - started).toBeLessThan(2_000);
  });
});

describe("OpenAI Responses protocol", () => {
  test("does not duplicate logprobs repeated by output_text.done", async () => {
    const logprob = { token: "Hi", logprob: -0.1, bytes: [72, 105], top_logprobs: [] };
    const { fetcher } = captureFetch(() =>
      sseResponse([
        { type: "response.output_text.delta", item_id: "m1", delta: "Hi", logprobs: [logprob] },
        { type: "response.output_text.done", item_id: "m1", text: "Hi", logprobs: [logprob] },
        { type: "response.completed", response: { status: "completed", output: [] } },
      ]),
    );
    const adapter = createResponsesAdapter({ baseURL: "https://example.com", model: "m", fetcher });
    const result = await adapter.stream!({ prompt: "hi", topLogprobs: 0 });
    expect(result.logprobs?.content).toHaveLength(1);
  });

  test("encodes assistant content parts as output_text", async () => {
    const { bodies, fetcher } = captureFetch(() =>
      new Response(JSON.stringify({ status: "completed", output_text: "ok" })),
    );
    const adapter = createResponsesAdapter({ baseURL: "https://example.com", model: "m", fetcher });
    await adapter.complete({
      messages: [
        { role: "user", content: [{ type: "text", text: "hello" }] },
        { role: "assistant", content: [{ type: "text", text: "previous answer" }] },
        { role: "user", content: "and now?" },
      ],
    });
    const input = bodies[0]?.input as Array<{ role: string; content: Array<{ type: string }> | string }>;
    expect((input[0]?.content as Array<{ type: string }>)[0]?.type).toBe("input_text");
    expect((input[1]?.content as Array<{ type: string }>)[0]?.type).toBe("output_text");
  });
});

describe("OpenAI Chat Completions protocol", () => {
  test("a name repeated in every tool-call delta is not doubled", async () => {
    const { fetcher } = captureFetch(() =>
      sseResponse([
        { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", type: "function", function: { name: "lookup", arguments: "" } }] } }] },
        { choices: [{ delta: { tool_calls: [{ index: 0, function: { name: "lookup", arguments: "{\"q\":" } }] } }] },
        { choices: [{ delta: { tool_calls: [{ index: 0, function: { name: "lookup", arguments: "1}" } }] } }] },
        { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
        "[DONE]",
      ]),
    );
    const adapter = createOpenAICompatibleLegacyAdapter({ baseURL: "https://example.com", model: "m", fetcher });
    const result = await adapter.stream!({ prompt: "hi", body: { tools: [{ type: "function", function: { name: "lookup" } }] } });
    expect(result.toolCalls?.[0]?.name).toBe("lookup");
    expect(result.toolCalls?.[0]?.arguments).toBe("{\"q\":1}");
  });

  test("a tool call streamed without any arguments still runs", async () => {
    const calls: unknown[] = [];
    const client: MCPToolClient = {
      id: "clock",
      async listTools() {
        return { tools: [{ name: "now", inputSchema: { type: "object", properties: {} } }] };
      },
      async callTool(params) {
        calls.push(params.arguments);
        return "12:00";
      },
    };
    let round = 0;
    const { fetcher } = captureFetch(() => {
      round += 1;
      return round === 1
        ? sseResponse([
            { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", type: "function", function: { name: "now" } }] } }] },
            { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
            "[DONE]",
          ])
        : sseResponse([{ choices: [{ delta: { content: "noon" }, finish_reason: "stop" }] }, "[DONE]"]);
    });
    const adapter = createOpenAICompatibleLegacyAdapter({ baseURL: "https://example.com", model: "m", fetcher });
    const result = await adapter.stream!({ prompt: "time?", mcpClients: [client] });
    expect(calls).toEqual([{}]);
    expect(result.toolCalls?.[0]?.error).toBeUndefined();
  });
});

describe("Anthropic protocol", () => {
  test("input tokens include cached prompt tokens", async () => {
    const { fetcher } = captureFetch(() =>
      new Response(JSON.stringify({
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 10, cache_creation_input_tokens: 100, cache_read_input_tokens: 1000, output_tokens: 5 },
      })),
    );
    const adapter = createAnthropicCompatibleAdapter({ baseURL: "https://example.com", model: "m", fetcher });
    const result = await adapter.complete({ prompt: "hi" });
    expect(result.usage?.inputTokens).toBe(1110);
    expect(result.usage?.outputTokens).toBe(5);
  });

  test("streamed usage keeps the cached prompt tokens of message_start", async () => {
    const { fetcher } = captureFetch(() =>
      sseResponse([
        { type: "message_start", message: { usage: { input_tokens: 10, cache_read_input_tokens: 1000, output_tokens: 1 } } },
        { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 7 } },
        { type: "message_stop" },
      ]),
    );
    const adapter = createAnthropicCompatibleAdapter({ baseURL: "https://example.com", model: "m", fetcher });
    const result = await adapter.stream!({ prompt: "hi" });
    expect(result.usage?.inputTokens).toBe(1010);
    expect(result.usage?.outputTokens).toBe(7);
  });

  test("results of parallel tool calls form a single user turn", async () => {
    const { bodies, fetcher } = captureFetch(() =>
      new Response(JSON.stringify({ content: [{ type: "text", text: "done" }], stop_reason: "end_turn" })),
    );
    const adapter = createAnthropicCompatibleAdapter({ baseURL: "https://example.com", model: "m", fetcher });
    await adapter.complete({
      messages: conversation("sys", [
        { role: "user", text: "compare" },
        { role: "tool_call", id: "a", name: "get", arguments: { id: 1 } },
        { role: "tool_call", id: "b", name: "get", arguments: { id: 2 } },
        { role: "tool_result", id: "a", output: { v: 1 } },
        { role: "tool_result", id: "b", output: { v: 2 } },
      ]),
    });
    const messages = bodies[0]?.messages as Array<{ role: string; content: Array<{ type: string }> }>;
    expect(messages.map((message) => message.role)).toEqual(["user", "assistant", "user"]);
    expect(messages[1]?.content.map((part) => part.type)).toEqual(["tool_use", "tool_use"]);
    expect(messages[2]?.content.map((part) => part.type)).toEqual(["tool_result", "tool_result"]);
  });
});

describe("conversation()", () => {
  test("the assistant text before tool calls belongs to the same message", () => {
    const messages = conversation("sys", [
      { role: "user", text: "compare" },
      { role: "assistant", text: "Let me look both up." },
      { role: "tool_call", id: "a", name: "get", arguments: { id: 1 } },
      { role: "tool_call", id: "b", name: "get", arguments: { id: 2 } },
      { role: "tool_result", id: "a", output: 1 },
      { role: "tool_result", id: "b", output: 2 },
    ]);
    expect(messages.map((message) => message.role)).toEqual(["system", "user", "assistant", "tool", "tool"]);
    expect(messages[2]?.content).toBe("Let me look both up.");
    expect((messages[2]?.tool_calls as unknown[]).length).toBe(2);
  });

  test("consecutive tool calls share one assistant message", () => {
    const messages = conversation("sys", [
      { role: "user", text: "compare" },
      { role: "tool_call", id: "a", name: "get", arguments: { id: 1 } },
      { role: "tool_call", id: "b", name: "get", arguments: { id: 2 } },
      { role: "tool_result", id: "a", output: undefined },
    ]);
    expect(messages).toHaveLength(4);
    expect((messages[2]?.tool_calls as unknown[]).length).toBe(2);
    expect(messages[3]?.content).toBe("null");
  });
});

describe("MCP tool loop", () => {
  test("stops running tools once the request is aborted", async () => {
    const controller = new AbortController();
    const ran: string[] = [];
    const client: MCPToolClient = {
      id: "c",
      async listTools() {
        return { tools: [{ name: "first" }, { name: "second" }] };
      },
      async callTool(params) {
        ran.push(params.name);
        controller.abort();
        return "ok";
      },
    };
    const toolset = await resolveMCPToolset([client]);
    await expect(
      executeMCPToolCalls(
        [
          { id: "1", type: "function", name: "first", arguments: "{}" },
          { id: "2", type: "function", name: "second", arguments: "{}" },
        ],
        toolset,
        { round: 1, request: { prompt: "x", signal: controller.signal } },
      ),
    ).rejects.toThrow();
    expect(ran).toEqual(["first"]);
  });

  test("forwards the request signal to the tool call", async () => {
    const controller = new AbortController();
    let received: AbortSignal | undefined;
    const client: MCPToolClient = {
      id: "c",
      async listTools() {
        return { tools: [{ name: "t" }] };
      },
      async callTool(_params, options) {
        received = options?.signal;
        return "ok";
      },
    };
    const toolset = await resolveMCPToolset([client]);
    await executeMCPToolCalls([{ id: "1", type: "function", name: "t", arguments: "{}" }], toolset, {
      round: 1,
      request: { prompt: "x", signal: controller.signal },
    });
    expect(received).toBe(controller.signal);
  });

  test("a server whose pagination never ends cannot hang the request", async () => {
    let pages = 0;
    const client: MCPToolClient = {
      id: "loop",
      async listTools() {
        pages += 1;
        return { tools: [], nextCursor: "again" };
      },
      async callTool() {
        return null;
      },
    };
    await expect(resolveMCPToolset([client])).rejects.toThrow(/cursor/i);
    expect(pages).toBeLessThan(10);
  });
});

describe("JSON extraction", () => {
  const schema = z.object({ size: z.number(), name: z.string() });

  test("an unbalanced quote in the prose does not hide the payload", () => {
    for (const text of [
      'I ordered a 12" pizza. Here it is: {"size": 12, "name": "margherita"} and that is all.',
      'I ordered a 12" pizza.\n{"size": 12, "name": "margherita"}\nThanks',
    ]) {
      const result = parseLLMOutput(text, schema, { repair: true });
      expect(result.success).toBe(true);
      expect(result.data).toEqual({ size: 12, name: "margherita" });
    }
  });

  test("a quoted decoy still loses to the real payload", () => {
    const result = parseLLMOutput('"draft: {not the payload}" {"size": 1, "name": "x"}', schema, { repair: true });
    expect(result.data).toEqual({ size: 1, name: "x" });
  });
});

describe("streaming preview", () => {
  test("a __proto__ key is kept as data, never as a prototype", () => {
    const parser = createStreamingStructuredParser();
    parser.update('{"__proto__": {"polluted": true}, "a": ');
    const preview = parser.update('{"__proto__": {"polluted": true}, "a": 1}') as Record<string, unknown>;
    expect(Object.getPrototypeOf(preview)).toBe(Object.prototype);
    expect((preview as { polluted?: unknown }).polluted).toBeUndefined();
    expect(Object.hasOwn(preview, "__proto__")).toBe(true);
    expect(preview.a).toBe(1);
  });
});
