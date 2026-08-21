import { describe, expect, test } from "bun:test";
import { createOpenAICompatibleAdapter } from "@/providers/openai-compatible";
import { createOpenAICompatibleLegacyAdapter } from "@/providers/openai-compatible-legacy";

function jsonResponse(payload: Record<string, unknown>): Response {
  return new Response(JSON.stringify(payload), { status: 200 });
}

function sseResponse(events: Record<string, unknown>[]): Response {
  return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

describe("openai-compatible Responses contract", () => {
  test("uses /v1/responses and translates common request options", async () => {
    let url = "";
    let body: Record<string, unknown> = {};
    const signal = new AbortController().signal;
    const fetcher = (async (input, init) => {
      url = String(input);
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(init?.signal).toBe(signal);
      return jsonResponse({
        status: "completed",
        output: [{
          type: "message",
          content: [{
            type: "output_text",
            text: "yes",
            logprobs: [{
              token: "yes",
              logprob: -0.1,
              bytes: [121, 101, 115],
              top_logprobs: [{ token: "no", logprob: -2, bytes: [110, 111] }],
            }],
          }],
        }],
        usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
      });
    }) as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });
    const result = await adapter.complete({
      prompt: "Answer",
      systemPrompt: "Be brief",
      temperature: 0.2,
      maxTokens: 20,
      reasoningEffort: "max",
      topLogprobs: 3,
      signal,
    });

    expect(url).toBe("https://example.com/v1/responses");
    expect(body).toMatchObject({
      model: "test-model",
      input: "Answer",
      instructions: "Be brief",
      temperature: 0.2,
      max_output_tokens: 20,
      reasoning: { effort: "max", summary: "auto" },
      top_logprobs: 3,
      include: ["message.output_text.logprobs"],
    });
    expect(body).not.toHaveProperty("messages");
    expect(body).not.toHaveProperty("reasoning_effort");
    expect(result.text).toBe("yes");
    expect(result.logprobs?.content?.[0]).toMatchObject({ token: "yes", logprob: -0.1 });
    expect(result.usage).toEqual({ inputTokens: 3, outputTokens: 1, totalTokens: 4 });
  });

  test("translates message content and images to Responses input parts", async () => {
    let body: Record<string, unknown> = {};
    const fetcher = (async (_input, init) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ output_text: "seen", status: "completed" });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await adapter.complete({
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "Describe" },
          { type: "image_url", image_url: { url: "data:image/png;base64,abc" } },
        ],
      }],
    });

    expect(body.input).toEqual([{
      role: "user",
      content: [
        { type: "input_text", text: "Describe" },
        { type: "input_image", image_url: "data:image/png;base64,abc" },
      ],
    }]);
  });

  test("uses input content parts for assistant history and preserves text beside tool calls", async () => {
    let body: Record<string, unknown> = {};
    const fetcher = (async (_input, init) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ output_text: "done", status: "completed" });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await adapter.complete({
      messages: [
        { role: "user", content: "Compute" },
        {
          role: "assistant",
          content: [{ type: "text", text: "I will calculate it." }],
          tool_calls: [{
            id: "call_1",
            type: "function",
            function: { name: "sum", arguments: "{\"a\":1,\"b\":2}" },
          }],
        },
        { role: "tool", content: "3", tool_call_id: "call_1" },
      ],
    });

    expect(body.input).toEqual([
      { role: "user", content: "Compute" },
      {
        role: "assistant",
        content: [{ type: "input_text", text: "I will calculate it." }],
      },
      { type: "function_call", call_id: "call_1", name: "sum", arguments: "{\"a\":1,\"b\":2}" },
      { type: "function_call_output", call_id: "call_1", output: "3" },
    ]);
  });

  test("returns non-streaming refusal output as text", async () => {
    const fetcher = (async () => jsonResponse({
      status: "completed",
      output: [{
        type: "message",
        content: [{ type: "refusal", refusal: "I cannot help with that." }],
      }],
    })) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });

    expect(result.text).toBe("I cannot help with that.");
  });

  test("streams and accumulates Responses logprobs", async () => {
    const chunks: unknown[] = [];
    const fetcher = (async () => sseResponse([
      {
        type: "response.output_text.delta",
        delta: "A",
        logprobs: [{ token: "A", logprob: -0.2, bytes: [65], top_logprobs: [] }],
      },
      {
        type: "response.completed",
        response: { status: "completed", output_text: "A" },
      },
    ])) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "letter", topLogprobs: 2 },
      { onChunk: (chunk) => chunks.push(chunk) },
    );

    expect(result.text).toBe("A");
    expect(result.logprobs?.content).toEqual([{ token: "A", logprob: -0.2, bytes: [65] }]);
    expect(chunks).toContainEqual(expect.objectContaining({
      textDelta: "A",
      logprobs: { content: [{ token: "A", logprob: -0.2, bytes: [65] }] },
    }));
  });

  test("streams refusal deltas as response text", async () => {
    const tokens: string[] = [];
    const fetcher = (async () => sseResponse([
      { type: "response.refusal.delta", delta: "I cannot" },
      { type: "response.refusal.delta", delta: " help." },
      {
        type: "response.completed",
        response: {
          status: "completed",
          output: [{
            type: "message",
            content: [{ type: "refusal", refusal: "I cannot help." }],
          }],
        },
      },
    ])) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!({ prompt: "test" }, {
      onToken: (token) => tokens.push(token),
    });

    expect(tokens).toEqual(["I cannot", " help."]);
    expect(result.text).toBe("I cannot help.");
  });

  test("throws when a Responses stream ends before a terminal event", async () => {
    const fetcher = (async () => sseResponse([
      { type: "response.output_text.delta", delta: "partial" },
    ])) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("before a terminal event");
  });

  test("throws for a malformed Responses stream event", async () => {
    const fetcher = (async () => new Response("data: not-json\n\ndata: [DONE]\n\n", {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    })) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("Invalid JSON event");
  });

  test("serializes Responses request options in streaming mode", async () => {
    let body: Record<string, unknown> = {};
    const signal = new AbortController().signal;
    const fetcher = (async (_input, init) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(init?.signal).toBe(signal);
      return sseResponse([{
        type: "response.completed",
        response: { status: "completed", output_text: "ok" },
      }]);
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      defaultBody: {
        include: ["message.output_text.logprobs"],
        reasoning: { summary: "auto" },
      },
      fetcher,
    });

    await adapter.stream!({
      prompt: "test",
      systemPrompt: "Be terse",
      reasoningEffort: "low",
      topLogprobs: 0,
      signal,
    });

    expect(body).toMatchObject({
      input: "test",
      instructions: "Be terse",
      reasoning: { summary: "auto", effort: "low" },
      top_logprobs: 0,
      include: ["message.output_text.logprobs"],
      stream: true,
    });
  });

  test("keeps an explicitly configured reasoning summary, null and undefined included", async () => {
    const bodies: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return jsonResponse({ status: "completed", output_text: "ok" });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await adapter.complete({ prompt: "test", reasoningEffort: "low", body: { reasoning: { summary: "detailed" } } });
    await adapter.complete({ prompt: "test", reasoningEffort: "low", body: { reasoning: { summary: null } } });
    await adapter.complete({ prompt: "test", reasoningEffort: "low", body: { reasoning: { summary: undefined } } });

    expect(bodies[0]?.reasoning).toEqual({ effort: "low", summary: "detailed" });
    expect(bodies[1]?.reasoning).toEqual({ effort: "low", summary: null });
    // `undefined` never survives JSON serialization: the field leaves the payload.
    expect(bodies[2]?.reasoning).toEqual({ effort: "low" });
  });

  test("streams parallel tool-call arguments by output item id", async () => {
    const fetcher = (async () => sseResponse([
      {
        type: "response.output_item.added",
        output_index: 0,
        item: { type: "function_call", id: "fc_1", call_id: "call_1", name: "first", arguments: "" },
      },
      {
        type: "response.output_item.added",
        output_index: 1,
        item: { type: "function_call", id: "fc_2", call_id: "call_2", name: "second", arguments: "" },
      },
      { type: "response.function_call_arguments.delta", item_id: "fc_1", delta: "{\"a\":" },
      { type: "response.function_call_arguments.delta", item_id: "fc_2", delta: "{\"b\":2}" },
      { type: "response.function_call_arguments.done", item_id: "fc_1", arguments: "{\"a\":1}" },
      {
        type: "response.completed",
        response: {
          status: "completed",
          output: [
            { type: "function_call", id: "fc_1", call_id: "call_1", name: "first", arguments: "{\"a\":1}" },
            { type: "function_call", id: "fc_2", call_id: "call_2", name: "second", arguments: "{\"b\":2}" },
          ],
        },
      },
    ])) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!({
      prompt: "use both",
      body: { tools: [{ type: "function", name: "first" }, { type: "function", name: "second" }] },
    });

    expect(result.toolCalls).toEqual([
      { id: "call_1", type: "function", name: "first", arguments: "{\"a\":1}" },
      { id: "call_2", type: "function", name: "second", arguments: "{\"b\":2}" },
    ]);
  });

  test("merges the logprobs include entry with a caller-provided include", async () => {
    let body: Record<string, unknown> = {};
    const fetcher = (async (_input, init) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ output_text: "ok", status: "completed" });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await adapter.complete({
      prompt: "test",
      topLogprobs: 1,
      body: { include: ["reasoning.encrypted_content"] },
    });

    expect(body.include).toEqual(["reasoning.encrypted_content", "message.output_text.logprobs"]);
  });

  test("passes the legacy logprobs dialect flag through defaultBody", async () => {
    let body: Record<string, unknown> = {};
    const fetcher = (async (_input, init) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ output_text: "ok", status: "completed" });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      defaultBody: { logprobs: true },
      fetcher,
    });

    await adapter.complete({ prompt: "test", topLogprobs: 2 });

    expect(body).toMatchObject({ logprobs: true, top_logprobs: 2 });
  });

  test("uses an explicit body.input without requiring a prompt", async () => {
    let body: Record<string, unknown> = {};
    const input = [{ role: "user", content: "from body" }];
    const fetcher = (async (_input, init) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ output_text: "ok", status: "completed" });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await adapter.complete({ body: { input, store: false } });

    expect(body.input).toEqual(input);
    expect(body.store).toBe(false);
  });

  test("extracts reasoning summaries from Responses output items", async () => {
    const fetcher = (async () => jsonResponse({
      status: "completed",
      output: [
        {
          type: "reasoning",
          summary: [
            { type: "summary_text", text: "First step. " },
            { type: "summary_text", text: "Second step." },
          ],
        },
        { type: "message", content: [{ type: "output_text", text: "done" }] },
      ],
    })) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });

    expect(result.text).toBe("done");
    expect(result.reasoning).toBe("First step. Second step.");
  });

  test("keeps a string prompt valid across MCP tool rounds", async () => {
    const inputs: unknown[] = [];
    let call = 0;
    const fetcher = (async (_input, init) => {
      call += 1;
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      inputs.push(body.input);
      if (call === 1) {
        return sseResponse([{
          type: "response.completed",
          response: {
            status: "completed",
            output: [{ type: "function_call", id: "fc_1", call_id: "call_1", name: "add", arguments: "{}" }],
          },
        }]);
      }
      return sseResponse([
        { type: "response.output_text.delta", delta: "done" },
        { type: "response.completed", response: { status: "completed", output_text: "done" } },
      ]);
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });
    const mcpClient = {
      id: "calc",
      listTools: async () => ({ tools: [{ name: "add", inputSchema: { type: "object" } }] }),
      callTool: async () => ({ content: [{ type: "text", text: "3" }] }),
    };

    const result = await adapter.stream!({ prompt: "Add 1 and 2", mcpClients: [mcpClient] });

    expect(result.text).toBe("done");
    expect(inputs[0]).toBe("Add 1 and 2");
    const secondInput = inputs[1] as Array<Record<string, unknown>>;
    // Every item of the second-round input array must be an object; a bare
    // string is rejected by Responses servers ("Cannot determine type of 'item'").
    expect(secondInput.every((item) => typeof item === "object" && item !== null)).toBe(true);
    expect(secondInput[0]).toEqual({ role: "user", content: "Add 1 and 2" });
    expect(secondInput.at(-1)).toMatchObject({ type: "function_call_output", call_id: "call_1" });
  });

  test("does not surface non-terminal statuses as finishReason", async () => {
    const chunks: Array<{ finishReason?: string }> = [];
    const fetcher = (async () => sseResponse([
      { type: "response.created", response: { status: "in_progress" } },
      { type: "response.in_progress", response: { status: "in_progress" } },
      { type: "response.output_text.delta", delta: "hi" },
      { type: "response.completed", response: { status: "completed", output_text: "hi" } },
    ])) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.stream!(
      { prompt: "test" },
      { onChunk: (chunk) => chunks.push({ finishReason: chunk.finishReason }) },
    );

    expect(result.finishReason).toBe("completed");
    const reasons = chunks.map((chunk) => chunk.finishReason).filter(Boolean);
    expect(reasons).toEqual(["completed"]);
  });

  test("throws for typed stream failures", async () => {
    const fetcher = (async () => sseResponse([{
      type: "response.failed",
      response: { error: { message: "model unavailable" } },
    }])) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await expect(adapter.stream!({ prompt: "test" })).rejects.toThrow("model unavailable");
  });

  test("throws for failed and malformed non-streaming responses", async () => {
    let call = 0;
    const fetcher = (async () => {
      call += 1;
      return call === 1
        ? jsonResponse({ status: "failed", error: { message: "request failed" } })
        : new Response("{ invalid", { status: 200 });
    }) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await expect(adapter.complete({ prompt: "test" })).rejects.toThrow("request failed");
    await expect(adapter.complete({ prompt: "test" })).rejects.toThrow(
      "Failed to parse OpenAI-compatible JSON response",
    );
  });

  test("validates topLogprobs before either adapter sends a request", async () => {
    let fetchCalls = 0;
    const fetcher = (async () => {
      fetchCalls += 1;
      return jsonResponse({ output_text: "unused" });
    }) as unknown as typeof fetch;
    const adapters = [
      createOpenAICompatibleAdapter({ baseURL: "https://example.com", model: "test-model", fetcher }),
      createOpenAICompatibleLegacyAdapter({ baseURL: "https://example.com", model: "test-model", fetcher }),
    ];

    for (const adapter of adapters) {
      for (const topLogprobs of [-1, 1.5, 21, Number.NaN]) {
        await expect(adapter.complete({ prompt: "test", topLogprobs })).rejects.toThrow(
          "topLogprobs must be an integer between 0 and 20",
        );
      }
    }
    expect(fetchCalls).toBe(0);
  });

  test("enforces maxToolRounds in non-streaming Responses MCP mode", async () => {
    const fetcher = (async () => jsonResponse({
      status: "requires_action",
      output: [{
        type: "function_call",
        call_id: "call_1",
        name: "add",
        arguments: "{}",
      }],
    })) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });
    const mcpClient = {
      id: "calc",
      listTools: async () => ({ tools: [{ name: "add", inputSchema: { type: "object" } }] }),
      callTool: async () => ({ content: [{ type: "text", text: "3" }] }),
    };

    await expect(adapter.complete({
      prompt: "Add",
      mcpClients: [mcpClient],
      maxToolRounds: 0,
    })).rejects.toThrow("Tool call loop exceeded maxToolRounds (0)");
  });
});

describe("openai-compatible-legacy contract", () => {
  test("uses Chat Completions and maps current request parameters", async () => {
    let url = "";
    let body: Record<string, unknown> = {};
    const fetcher = (async (input, init) => {
      url = String(input);
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ choices: [{ message: { role: "assistant", content: "ok" } }] });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleLegacyAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      defaultBody: { max_tokens: 64 },
      fetcher,
    });

    await adapter.complete({
      prompt: "test",
      topLogprobs: 4,
      maxTokens: 32,
      reasoningEffort: "max",
    });
    expect(url).toBe("https://example.com/v1/chat/completions");
    expect(body).toMatchObject({
      logprobs: true,
      top_logprobs: 4,
      max_completion_tokens: 32,
      reasoning_effort: "max",
    });
    expect(body).not.toHaveProperty("max_tokens");
  });

  test("keeps xhigh and max as distinct reasoning effort values", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher = (async (_input, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return jsonResponse({ choices: [{ message: { role: "assistant", content: "ok" } }] });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleLegacyAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await adapter.complete({ prompt: "test", reasoningEffort: "xhigh" });
    await adapter.complete({ prompt: "test", reasoningEffort: "max" });

    expect(requests[0]?.reasoning_effort).toBe("xhigh");
    expect(requests[1]?.reasoning_effort).toBe("max");
  });

  test("returns Chat Completions refusal content as text", async () => {
    const fetcher = (async () => jsonResponse({
      choices: [{
        finish_reason: "content_filter",
        message: { role: "assistant", content: null, refusal: "I cannot help with that." },
      }],
    })) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleLegacyAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.complete({ prompt: "test" });

    expect(result.text).toBe("I cannot help with that.");
  });

  test("retains embeddings support", async () => {
    const fetcher = (async (input) => {
      expect(String(input)).toBe("https://example.com/v1/embeddings");
      return jsonResponse({ model: "embed", data: [{ embedding: [0.1, 0.2] }] });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleLegacyAdapter({
      baseURL: "https://example.com",
      model: "embed",
      fetcher,
    });

    await expect(adapter.embed!({ input: "hello" })).resolves.toMatchObject({
      embeddings: [[0.1, 0.2]],
      model: "embed",
    });
  });
});

describe("openai-compatible Responses tool-message conversion", () => {
  test("converts chat-style tool history to function_call / function_call_output items", async () => {
    let body: Record<string, unknown> = {};
    const fetcher = (async (_input, init) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ output_text: "Bring an umbrella.", status: "completed" });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    const result = await adapter.complete({
      messages: [
        { role: "system", content: "You are a weather assistant." },
        { role: "user", content: "Weather in Paris?" },
        {
          role: "assistant",
          content: "",
          tool_calls: [{
            id: "call_weather",
            type: "function",
            function: { name: "get_weather", arguments: "{\"city\":\"Paris\"}" },
          }],
        },
        { role: "tool", content: "{\"temp\":18}", tool_call_id: "call_weather" },
        { role: "user", content: "Should I bring an umbrella?" },
      ],
    });

    expect(result.text).toBe("Bring an umbrella.");
    expect(body.input).toEqual([
      { role: "system", content: "You are a weather assistant." },
      { role: "user", content: "Weather in Paris?" },
      { type: "function_call", call_id: "call_weather", name: "get_weather", arguments: "{\"city\":\"Paris\"}" },
      { type: "function_call_output", call_id: "call_weather", output: "{\"temp\":18}" },
      { role: "user", content: "Should I bring an umbrella?" },
    ]);
  });

  test("keeps assistant text alongside converted tool calls and serializes object arguments", async () => {
    let body: Record<string, unknown> = {};
    const fetcher = (async (_input, init) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ output_text: "ok", status: "completed" });
    }) as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "test-model",
      fetcher,
    });

    await adapter.complete({
      messages: [
        { role: "user", content: "Compute" },
        {
          role: "assistant",
          content: "Let me check.",
          tool_calls: [{
            id: "call_1",
            type: "function",
            function: { name: "sum", arguments: { a: 1, b: 2 } },
          }],
        },
        { role: "tool", content: "3", tool_call_id: "call_1" },
      ],
    });

    expect(body.input).toEqual([
      { role: "user", content: "Compute" },
      { role: "assistant", content: "Let me check." },
      { type: "function_call", call_id: "call_1", name: "sum", arguments: "{\"a\":1,\"b\":2}" },
      { type: "function_call_output", call_id: "call_1", output: "3" },
    ]);
  });
});
