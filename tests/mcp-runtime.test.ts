import { describe, expect, test } from "bun:test";
import {
  resolveMCPToolset,
  toProviderFunctionTools,
  executeMCPToolCalls,
  DEFAULT_MAX_TOOL_ROUNDS,
  normalizeMaxToolRounds,
  parseToolArguments,
  stringifyToolOutput,
  formatToolExecutionDebugLine,
  sanitizeToolName,
} from "@/providers/mcp-runtime";
import type { MCPToolClient, LLMToolExecution, MCPToolSchema } from "@/types";

function createMockClient(
  id: string,
  tools: Array<{ name: string; description?: string; inputSchema?: unknown }>,
  callImpl?: (params: { name: string; arguments: Record<string, unknown> }) => unknown,
): MCPToolClient {
  return {
    id,
    async listTools() {
      return {
        tools: tools.map((t) => ({
          name: t.name,
          description: t.description,
          inputSchema: (t.inputSchema ?? { type: "object", properties: {} }) as MCPToolSchema,
        })),
      };
    },
    async callTool(params) {
      if (callImpl) {
        return callImpl({ name: params.name, arguments: params.arguments ?? {} });
      }
      return { result: "ok" };
    },
  };
}

describe("resolveMCPToolset", () => {
  test("returns empty toolset for undefined", async () => {
    const result = await resolveMCPToolset(undefined);
    expect(result.tools).toEqual([]);
    expect(result.byName.size).toBe(0);
  });

  test("returns empty toolset for empty array", async () => {
    const result = await resolveMCPToolset([]);
    expect(result.tools).toEqual([]);
    expect(result.byName.size).toBe(0);
  });

  test("resolves a single client with one tool", async () => {
    const client = createMockClient("calc", [{ name: "add", description: "Add numbers" }]);
    const result = await resolveMCPToolset([client]);
    expect(result.tools.length).toBe(1);
    expect(result.tools[0]!.name).toBe("add");
    expect(result.tools[0]!.description).toBe("Add numbers");
    expect(result.byName.has("add")).toBe(true);
  });

  test("prefixes names on collision across clients", async () => {
    const client1 = createMockClient("alpha", [{ name: "run" }]);
    const client2 = createMockClient("beta", [{ name: "run" }]);
    const result = await resolveMCPToolset([client1, client2]);
    expect(result.tools.length).toBe(2);
    expect(result.tools[0]!.name).toBe("alpha__run");
    expect(result.tools[1]!.name).toBe("beta__run");
  });

  test("normalizes missing inputSchema", async () => {
    const client = createMockClient("svc", [{ name: "do", inputSchema: undefined }]);
    const result = await resolveMCPToolset([client]);
    expect(result.tools[0]!.inputSchema).toEqual({ type: "object", properties: {} });
  });

  test("normalizes schema without type", async () => {
    const client = createMockClient("svc", [
      { name: "do", inputSchema: { properties: { a: { type: "string" } } } },
    ]);
    const result = await resolveMCPToolset([client]);
    expect(result.tools[0]!.inputSchema.type).toBe("object");
  });

  test("normalizes schema without properties", async () => {
    const client = createMockClient("svc", [
      { name: "do", inputSchema: { type: "object" } },
    ]);
    const result = await resolveMCPToolset([client]);
    expect(result.tools[0]!.inputSchema.properties).toEqual({});
  });

  test("describeTool returns undefined when no collision and no description", async () => {
    const client = createMockClient("svc", [{ name: "action" }]);
    const result = await resolveMCPToolset([client]);
    expect(result.tools[0]!.description).toBeUndefined();
  });

  test("describeTool adds prefix on collision with description", async () => {
    const client1 = createMockClient("alpha", [{ name: "run", description: "Run task" }]);
    const client2 = createMockClient("beta", [{ name: "run", description: "Run job" }]);
    const result = await resolveMCPToolset([client1, client2]);
    expect(result.tools[0]!.description).toBe("[alpha] Run task");
    expect(result.tools[1]!.description).toBe("[beta] Run job");
  });

  test("describeTool adds prefix on collision without description", async () => {
    const client1 = createMockClient("alpha", [{ name: "run" }]);
    const client2 = createMockClient("beta", [{ name: "run" }]);
    const result = await resolveMCPToolset([client1, client2]);
    expect(result.tools[0]!.description).toBe("[alpha] run");
    expect(result.tools[1]!.description).toBe("[beta] run");
  });

  test("sanitizes tool names with special characters", async () => {
    const client = createMockClient("my-svc.v2", [{ name: "do-thing" }]);
    const result = await resolveMCPToolset([client]);
    expect(result.tools[0]!.name).toBe("do_thing");
  });

  test("sanitizes tool name starting with digit", async () => {
    const client = createMockClient("svc", [{ name: "1abc" }]);
    const result = await resolveMCPToolset([client]);
    expect(result.tools[0]!.name).toBe("tool_1abc");
  });
});

describe("toProviderFunctionTools", () => {
  test("returns undefined for empty toolset", () => {
    const result = toProviderFunctionTools({ tools: [], byName: new Map() });
    expect(result).toBeUndefined();
  });

  test("converts non-empty toolset to function tools", async () => {
    const client = createMockClient("calc", [{ name: "add", description: "Add" }]);
    const toolset = await resolveMCPToolset([client]);
    const result = toProviderFunctionTools(toolset);
    expect(result).toEqual([
      {
        type: "function",
        function: {
          name: "add",
          description: "Add",
          parameters: { type: "object", properties: {} },
        },
      },
    ]);
  });
});

describe("executeMCPToolCalls", () => {
  test("throws when call has no id", async () => {
    const client = createMockClient("svc", [{ name: "run" }]);
    const toolset = await resolveMCPToolset([client]);
    await expect(
      executeMCPToolCalls(
        [{ name: "run" }],
        toolset,
        { round: 1, request: { prompt: "test" } },
      ),
    ).rejects.toThrow("without id or name");
  });

  test("throws when call has no name", async () => {
    const client = createMockClient("svc", [{ name: "run" }]);
    const toolset = await resolveMCPToolset([client]);
    await expect(
      executeMCPToolCalls(
        [{ id: "c1" }],
        toolset,
        { round: 1, request: { prompt: "test" } },
      ),
    ).rejects.toThrow("without id or name");
  });

  test("returns local error execution when tool name is unknown", async () => {
    const client = createMockClient("svc", [{ name: "run" }]);
    const toolset = await resolveMCPToolset([client]);
    const results = await executeMCPToolCalls(
      [{ id: "c1", name: "unknown", arguments: '{"x":1}' }],
      toolset,
      { round: 1, request: { prompt: "test" } },
    );
    expect(results).toHaveLength(1);
    expect(results[0]!.call.error).toBe('Tool "unknown" is not registered in the current toolset.');
    expect(results[0]!.execution.error).toBe('Tool "unknown" is not registered in the current toolset.');
    expect(results[0]!.execution.clientId).toBe("__unregistered__");
    expect(results[0]!.execution.remoteName).toBe("unknown");
    expect(results[0]!.call.arguments).toEqual({ x: 1 });
  });

  test("uses custom unknownToolError message when tool name is unknown", async () => {
    const client = createMockClient("svc", [{ name: "run" }]);
    const toolset = await resolveMCPToolset([client]);
    const results = await executeMCPToolCalls(
      [{ id: "c1", name: "unknown" }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          unknownToolError: (toolName) => `Missing tool: ${toolName}`,
        },
      },
    );
    expect(results).toHaveLength(1);
    expect(results[0]!.call.error).toBe("Missing tool: unknown");
    expect(results[0]!.execution.error).toBe("Missing tool: unknown");
  });

  test("executes tool successfully", async () => {
    const client = createMockClient("svc", [{ name: "run" }], () => ({ done: true }));
    const toolset = await resolveMCPToolset([client]);
    const results = await executeMCPToolCalls(
      [{ id: "c1", name: "run", arguments: '{"x":1}' }],
      toolset,
      { round: 1, request: { prompt: "test" } },
    );
    expect(results.length).toBe(1);
    expect(results[0]!.execution.output).toEqual({ done: true });
    expect(results[0]!.execution.handledLocally).toBe(true);
    expect(results[0]!.execution.error).toBeUndefined();
  });

  test("captures error when callTool throws Error", async () => {
    const client = createMockClient("svc", [{ name: "run" }], () => {
      throw new Error("tool failed");
    });
    const toolset = await resolveMCPToolset([client]);
    const results = await executeMCPToolCalls(
      [{ id: "c1", name: "run" }],
      toolset,
      { round: 1, request: { prompt: "test" } },
    );
    expect(results[0]!.execution.error).toBe("tool failed");
    expect(results[0]!.call.error).toBe("tool failed");
  });

  test("captures error when callTool throws non-Error", async () => {
    const client = createMockClient("svc", [{ name: "run" }], () => {
      throw "oops";
    });
    const toolset = await resolveMCPToolset([client]);
    const results = await executeMCPToolCalls(
      [{ id: "c1", name: "run" }],
      toolset,
      { round: 1, request: { prompt: "test" } },
    );
    expect(results[0]!.execution.error).toBe("oops");
  });

  test("transformToolOutput transforms the output before sending to LLM", async () => {
    const client = createMockClient("svc", [{ name: "run" }], () => ({ raw: "verbose data" }));
    const toolset = await resolveMCPToolset([client]);
    const results = await executeMCPToolCalls(
      [{ id: "c1", name: "run" }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          transformToolOutput: (output) => ({ cleaned: true }),
        },
      },
    );
    expect(results[0]!.execution.output).toEqual({ cleaned: true });
    expect(results[0]!.call.output).toEqual({ cleaned: true });
  });

  test("transformToolOutput receives the raw output and execution context", async () => {
    let capturedOutput: unknown;
    let capturedContext: unknown;
    const client = createMockClient("svc", [{ name: "run" }], () => ({ value: 42 }));
    const toolset = await resolveMCPToolset([client]);
    await executeMCPToolCalls(
      [{ id: "c1", name: "run", arguments: '{"x":1}' }],
      toolset,
      {
        round: 2,
        request: {
          prompt: "test",
          transformToolOutput: (output, context) => {
            capturedOutput = output;
            capturedContext = context;
            return output;
          },
        },
        provider: "openai-compatible",
        model: "gpt-4",
      },
    );
    expect(capturedOutput).toEqual({ value: 42 });
    expect(capturedContext).toMatchObject({
      callId: "c1",
      name: "run",
      round: 2,
      provider: "openai-compatible",
      model: "gpt-4",
    });
  });

  test("transformToolOutput async works correctly", async () => {
    const client = createMockClient("svc", [{ name: "run" }], () => "original");
    const toolset = await resolveMCPToolset([client]);
    const results = await executeMCPToolCalls(
      [{ id: "c1", name: "run" }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          transformToolOutput: async (output) => {
            await new Promise((resolve) => setTimeout(resolve, 0));
            return "transformed";
          },
        },
      },
    );
    expect(results[0]!.execution.output).toBe("transformed");
  });

  test("transformToolArguments transforms args before callTool and receives context", async () => {
    let calledWith: { name: string; arguments: Record<string, unknown> } | undefined;
    const client = createMockClient("svc", [{ name: "do-work" }], (params) => {
      calledWith = params;
      return { ok: true };
    });
    const toolset = await resolveMCPToolset([client]);

    const results = await executeMCPToolCalls(
      [{ id: "c1", name: "do_work", arguments: '{"x":1}' }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          transformToolArguments: (args, context) => {
            expect(args).toEqual({ x: 1 });
            expect(context).toEqual({
              name: "do_work",
              remoteName: "do-work",
              clientId: "svc",
            });
            return { x: 2, injected: true };
          },
        },
      },
    );

    expect(calledWith).toEqual({
      name: "do-work",
      arguments: { x: 2, injected: true },
    });
    expect(results[0]!.call.arguments).toEqual({ x: 1 });
    expect(results[0]!.execution.arguments).toEqual({ x: 1 });
    expect(results[0]!.execution.output).toEqual({ ok: true });
  });

  test("transformToolCallParams transforms the full MCP params sent to callTool", async () => {
    let capturedParams: unknown;
    const client: MCPToolClient = {
      id: "svc",
      async listTools() {
        return { tools: [{ name: "do-work", inputSchema: { type: "object", properties: {} } as MCPToolSchema }] };
      },
      async callTool(params) {
        capturedParams = params;
        return { ok: true };
      },
    };
    const toolset = await resolveMCPToolset([client]);

    await executeMCPToolCalls(
      [{ id: "c1", name: "do_work", arguments: '{"x":1}' }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          transformToolCallParams: (params, context) => {
            expect(context).toEqual({ name: "do_work", remoteName: "do-work", clientId: "svc" });
            return { ...params, _meta: { source: "test", clientId: context.clientId } };
          },
        },
      },
    );

    expect(capturedParams).toEqual({
      name: "do-work",
      arguments: { x: 1 },
      _meta: { source: "test", clientId: "svc" },
    });
  });

  test("transformToolCallParams receives args already transformed by transformToolArguments", async () => {
    let capturedParams: unknown;
    const client: MCPToolClient = {
      id: "svc",
      async listTools() {
        return { tools: [{ name: "run", inputSchema: { type: "object", properties: {} } as MCPToolSchema }] };
      },
      async callTool(params) {
        capturedParams = params;
        return { ok: true };
      },
    };
    const toolset = await resolveMCPToolset([client]);

    await executeMCPToolCalls(
      [{ id: "c1", name: "run", arguments: '{"x":1}' }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          transformToolArguments: (args) => ({ ...args, injected: true }),
          transformToolCallParams: (params) => ({ ...params, _meta: { seen: params.arguments } }),
        },
      },
    );

    expect(capturedParams).toEqual({
      name: "run",
      arguments: { x: 1, injected: true },
      _meta: { seen: { x: 1, injected: true } },
    });
  });

  test("transformToolOutput is NOT called when the tool throws", async () => {
    let called = false;
    const client = createMockClient("svc", [{ name: "run" }], () => {
      throw new Error("tool failed");
    });
    const toolset = await resolveMCPToolset([client]);
    const results = await executeMCPToolCalls(
      [{ id: "c1", name: "run" }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          transformToolOutput: () => {
            called = true;
            return "should not be called";
          },
        },
      },
    );
    expect(called).toBe(false);
    expect(results[0]!.execution.error).toBe("tool failed");
  });

  test("invokes onToolExecution callback", async () => {
    const executions: LLMToolExecution[] = [];
    const client = createMockClient("svc", [{ name: "run" }]);
    const toolset = await resolveMCPToolset([client]);
    await executeMCPToolCalls(
      [{ id: "c1", name: "run" }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          onToolExecution: (exec) => executions.push(exec),
        },
      },
    );
    expect(executions.length).toBe(1);
    expect(executions[0]!.callId).toBe("c1");
  });

  test("emits debug lines when toolDebug is true", async () => {
    const logs: string[] = [];
    const client = createMockClient("svc", [{ name: "run" }]);
    const toolset = await resolveMCPToolset([client]);
    await executeMCPToolCalls(
      [{ id: "c1", name: "run", arguments: '{"x":1}' }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          toolDebug: {
            enabled: true,
            logger: (line) => logs.push(line),
          },
        },
        provider: "test",
        model: "m1",
      },
    );
    expect(logs.some((l) => l.includes("[tool:mcp:ok]"))).toBe(true);
    expect(logs.some((l) => l.includes("[tool:mcp:request]"))).toBe(true);
    expect(logs.some((l) => l.includes("[tool:mcp:result:ok]"))).toBe(true);
  });

  test("emits error debug lines on failure", async () => {
    const logs: string[] = [];
    const client = createMockClient("svc", [{ name: "run" }], () => {
      throw new Error("boom");
    });
    const toolset = await resolveMCPToolset([client]);
    await executeMCPToolCalls(
      [{ id: "c1", name: "run" }],
      toolset,
      {
        round: 1,
        request: {
          prompt: "test",
          toolDebug: {
            enabled: true,
            logger: (line) => logs.push(line),
            includeResultOnError: true,
          },
        },
      },
    );
    expect(logs.some((l) => l.includes("[tool:mcp:error]"))).toBe(true);
    expect(logs.some((l) => l.includes("[tool:mcp:result:error]"))).toBe(true);
  });
});

describe("normalizeMaxToolRounds", () => {
  test("returns DEFAULT_MAX_TOOL_ROUNDS for undefined", () => {
    expect(normalizeMaxToolRounds(undefined)).toBe(DEFAULT_MAX_TOOL_ROUNDS);
  });

  test("returns DEFAULT_MAX_TOOL_ROUNDS for NaN", () => {
    expect(normalizeMaxToolRounds(NaN)).toBe(DEFAULT_MAX_TOOL_ROUNDS);
  });

  test("returns DEFAULT_MAX_TOOL_ROUNDS for Infinity", () => {
    expect(normalizeMaxToolRounds(Infinity)).toBe(DEFAULT_MAX_TOOL_ROUNDS);
  });

  test("clamps negative to 0", () => {
    expect(normalizeMaxToolRounds(-3)).toBe(0);
  });

  test("floors fractional values", () => {
    expect(normalizeMaxToolRounds(2.7)).toBe(2);
  });

  test("passes through normal values", () => {
    expect(normalizeMaxToolRounds(5)).toBe(5);
  });

  test("returns 0 for 0", () => {
    expect(normalizeMaxToolRounds(0)).toBe(0);
  });
});

describe("parseToolArguments", () => {
  test("parses valid JSON string", () => {
    expect(parseToolArguments('{"a":1}')).toEqual({ a: 1 });
  });

  test("returns empty object for invalid JSON string", () => {
    expect(parseToolArguments("not json")).toEqual({});
  });

  test("returns object as-is", () => {
    expect(parseToolArguments({ a: 1 })).toEqual({ a: 1 });
  });

  test("returns empty object for undefined", () => {
    expect(parseToolArguments(undefined)).toEqual({});
  });

  test("returns empty object for null", () => {
    expect(parseToolArguments(null)).toEqual({});
  });
});

describe("stringifyToolOutput", () => {
  test("returns string as-is", () => {
    expect(stringifyToolOutput("hello")).toBe("hello");
  });

  test("JSON-stringifies object", () => {
    expect(stringifyToolOutput({ x: 1 })).toBe('{"x":1}');
  });

  test("returns null for undefined", () => {
    expect(stringifyToolOutput(undefined)).toBe("null");
  });

  test("returns null for null", () => {
    expect(stringifyToolOutput(null)).toBe("null");
  });
});

describe("sanitizeToolName", () => {
  test("leaves valid names unchanged", () => {
    expect(sanitizeToolName("my_tool")).toBe("my_tool");
    expect(sanitizeToolName("tool123")).toBe("tool123");
    expect(sanitizeToolName("ABC")).toBe("ABC");
  });

  test("replaces hyphens and dots with underscores", () => {
    expect(sanitizeToolName("my-tool")).toBe("my_tool");
    expect(sanitizeToolName("my.tool")).toBe("my_tool");
    expect(sanitizeToolName("my-tool.v2")).toBe("my_tool_v2");
  });

  test("collapses multiple consecutive special characters into one underscore", () => {
    expect(sanitizeToolName("my--tool")).toBe("my_tool");
    expect(sanitizeToolName("a...b")).toBe("a_b");
  });

  test("strips leading and trailing underscores", () => {
    expect(sanitizeToolName("-leading")).toBe("leading");
    expect(sanitizeToolName("trailing-")).toBe("trailing");
    expect(sanitizeToolName("-both-")).toBe("both");
  });

  test("prefixes tool_ when name starts with a digit", () => {
    expect(sanitizeToolName("1abc")).toBe("tool_1abc");
    expect(sanitizeToolName("9")).toBe("tool_9");
  });

  test("returns 'tool' for empty or all-special-character input", () => {
    expect(sanitizeToolName("")).toBe("tool");
    expect(sanitizeToolName("---")).toBe("tool");
    expect(sanitizeToolName("...")).toBe("tool");
  });
});

describe("dynamic tool discovery", () => {
  test("resolveMCPToolset called each round sees newly available tools", async () => {
    let callCount = 0;
    const client: MCPToolClient = {
      id: "dynamic",
      async listTools() {
        callCount += 1;
        if (callCount === 1) {
          return { tools: [{ name: "tool_get", description: "Bootstrap", inputSchema: { type: "object", properties: {} } as MCPToolSchema }] };
        }
        return {
          tools: [
            { name: "tool_get", description: "Bootstrap", inputSchema: { type: "object", properties: {} } as MCPToolSchema },
            { name: "websearch", description: "Search the web", inputSchema: { type: "object", properties: {} } as MCPToolSchema },
          ],
        };
      },
      async callTool() {
        return { result: "ok" };
      },
    };

    // Round 1: only tool_get visible
    const toolset1 = await resolveMCPToolset([client]);
    expect(toolset1.tools.map((t) => t.name)).toEqual(["tool_get"]);
    expect(toolset1.byName.has("websearch")).toBe(false);

    // Round 2: websearch now visible
    const toolset2 = await resolveMCPToolset([client]);
    expect(toolset2.tools.map((t) => t.name)).toContain("websearch");
    expect(toolset2.byName.has("websearch")).toBe(true);
  });
});

describe("formatToolExecutionDebugLine", () => {
  test("formats successful execution", () => {
    const execution: LLMToolExecution = {
      callId: "c1",
      type: "function",
      name: "run",
      clientId: "svc",
      handledLocally: true,
      provider: "openai",
      model: "gpt-4",
      startedAt: new Date().toISOString(),
      durationMs: 42,
    };
    const result = formatToolExecutionDebugLine(execution);
    expect(result).toContain("[tool:mcp:ok]");
    expect(result).toContain("openai/gpt-4");
    expect(result).toContain("svc:run");
    expect(result).toContain("42ms");
  });

  test("formats error execution", () => {
    const execution: LLMToolExecution = {
      callId: "c1",
      type: "function",
      name: "run",
      clientId: "svc",
      handledLocally: true,
      startedAt: new Date().toISOString(),
      error: "boom",
    };
    const result = formatToolExecutionDebugLine(execution);
    expect(result).toContain("[tool:mcp:error]");
    expect(result).toContain("-> boom");
  });

  test("shows unknown for missing provider/model", () => {
    const execution: LLMToolExecution = {
      callId: "c1",
      type: "function",
      name: "run",
      handledLocally: true,
      startedAt: new Date().toISOString(),
    };
    const result = formatToolExecutionDebugLine(execution);
    expect(result).toContain("unknown");
  });

  test("uses name only when no clientId", () => {
    const execution: LLMToolExecution = {
      callId: "c1",
      type: "function",
      name: "run",
      handledLocally: true,
      provider: "p",
      startedAt: new Date().toISOString(),
    };
    const result = formatToolExecutionDebugLine(execution);
    expect(result).toContain(" run#c1");
    // No clientId:name pattern, just name#callId
    expect(result).not.toMatch(/\brun:.*#c1/);
  });
});
