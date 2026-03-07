import { describe, expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { createMCPClient, wrapMCPClient } from "@/mcp";
import { z } from "zod";

describe("MCP client helpers", () => {
  test("createMCPClient() connects to an SDK server and exposes listTools/callTool", async () => {
    const server = new McpServer({
      name: "test-server",
      version: "1.0.0",
    });

    server.registerTool(
      "add",
      {
        inputSchema: {
          a: z.number(),
          b: z.number(),
        },
      },
      async ({ a, b }) => ({
        content: [{ type: "text", text: String(a + b) }],
        structuredContent: { result: a + b },
      }),
    );

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);

    const mcp = await createMCPClient({
      id: "calculator",
      transport: {
        type: "in-memory",
        transport: clientTransport,
      },
    });

    const listed = await mcp.listTools();
    expect(listed.tools.map((tool) => tool.name)).toEqual(["add"]);

    const result = (await mcp.callTool({ name: "add", arguments: { a: 2, b: 5 } })) as {
      structuredContent?: { result?: number };
    };

    expect(result.structuredContent?.result).toBe(7);

    await mcp.close?.();
    await server.close();
  });

  test("wrapMCPClient() adapts an already connected SDK client", async () => {
    const server = new McpServer({
      name: "test-server",
      version: "1.0.0",
    });

    server.registerTool(
      "ping",
      {
        inputSchema: {},
      },
      async () => ({
        content: [{ type: "text", text: "pong" }],
      }),
    );

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);

    const client = new Client({ name: "wrapper-test", version: "1.0.0" }, { capabilities: {} });
    await client.connect(clientTransport);

    const wrapped = wrapMCPClient({
      id: "wrapped",
      client,
      transport: clientTransport,
    });

    const tools = await wrapped.listTools();
    expect(tools.tools[0]?.name).toBe("ping");

    await wrapped.close?.();
    await server.close();
  });

  test("createMCPClient() supports stdio transport against a real MCP server process", async () => {
    const scriptPath = new URL("../examples/calculator-mcp-server.ts", import.meta.url).pathname;
    const mcp = await createMCPClient({
      id: "calculator-stdio",
      transport: {
        type: "stdio",
        command: process.execPath,
        args: [scriptPath],
      },
    });

    try {
      const tools = await mcp.listTools();
      expect(tools.tools.some((tool) => tool.name === "calculate")).toBe(true);

      const result = (await mcp.callTool({
        name: "calculate",
        arguments: { operation: "multiply", a: 6, b: 7 },
      })) as {
        structuredContent?: { result?: number };
      };

      expect(result.structuredContent?.result).toBe(42);
    } finally {
      await mcp.close?.();
    }
  });

  test("createMCPClient() uses streamable-http transport and forwards connect failures", async () => {
    const fetchCalls: Array<{ input: string; method: string }> = [];

    await expect(
      createMCPClient({
        id: "streamable-http",
        transport: {
          type: "streamable-http",
          url: "http://mcp.invalid/stream",
          options: {
            fetch: async (input, init) => {
              fetchCalls.push({
                input: typeof input === "string" ? input : input.toString(),
                method: init?.method ?? "GET",
              });
              throw new Error("network disabled for test");
            },
            reconnectionOptions: {
              maxRetries: 0,
              initialReconnectionDelay: 1,
              maxReconnectionDelay: 1,
              reconnectionDelayGrowFactor: 1,
            },
          },
        },
      }),
    ).rejects.toThrow();

    expect(fetchCalls.length).toBeGreaterThan(0);
  });
});
