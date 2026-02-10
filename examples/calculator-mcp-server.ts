import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "calculator-mcp",
  version: "1.0.0",
});

server.registerTool(
  "calculate",
  {
    description: "Performs a simple mathematical calculation",
    inputSchema: {
      operation: z.enum(["add", "subtract", "multiply", "divide"]).describe("Operation to perform"),
      a: z.number().describe("First number"),
      b: z.number().describe("Second number"),
    },
  },
  async ({ operation, a, b }) => {
    let result: number;

    switch (operation) {
      case "add":
        result = a + b;
        break;
      case "subtract":
        result = a - b;
        break;
      case "multiply":
        result = a * b;
        break;
      case "divide":
        if (b === 0) {
          return {
            isError: true,
            content: [{ type: "text", text: "Division by zero is not allowed" }],
          };
        }
        result = a / b;
        break;
      default:
        return {
          isError: true,
          content: [{ type: "text", text: `Unsupported operation: ${operation}` }],
        };
    }

    return {
      content: [{ type: "text", text: String(result) }],
      structuredContent: { result },
    };
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
