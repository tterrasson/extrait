import { z } from "zod";
import { createLLM, createMCPClient, prompt, s, StructuredParseError } from "../src/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "anthropic-compatible";

const model = process.env.LLM_MODEL ?? "gpt-5-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;
const debugEnabled = process.env.STRUCTURED_DEBUG === "1";

const llm = createLLM({
  provider,
  model,
  transport: {
    baseURL,
    apiKey,
  },
  defaults: {
    mode: "loose",
    selfHeal: 1,
    debug: debugEnabled,
  },
});

const calculatorMCP = await createMCPClient({
  id: "calculator",
  transport: {
    type: "stdio",
    command: "bun",
    args: ["run", "examples/calculator-mcp-server.ts"],
  },
});

const toolExecutions: Array<{
  callId: string;
  name?: string;
  arguments?: unknown;
  output?: unknown;
  durationMs?: number;
  error?: string;
}> = [];

const ResultSchema = s.schema(
  "Result",
  z.object({
    result: s.number().describe("The calculation result"),
  }),
);

const userInput = process.argv.slice(3).join(" ").trim();
const question = userInput || "Calculate 15 multiplied by 7, then add 23 to the result";

console.log(`Question: ${question}\n`);

try {
  const result = await llm.structured(
    ResultSchema,
    prompt()
      .system`You are a precise calculator assistant.`
      .user`
        Calculate: """${question}"""

        Use the available MCP tools to perform the calculations.
      `,
    {
      request: {
        mcpClients: [calculatorMCP],
        maxToolRounds: 5,
        toolDebug: debugEnabled
          ? {
              enabled: true,
              includeRequest: true,
              includeResult: true,
              includeResultOnError: true,
              pretty: false,
            }
          : false,
        onToolExecution: (execution) => {
          toolExecutions.push({
            callId: execution.callId,
            name: execution.name,
            arguments: execution.arguments,
            output: execution.output,
            durationMs: execution.durationMs,
            error: execution.error,
          });
        },
      },
      stream: {
        to: "stdout",
        onData: (event) => {
          if (event.done) {
            console.log("\n[stream] snapshot complete:", JSON.stringify(event.data));
          }
        },
      },
    },
  );

  console.log("\n\n=== Result ===");
  console.log(`Answer: ${result.data.result}`);
  console.log("\nUsage:", result.usage ?? {});
  console.log("Finish reason:", result.finishReason ?? "unknown");
  console.log("Tool executions:", toolExecutions);
} catch (error) {
  if (error instanceof StructuredParseError) {
    console.error("Structured parsing failed.");
    console.error("Attempt:", error.attempt);
    console.error("Candidates:", error.candidates.length);
    console.error("Zod issues:", error.zodIssues ?? []);
    console.error("Repair log:", error.repairLog ?? []);
    process.exit(1);
  }

  throw error;
} finally {
  await calculatorMCP.close?.();
}
