/**
 * Streaming Turns With Tools Example (real LLM call)
 *
 * Demonstrates:
 * - `turnIndex` on streaming events while MCP tool rounds run
 * - `snapshot.reasoningBlocks` as a cumulative UI-friendly state
 * - `onTurnTransition` for closing/opening per-turn UI sections
 * - Final `result.reasoningBlocks`
 *
 * Usage: bun run dev streaming-turns-with-tools
 */

import { createLLM, createMCPClient, type ReasoningBlock, type StreamTurnTransition } from "@/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "anthropic-compatible";

const model = process.env.LLM_MODEL ?? "gpt-5-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;
const debugEnabled = process.env.STRUCTURED_DEBUG === "1";

if (!apiKey) {
  console.error("Missing LLM_API_KEY in environment.");
  console.error("Set it before running: bun run dev streaming-turns-with-tools");
  process.exit(1);
}

const llm = createLLM({
  provider,
  model,
  transport: {
    baseURL,
    apiKey,
  },
  defaults: {
    debug: debugEnabled,
  },
});

if (!llm.adapter.stream) {
  console.error(`Provider "${provider}" does not support streaming.`);
  process.exit(1);
}

const calculatorMCP = await createMCPClient({
  id: "calculator",
  transport: {
    type: "stdio",
    command: "bun",
    args: ["run", "examples/calculator-mcp-server.ts"],
  },
});

const transitions: StreamTurnTransition[] = [];
const streamedText: string[] = [];
const streamedReasoningBlocks: ReasoningBlock[][] = [];
const reasoningByTurn = new Map<number, string>();
const printedReasoningTurns = new Set<number>();
let currentlyStreamingReasoning = false;
const useColor = process.env.NO_COLOR !== "1";

console.log(bold("Streaming turns with tools"));
console.log(dim(`Provider: ${provider}`));
console.log(dim(`Model: ${model}`));
console.log(`\n${bold("Visible stream")}\n`);

try {
  const result = await llm.generate(
    [
      "Use the calculate MCP tool twice.",
      "First compute 9 * 8.",
      "Then compute that result + 11.",
      "Do not do mental math; use the tool for both computations.",
      "Reply with one concise sentence containing the final number.",
    ].join("\n"),
    {
      request: {
        temperature: 0,
        mcpClients: [calculatorMCP],
        maxToolRounds: 6,
        toolDebug: debugEnabled,
      },
      stream: {
        enabled: true,
        onTurnTransition: (transition) => {
          transitions.push(transition);

          const label = `turn ${transition.turnIndex}: ${transition.kind}`;
          if (transition.kind === "reasoningComplete") {
            const text = transition.reasoningText?.trim() || reasoningByTurn.get(transition.turnIndex)?.trim();
            if (currentlyStreamingReasoning) {
              process.stdout.write("\n");
              currentlyStreamingReasoning = false;
            }

            if (text && !printedReasoningTurns.has(transition.turnIndex)) {
              writeHeader(label, "reasoning");
              process.stdout.write(`${cyan(indent(text))}\n`);
              return;
            }
            return;
          }

          if (transition.kind === "toolCallsEmit") {
            const names = transition.toolCalls
              ?.map((call) => call.name ?? call.id)
              .filter(Boolean)
              .join(", ");
            writeHeader(`${label}${names ? ` -> ${names}` : ""}`, "tool");
            return;
          }

          writeHeader(label, transition.kind === "toolResultsReceived" ? "result" : "end");
        },
        onData: (event) => {
          if (event.delta.text) {
            if (currentlyStreamingReasoning) {
              currentlyStreamingReasoning = false;
              process.stdout.write("\n");
              writeHeader("visible answer", "visible");
            }
            streamedText.push(event.delta.text);
            process.stdout.write(green(event.delta.text));
          }

          if (event.snapshot.reasoningBlocks) {
            streamedReasoningBlocks.push(event.snapshot.reasoningBlocks);
          }

          if (event.delta.reasoning) {
            const turnIndex = event.turnIndex ?? 0;
            if (!printedReasoningTurns.has(turnIndex)) {
              printedReasoningTurns.add(turnIndex);
              writeHeader(`turn ${turnIndex}: reasoning stream`, "reasoning");
            }
            reasoningByTurn.set(turnIndex, `${reasoningByTurn.get(turnIndex) ?? ""}${event.delta.reasoning}`);
            process.stdout.write(cyan(event.delta.reasoning));
            currentlyStreamingReasoning = true;
          }
        },
      },
    },
  );

  console.log(`\n\n${bold("Turn transitions")}`);
  for (const transition of transitions) {
    console.log(dim(
      `- turn ${transition.turnIndex}: ${transition.kind}`
      + (transition.toolCalls ? ` (${transition.toolCalls.length} tool call(s))` : ""),
    ));
  }

  console.log(`\n${bold("Reasoning blocks")}`);
  for (const block of result.reasoningBlocks ?? []) {
    console.log(cyan(`- turn ${block.turnIndex}: ${JSON.stringify(block.text)}`));
  }

  console.log(`\n${bold("Snapshot block counts observed")}`);
  console.log(dim(streamedReasoningBlocks.map((blocks) => blocks.length).join(" -> ") || "(none)"));

  console.log(`\n${bold("Final text")}: ${green(JSON.stringify(result.text))}`);
  console.log(`${bold("Streamed text")}: ${green(JSON.stringify(streamedText.join("")))}`);
} catch (error) {
  console.error(`\n${red("Streaming turns-with-tools example failed:")}`, error);
  process.exit(1);
} finally {
  await calculatorMCP.close?.();
}

function indent(text: string): string {
  return text
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n");
}

function writeHeader(
  label: string,
  tone: "reasoning" | "tool" | "result" | "visible" | "end",
): void {
  const colorize = {
    reasoning: brightCyan,
    tool: brightMagenta,
    result: brightYellow,
    visible: brightGreen,
    end: dim,
  }[tone];

  process.stdout.write(`\n${colorize(`━━ ${label} ━━`)}\n`);
}

function color(code: number | string, text: string): string {
  return useColor ? `\x1b[${code}m${text}\x1b[0m` : text;
}

function bold(text: string): string {
  return color(1, text);
}

function dim(text: string): string {
  return color(90, text);
}

function green(text: string): string {
  return color(32, text);
}

function brightGreen(text: string): string {
  return color("1;92", text);
}

function cyan(text: string): string {
  return color(36, text);
}

function brightCyan(text: string): string {
  return color("1;96", text);
}

function yellow(text: string): string {
  return color(33, text);
}

function brightYellow(text: string): string {
  return color("1;93", text);
}

function magenta(text: string): string {
  return color(35, text);
}

function brightMagenta(text: string): string {
  return color("1;95", text);
}

function red(text: string): string {
  return color(31, text);
}
