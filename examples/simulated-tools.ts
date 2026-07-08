/**
 * Simulated Tools Example
 *
 * Demonstrates:
 * - Injecting fake tool calls into a conversation using `tool_call` and `tool_result` entries
 * - The LLM receives the context as if it had previously called a tool and gotten a result
 * - No actual MCP or tool execution happens — it's pure context injection
 *
 * Usage: bun run dev simulated-tools
 */

import { conversation, createLLM } from "@/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "openai-compatible-legacy"
  | "anthropic-compatible";
const debugEnabled = process.env.STRUCTURED_DEBUG === "1";

const llm = createLLM({
  provider,
  model: process.env.LLM_MODEL ?? "gpt-4o-mini",
  transport: { baseURL: process.env.LLM_BASE_URL, apiKey: process.env.LLM_API_KEY },
  defaults: { debug: debugEnabled },
});

// Build a conversation where the assistant "already called" a get_weather tool
// and received a result — this context is injected, not actually executed.
const messages = conversation("You are a helpful weather assistant.", [
  { role: "user", text: "What's the weather like in Paris right now?" },
  {
    role: "tool_call",
    id: "call_weather_paris",
    name: "get_weather",
    arguments: { city: "Paris", unit: "celsius" },
  },
  {
    role: "tool_result",
    id: "call_weather_paris",
    output: {
      city: "Paris",
      temperature: 18,
      unit: "celsius",
      condition: "partly cloudy",
      humidity: 62,
    },
  },
  { role: "user", text: "Should I bring an umbrella?" },
]);

console.log("=== Simulated tool call context ===\n");
console.log("Messages sent to LLM:");
console.log(JSON.stringify(messages, null, 2));
console.log("\n=== LLM response ===\n");

const response = await llm.generate({ prompt: { messages } });
console.log(response.text);
