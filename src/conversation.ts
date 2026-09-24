import { images, type SyncImageSource } from "./image";
import type { LLMMessage, LLMToolCallRef } from "./types";

export type ConversationEntry =
  | { role: "user"; text: string; images?: SyncImageSource[] }
  | { role: "assistant"; text: string; images?: SyncImageSource[] }
  | { role: "tool_call"; id: string; name: string; arguments?: Record<string, unknown> }
  | { role: "tool_result"; id: string; output: unknown };

export function conversation(systemPrompt: string, entries: ConversationEntry[]): LLMMessage[] {
  const messages: LLMMessage[] = [{ role: "system", content: systemPrompt }];

  for (const entry of entries) {
    if (entry.role === "tool_call") {
      const toolCall: LLMToolCallRef = {
        id: entry.id,
        type: "function",
        function: { name: entry.name, arguments: JSON.stringify(entry.arguments ?? {}) },
      };
      // Consecutive calls, and the text the assistant wrote just before them,
      // are one assistant turn. Split into several assistant messages, the first
      // call would be followed by another assistant message instead of its
      // result, which Chat Completions and Anthropic both reject.
      const previous = messages.at(-1);
      if (previous?.role === "assistant" && Array.isArray(previous.tool_calls)) {
        previous.tool_calls.push(toolCall);
      } else if (previous?.role === "assistant" && typeof previous.content === "string") {
        previous.tool_calls = [toolCall];
      } else {
        messages.push({ role: "assistant", content: "", tool_calls: [toolCall] });
      }
      continue;
    }

    if (entry.role === "tool_result") {
      messages.push({
        role: "tool",
        content: typeof entry.output === "string" ? entry.output : JSON.stringify(entry.output ?? null),
        tool_call_id: entry.id,
      });
      continue;
    }

    messages.push({
      role: entry.role,
      content:
        entry.images && entry.images.length > 0
          ? [{ type: "text" as const, text: entry.text }, ...images(entry.images)]
          : entry.text,
    });
  }

  return messages;
}
