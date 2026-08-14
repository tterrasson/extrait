import { images, type SyncImageSource } from "./image";
import type { LLMMessage } from "./types";

export type ConversationEntry =
  | { role: "user"; text: string; images?: SyncImageSource[] }
  | { role: "assistant"; text: string; images?: SyncImageSource[] }
  | { role: "tool_call"; id: string; name: string; arguments?: Record<string, unknown> }
  | { role: "tool_result"; id: string; output: unknown };

export function conversation(systemPrompt: string, entries: ConversationEntry[]): LLMMessage[] {
  return [
    { role: "system", content: systemPrompt },
    ...entries.map((entry): LLMMessage => {
      if (entry.role === "tool_call") {
        return {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: entry.id,
              type: "function",
              function: { name: entry.name, arguments: JSON.stringify(entry.arguments ?? {}) },
            },
          ],
        };
      }
      if (entry.role === "tool_result") {
        return {
          role: "tool",
          content: typeof entry.output === "string" ? entry.output : JSON.stringify(entry.output),
          tool_call_id: entry.id,
        };
      }
      return {
        role: entry.role,
        content:
          entry.images && entry.images.length > 0
            ? [{ type: "text" as const, text: entry.text }, ...images(entry.images)]
            : entry.text,
      };
    }),
  ];
}
