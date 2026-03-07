import { images, type ImageInput } from "./image";
import type { LLMMessage } from "./types";

export interface ConversationEntry {
  role: "user" | "assistant";
  text: string;
  images?: ImageInput[];
}

export function conversation(systemPrompt: string, entries: ConversationEntry[]): LLMMessage[] {
  return [
    { role: "system", content: systemPrompt },
    ...entries.map((entry) => ({
      role: entry.role,
      content:
        entry.images && entry.images.length > 0
          ? [{ type: "text" as const, text: entry.text }, ...images(entry.images)]
          : entry.text,
    })),
  ];
}
