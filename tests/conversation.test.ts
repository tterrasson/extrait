import { describe, expect, test } from "bun:test";
import { conversation } from "@/conversation";

describe("conversation()", () => {
  test("builds messages with system prompt first", () => {
    const messages = conversation("You are a helpful assistant.", [
      { role: "user", text: "Hello" },
      { role: "assistant", text: "Hi there!" },
      { role: "user", text: "What is 2+2?" },
    ]);

    expect(messages).toEqual([
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there!" },
      { role: "user", content: "What is 2+2?" },
    ]);
  });

  test("embeds images in user messages when provided", () => {
    const img = { base64: "abc123", mimeType: "image/png" };
    const messages = conversation("You are a vision assistant.", [
      { role: "user", text: "Describe this image.", images: [img] },
      { role: "assistant", text: "I see a cat." },
    ]);

    expect(messages).toEqual([
      { role: "system", content: "You are a vision assistant." },
      {
        role: "user",
        content: [
          { type: "text", text: "Describe this image." },
          { type: "image_url", image_url: { url: "data:image/png;base64,abc123" } },
        ],
      },
      { role: "assistant", content: "I see a cat." },
    ]);
  });

  test("returns only system message when entries are empty", () => {
    const messages = conversation("System only.", []);

    expect(messages).toEqual([{ role: "system", content: "System only." }]);
  });

  test("uses plain string content when no images are provided", () => {
    const messages = conversation("sys", [{ role: "user", text: "Hello" }]);

    expect(messages[1]).toEqual({ role: "user", content: "Hello" });
  });

  test("uses plain string content when images array is empty", () => {
    const messages = conversation("sys", [{ role: "user", text: "Hello", images: [] }]);

    expect(messages[1]).toEqual({ role: "user", content: "Hello" });
  });
});
