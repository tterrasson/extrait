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

  test("tool_call entry produces assistant message with tool_calls", () => {
    const messages = conversation("sys", [
      { role: "tool_call", id: "call_1", name: "get_weather", arguments: { city: "Paris" } },
    ]);

    expect(messages[1]).toEqual({
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "get_weather", arguments: '{"city":"Paris"}' },
        },
      ],
    });
  });

  test("tool_call entry with no arguments defaults to empty object", () => {
    const messages = conversation("sys", [{ role: "tool_call", id: "call_2", name: "ping" }]);

    expect(messages[1]).toEqual({
      role: "assistant",
      content: "",
      tool_calls: [{ id: "call_2", type: "function", function: { name: "ping", arguments: "{}" } }],
    });
  });

  test("tool_result entry produces tool message with tool_call_id and stringified output", () => {
    const messages = conversation("sys", [
      { role: "tool_result", id: "call_1", output: { temp: 18, unit: "C" } },
    ]);

    expect(messages[1]).toEqual({
      role: "tool",
      content: '{"temp":18,"unit":"C"}',
      tool_call_id: "call_1",
    });
  });

  test("tool_result with string output is not double-stringified", () => {
    const messages = conversation("sys", [{ role: "tool_result", id: "call_1", output: "ok" }]);

    expect(messages[1]).toEqual({ role: "tool", content: "ok", tool_call_id: "call_1" });
  });

  test("mixed sequence: user → tool_call → tool_result → user", () => {
    const messages = conversation("sys", [
      { role: "user", text: "What is the weather in Paris?" },
      { role: "tool_call", id: "call_1", name: "get_weather", arguments: { city: "Paris" } },
      { role: "tool_result", id: "call_1", output: { temp: 18 } },
      { role: "user", text: "Thanks!" },
    ]);

    expect(messages).toEqual([
      { role: "system", content: "sys" },
      { role: "user", content: "What is the weather in Paris?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } }],
      },
      { role: "tool", content: '{"temp":18}', tool_call_id: "call_1" },
      { role: "user", content: "Thanks!" },
    ]);
  });
});
