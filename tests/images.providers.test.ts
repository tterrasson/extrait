import { describe, expect, test } from "bun:test";
import { images } from "@/image";
import { createAnthropicCompatibleAdapter } from "@/providers/anthropic-compatible";
import { createOpenAICompatibleAdapter } from "@/providers/openai-compatible";
import { createOpenAICompatibleLegacyAdapter } from "@/providers/openai-compatible-legacy";

const REMOTE_URL = "https://example.com/photo.png";
const DATA_URL = "data:image/png;base64,AAAA";

function captureBody(payload: Record<string, unknown>) {
  const captured: { body: Record<string, unknown> } = { body: {} };
  const fetcher = (async (_input, init) => {
    captured.body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    return new Response(JSON.stringify(payload), { status: 200 });
  }) as typeof fetch;
  return { captured, fetcher };
}

const imageMessages = [
  {
    role: "user" as const,
    content: [{ type: "text" as const, text: "Compare." }, ...images([REMOTE_URL, DATA_URL])],
  },
];

describe("image passthrough reaches provider payloads", () => {
  test("openai-compatible forwards both URLs as input_image", async () => {
    const { captured, fetcher } = captureBody({
      status: "completed",
      output: [{ type: "message", content: [{ type: "output_text", text: "ok" }] }],
    });

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });
    await adapter.complete({ messages: imageMessages });

    const input = captured.body.input as Array<{ content: Array<Record<string, unknown>> }>;
    expect(input[0]?.content).toEqual([
      { type: "input_text", text: "Compare." },
      { type: "input_image", image_url: REMOTE_URL },
      { type: "input_image", image_url: DATA_URL },
    ]);
  });

  test("openai-compatible-legacy forwards the content parts unchanged", async () => {
    const { captured, fetcher } = captureBody({
      choices: [{ message: { content: "ok" } }],
    });

    const adapter = createOpenAICompatibleLegacyAdapter({
      baseURL: "https://example.com",
      model: "gpt-test",
      fetcher,
    });
    await adapter.complete({ messages: imageMessages });

    const messages = captured.body.messages as Array<{ content: Array<Record<string, unknown>> }>;
    expect(messages[0]?.content).toEqual([
      { type: "text", text: "Compare." },
      { type: "image_url", image_url: { url: REMOTE_URL } },
      { type: "image_url", image_url: { url: DATA_URL } },
    ]);
  });

  test("anthropic-compatible maps http URLs to url sources and data URLs to base64", async () => {
    const { captured, fetcher } = captureBody({ content: [{ type: "text", text: "ok" }] });

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });
    await adapter.complete({ messages: imageMessages });

    const messages = captured.body.messages as Array<{ content: Array<Record<string, unknown>> }>;
    expect(messages[0]?.content).toEqual([
      { type: "text", text: "Compare." },
      { type: "image", source: { type: "url", url: REMOTE_URL } },
      { type: "image", source: { type: "base64", media_type: "image/png", data: "AAAA" } },
    ]);
  });

  test("anthropic-compatible decodes a data URL whatever the case of its scheme", async () => {
    const { captured, fetcher } = captureBody({ content: [{ type: "text", text: "ok" }] });

    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://example.com",
      model: "claude-test",
      fetcher,
    });
    await adapter.complete({
      messages: [
        { role: "user", content: images("DATA:image/png;BASE64,AAAA") },
      ],
    });

    const messages = captured.body.messages as Array<{ content: Array<Record<string, unknown>> }>;
    expect(messages[0]?.content).toEqual([
      { type: "image", source: { type: "base64", media_type: "image/png", data: "AAAA" } },
    ]);
  });
});
