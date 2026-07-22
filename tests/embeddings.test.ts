import { describe, expect, test } from "bun:test";
import { createOpenAICompatibleAdapter } from "@/providers/openai-compatible";
import { createAnthropicCompatibleAdapter } from "@/providers/anthropic-compatible";
import { createLLM } from "@/llm";

function embeddingResponse(embeddings: number[][], model = "text-embedding-3-small"): Response {
  return new Response(
    JSON.stringify({
      object: "list",
      data: embeddings.map((embedding, index) => ({ object: "embedding", embedding, index })),
      model,
      usage: { prompt_tokens: 4, total_tokens: 4 },
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

function errorResponse(status: number, message: string): Response {
  return new Response(message, { status });
}

const VECTOR_A = [0.1, 0.2, 0.3];
const VECTOR_B = [0.4, 0.5, 0.6];

describe("openai-compatible embed()", () => {
  test("embeds a single string and returns number[][]", async () => {
    const fetcher = (async () => embeddingResponse([VECTOR_A])) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://api.openai.com",
      model: "text-embedding-3-small",
      fetcher,
    });

    const result = await adapter.embed!({ input: "Hello world" });

    expect(result.embeddings).toHaveLength(1);
    expect(result.embeddings[0]).toEqual(VECTOR_A);
    expect(result.model).toBe("text-embedding-3-small");
    expect(result.usage?.inputTokens).toBe(4);
    expect(result.usage?.totalTokens).toBe(4);
  });

  test("embeds an array of strings and returns one vector per input", async () => {
    const fetcher = (async () => embeddingResponse([VECTOR_A, VECTOR_B])) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://api.openai.com",
      model: "text-embedding-3-small",
      fetcher,
    });

    const result = await adapter.embed!({ input: ["text one", "text two"] });

    expect(result.embeddings).toHaveLength(2);
    expect(result.embeddings[0]).toEqual(VECTOR_A);
    expect(result.embeddings[1]).toEqual(VECTOR_B);
  });

  test("overrides the model via request options", async () => {
    let capturedBody: Record<string, unknown> | undefined;
    const fetcher = (async (_url: string, init: RequestInit) => {
      capturedBody = JSON.parse(init.body as string);
      return embeddingResponse([VECTOR_A], "text-embedding-ada-002");
    }) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://api.openai.com",
      model: "gpt-4o",
      fetcher,
    });

    const result = await adapter.embed!({ input: "hello", model: "text-embedding-ada-002" });

    expect(capturedBody?.model).toBe("text-embedding-ada-002");
    expect(result.model).toBe("text-embedding-ada-002");
  });

  test("sends dimensions when specified", async () => {
    let capturedBody: Record<string, unknown> | undefined;
    const fetcher = (async (_url: string, init: RequestInit) => {
      capturedBody = JSON.parse(init.body as string);
      return embeddingResponse([VECTOR_A]);
    }) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://api.openai.com",
      model: "text-embedding-3-small",
      fetcher,
    });

    await adapter.embed!({ input: "hello", dimensions: 256 });

    expect(capturedBody?.dimensions).toBe(256);
  });

  test("sends to custom embeddingPath when configured", async () => {
    let capturedURL: string | undefined;
    const fetcher = (async (url: string) => {
      capturedURL = url;
      return embeddingResponse([VECTOR_A]);
    }) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://custom.example.com",
      model: "my-embed-model",
      embeddingPath: "/v2/embeddings",
      fetcher,
    });

    await adapter.embed!({ input: "hello" });

    expect(capturedURL).toContain("/v2/embeddings");
  });

  test("throws on non-ok response", async () => {
    const fetcher = (async () => errorResponse(401, "Unauthorized")) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://api.openai.com",
      model: "text-embedding-3-small",
      fetcher,
    });

    await expect(adapter.embed!({ input: "hello" })).rejects.toThrow("HTTP 401");
  });

  test("sends body passthrough fields", async () => {
    let capturedBody: Record<string, unknown> | undefined;
    const fetcher = (async (_url: string, init: RequestInit) => {
      capturedBody = JSON.parse(init.body as string);
      return embeddingResponse([VECTOR_A]);
    }) as unknown as typeof fetch;

    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://api.openai.com",
      model: "text-embedding-3-small",
      fetcher,
    });

    await adapter.embed!({ input: "hello", body: { user: "test-user" } });

    expect(capturedBody?.user).toBe("test-user");
  });
});

describe("LLMClient.embed()", () => {
  test("delegates to adapter.embed with correct input", async () => {
    const fetcher = (async () => embeddingResponse([VECTOR_A])) as unknown as typeof fetch;
    const llm = createLLM({
      provider: "openai-compatible",
      baseURL: "https://api.openai.com",
      model: "text-embedding-3-small",
      transport: { fetcher },
    });

    const result = await llm.embed("Hello world");

    expect(result.embeddings).toHaveLength(1);
    expect(result.embeddings[0]).toEqual(VECTOR_A);
  });

  test("delegates array input to adapter.embed", async () => {
    const fetcher = (async () => embeddingResponse([VECTOR_A, VECTOR_B])) as unknown as typeof fetch;
    const llm = createLLM({
      provider: "openai-compatible",
      baseURL: "https://api.openai.com",
      model: "text-embedding-3-small",
      transport: { fetcher },
    });

    const result = await llm.embed(["a", "b"]);

    expect(result.embeddings).toHaveLength(2);
  });

  test("passes options through to adapter.embed", async () => {
    let capturedBody: Record<string, unknown> | undefined;
    const fetcher = (async (_url: string, init: RequestInit) => {
      capturedBody = JSON.parse(init.body as string);
      return embeddingResponse([VECTOR_A]);
    }) as unknown as typeof fetch;

    const llm = createLLM({
      provider: "openai-compatible",
      baseURL: "https://api.openai.com",
      model: "gpt-4o",
      transport: { fetcher },
    });

    await llm.embed("hello", { model: "text-embedding-3-small", dimensions: 512 });

    expect(capturedBody?.model).toBe("text-embedding-3-small");
    expect(capturedBody?.dimensions).toBe(512);
  });
});

describe("anthropic-compatible embed()", () => {
  test("throws a descriptive error mentioning Voyage AI", async () => {
    const adapter = createAnthropicCompatibleAdapter({
      baseURL: "https://api.anthropic.com",
      model: "claude-3-5-sonnet-20241022",
      apiKey: "test",
    });

    await expect(adapter.embed!({ input: "hello" })).rejects.toThrow("Voyage AI");
    await expect(adapter.embed!({ input: "hello" })).rejects.toThrow("Anthropic");
  });
});

describe("openai-compatible embed() encoding_format", () => {
  test("defaults encoding_format to float", async () => {
    let body: Record<string, unknown> = {};
    const fetcher = (async (_input: Request | string | URL, init?: RequestInit) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return embeddingResponse([VECTOR_A]);
    }) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://api.openai.com",
      model: "text-embedding-3-small",
      fetcher,
    });

    await adapter.embed!({ input: "hello" });

    expect(body.encoding_format).toBe("float");
  });

  test("lets defaultBody and request.body override encoding_format (Voyage AI)", async () => {
    const bodies: Record<string, unknown>[] = [];
    const fetcher = (async (_input: Request | string | URL, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return embeddingResponse([VECTOR_A], "voyage-3");
    }) as unknown as typeof fetch;
    const adapter = createOpenAICompatibleAdapter({
      baseURL: "https://api.voyageai.com",
      model: "voyage-3",
      defaultBody: { encoding_format: null },
      fetcher,
    });

    await adapter.embed!({ input: "hello" });
    await adapter.embed!({ input: "hello", body: { encoding_format: "base64" } });

    expect(bodies[0]?.encoding_format).toBeNull();
    expect(bodies[1]?.encoding_format).toBe("base64");
  });
});
