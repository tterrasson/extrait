import { describe, expect, test } from "bun:test";
import {
  buildURL,
  cleanUndefined,
  mergeUsage,
  normalizeBaseURL,
  preferLatestUsage,
  readErrorBody,
} from "@/providers/utils";

describe("providers/utils normalizeBaseURL", () => {
  test("adds a trailing slash if absent", () => {
    expect(normalizeBaseURL("https://api.example.com")).toBe("https://api.example.com/");
  });

  test("preserves the trailing slash if present", () => {
    expect(normalizeBaseURL("https://api.example.com/")).toBe("https://api.example.com/");
  });

  test("handles URLs with path", () => {
    expect(normalizeBaseURL("https://api.example.com/v1")).toBe("https://api.example.com/v1/");
  });

  test("handles localhost", () => {
    expect(normalizeBaseURL("http://localhost:3000")).toBe("http://localhost:3000/");
  });

  test("handles IPs", () => {
    expect(normalizeBaseURL("http://127.0.0.1:11434")).toBe("http://127.0.0.1:11434/");
  });

  test("handles very short URLs", () => {
    expect(normalizeBaseURL("http://a")).toBe("http://a/");
  });

  test("handles URLs with multiple slashes", () => {
    expect(normalizeBaseURL("https://api.example.com/")).toBe("https://api.example.com/");
  });
});

describe("providers/utils buildURL", () => {
  test("preserves a base path for openai-compatible style endpoints", () => {
    expect(buildURL("https://example.com/api/", "/v1/chat/completions")).toBe(
      "https://example.com/api/v1/chat/completions",
    );
  });

  test("deduplicates overlapping version segments", () => {
    expect(buildURL("https://openrouter.ai/api/v1/", "/v1/chat/completions")).toBe(
      "https://openrouter.ai/api/v1/chat/completions",
    );
  });

  test("keeps root-based providers unchanged", () => {
    expect(buildURL("https://example.com", "/v1/chat/completions")).toBe(
      "https://example.com/v1/chat/completions",
    );
  });

  test("preserves a base path for anthropic-compatible style endpoints", () => {
    expect(buildURL("https://example.com/api/", "/v1/messages")).toBe(
      "https://example.com/api/v1/messages",
    );
  });

  test("rejects an absolute path pointing at another host", () => {
    expect(() => buildURL("https://example.com/api/", "https://attacker.example/v1/responses"))
      .toThrow(/different origin/);
  });

  test("rejects an absolute path on a non-http scheme", () => {
    expect(() => buildURL("https://example.com/api/", "file:///etc/passwd")).toThrow(
      /different origin/,
    );
  });

  test("accepts an absolute path on the same origin", () => {
    expect(buildURL("https://example.com/api/", "https://example.com/v1/responses")).toBe(
      "https://example.com/v1/responses",
    );
  });
});

describe("providers/utils cleanUndefined", () => {
  test("drops undefined values", () => {
    expect(cleanUndefined({ a: 1, b: undefined } as Record<string, unknown>)).toEqual({ a: 1 });
  });

  test("drops __proto__ instead of hitting the prototype setter", () => {
    const polluted = JSON.parse('{"model":"m","__proto__":{"polluted":true}}') as Record<
      string,
      unknown
    >;
    const cleaned = cleanUndefined(polluted);

    expect(cleaned).toEqual({ model: "m" });
    expect(Object.getPrototypeOf(cleaned)).toBe(Object.prototype);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });
});

describe("providers/utils readErrorBody", () => {
  test("returns the trimmed body when it fits", async () => {
    expect(await readErrorBody(new Response("  boom  "))).toBe("boom");
  });

  test("truncates an oversized body", async () => {
    const body = await readErrorBody(new Response("x".repeat(500)), 100);
    expect(body).toBe(`${"x".repeat(100)}...[truncated]`);
  });

  test("stops reading an endless stream", async () => {
    const encoder = new TextEncoder();
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.enqueue(encoder.encode("x".repeat(1_000)));
      },
      cancel() {
        cancelled = true;
      },
    });

    const body = await readErrorBody(new Response(stream), 5_000);

    expect(body.length).toBeLessThanOrEqual(5_000 + "...[truncated]".length);
    expect(cancelled).toBe(true);
  });
});

describe("providers/utils preferLatestUsage", () => {
  test("prefers newer defined fields without summing", () => {
    expect(
      preferLatestUsage(
        { inputTokens: 10, outputTokens: 1, totalTokens: 11 },
        { inputTokens: 10, outputTokens: 2, totalTokens: 12 },
      ),
    ).toEqual({ inputTokens: 10, contextTokens: 10, outputTokens: 2, totalTokens: 12 });
  });

  test("preserves older fields when the newer snapshot is partial", () => {
    expect(
      preferLatestUsage(
        { inputTokens: 10, outputTokens: 1, totalTokens: 11 },
        { outputTokens: 2 },
      ),
    ).toEqual({ inputTokens: 10, contextTokens: 10, outputTokens: 2, totalTokens: 11 });
  });

  test("never lets a partial chunk shrink the context reading", () => {
    expect(
      preferLatestUsage({ inputTokens: 4000 }, { inputTokens: 0, outputTokens: 12 })?.contextTokens,
    ).toBe(4000);
  });
});

describe("providers/utils usage context tokens", () => {
  test("keeps the largest round prompt while input tokens sum", () => {
    const round1 = { inputTokens: 1000, outputTokens: 50 };
    const round2 = { inputTokens: 1800, outputTokens: 70 };
    const round3 = { inputTokens: 2600, outputTokens: 90 };

    const merged = mergeUsage(mergeUsage(round1, round2), round3);

    expect(merged).toEqual({
      inputTokens: 5400,
      contextTokens: 2600,
      outputTokens: 210,
    });
  });

  test("carries an already-merged context through further merges", () => {
    expect(
      mergeUsage({ inputTokens: 5400, contextTokens: 2600 }, { inputTokens: 900 })?.contextTokens,
    ).toBe(2600);
  });

  test("is left undefined when no side reports tokens", () => {
    expect(mergeUsage({ cost: 0.5 }, { cost: 0.25 })).toEqual({ cost: 0.75 });
  });
});
