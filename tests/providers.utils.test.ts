import { describe, expect, test } from "bun:test";
import { buildURL, normalizeBaseURL, preferLatestUsage } from "@/providers/utils";

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
});

describe("providers/utils preferLatestUsage", () => {
  test("prefers newer defined fields without summing", () => {
    expect(
      preferLatestUsage(
        { inputTokens: 10, outputTokens: 1, totalTokens: 11 },
        { inputTokens: 10, outputTokens: 2, totalTokens: 12 },
      ),
    ).toEqual({ inputTokens: 10, outputTokens: 2, totalTokens: 12 });
  });

  test("preserves older fields when the newer snapshot is partial", () => {
    expect(
      preferLatestUsage(
        { inputTokens: 10, outputTokens: 1, totalTokens: 11 },
        { outputTokens: 2 },
      ),
    ).toEqual({ inputTokens: 10, outputTokens: 2, totalTokens: 11 });
  });
});
