import { describe, expect, test } from "bun:test";
import { normalizeBaseURL } from "../src/providers/utils";

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
