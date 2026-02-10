import type { LLMUsage } from "../types";

export function normalizeBaseURL(baseURL: string): string {
  return baseURL.endsWith("/") ? baseURL : `${baseURL}/`;
}

export function buildURL(baseURL: string, path: string): string {
  return new URL(path, normalizeBaseURL(baseURL)).toString();
}

export function safeJSONParse(input: string): unknown {
  try {
    return JSON.parse(input);
  } catch {
    return null;
  }
}

export function cleanUndefined<T extends Record<string, unknown>>(input: T): T {
  const out = {} as T;
  for (const [key, value] of Object.entries(input)) {
    if (value !== undefined) {
      (out as Record<string, unknown>)[key] = value;
    }
  }
  return out;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function pickString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

export function toFiniteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function mergeUsage(base: LLMUsage | undefined, next: LLMUsage | undefined): LLMUsage | undefined {
  if (!base && !next) {
    return undefined;
  }

  const inputTokens = addOptional(base?.inputTokens, next?.inputTokens);
  const outputTokens = addOptional(base?.outputTokens, next?.outputTokens);
  const totalTokens = addOptional(base?.totalTokens, next?.totalTokens);
  const cost = addOptional(base?.cost, next?.cost);

  const merged: LLMUsage = {};
  if (inputTokens !== undefined) {
    merged.inputTokens = inputTokens;
  }
  if (outputTokens !== undefined) {
    merged.outputTokens = outputTokens;
  }
  if (totalTokens !== undefined) {
    merged.totalTokens = totalTokens;
  }
  if (cost !== undefined) {
    merged.cost = cost;
  }

  return Object.keys(merged).length > 0 ? merged : undefined;
}

function addOptional(a: number | undefined, b: number | undefined): number | undefined {
  if (a === undefined && b === undefined) {
    return undefined;
  }

  return (a ?? 0) + (b ?? 0);
}
