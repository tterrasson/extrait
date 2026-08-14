import type { LLMUsage, ReasoningBlock } from "../types";
import { stripThinkTags } from "../think";

export function pushReasoningBlock(blocks: ReasoningBlock[], turnIndex: number, text: string | undefined): void {
  const clean = text ? stripThinkTags(text).trim() : undefined;
  if (!clean) {
    return;
  }

  blocks.push({ turnIndex, text: clean });
}

export function joinReasoningBlocks(blocks: ReasoningBlock[]): string {
  return blocks.map((block) => block.text).filter(Boolean).join("\n\n");
}

export function normalizeBaseURL(baseURL: string): string {
  return baseURL.endsWith("/") ? baseURL : `${baseURL}/`;
}

export function buildURL(baseURL: string, path: string): string {
  const base = new URL(normalizeBaseURL(baseURL));

  let absolutePath: URL | undefined;
  try {
    absolutePath = new URL(path);
  } catch {
    // Treat provider paths as relative to the configured base URL, even when they start with "/".
  }

  if (absolutePath) {
    // An absolute `path` would otherwise silently override `baseURL`, and every
    // request carries the API key — so a stray or attacker-influenced path could
    // ship credentials to another host. Only same-origin absolute paths pass.
    if (absolutePath.origin !== base.origin) {
      throw new Error(
        `Provider path "${path}" points to a different origin than baseURL "${baseURL}". ` +
          "Use a relative path so credentials are never sent to an unintended host.",
      );
    }

    return absolutePath.toString();
  }

  const resolvedPath = new URL(path, "http://provider-path.local");

  base.pathname = mergePathnames(base.pathname, resolvedPath.pathname);
  base.search = resolvedPath.search;
  base.hash = resolvedPath.hash;

  return base.toString();
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
    // Request bodies can be merged from LLM- or config-authored JSON: assigning a
    // `__proto__` key would hit the prototype setter instead of adding a field.
    if (value !== undefined && key !== "__proto__") {
      (out as Record<string, unknown>)[key] = value;
    }
  }
  return out;
}

const DEFAULT_MAX_ERROR_BODY_BYTES = 16_000;

/**
 * Read an error response body without letting a hostile or broken provider
 * stream an unbounded amount of memory into an exception message.
 */
export async function readErrorBody(
  response: Response,
  maxBytes = DEFAULT_MAX_ERROR_BODY_BYTES,
): Promise<string> {
  if (!response.body) {
    return truncateBody(await response.text(), maxBytes);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let out = "";

  try {
    while (out.length < maxBytes) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      out += decoder.decode(value, { stream: true });
    }
  } catch {
    // A truncated/aborted error body is still worth reporting.
  } finally {
    await reader.cancel().catch(() => {});
  }

  return truncateBody(out, maxBytes);
}

function truncateBody(body: string, maxBytes: number): string {
  const normalized = body.trim();
  if (normalized.length <= maxBytes) {
    return normalized;
  }

  return `${normalized.slice(0, maxBytes)}...[truncated]`;
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

export function preferLatestUsage(base: LLMUsage | undefined, next: LLMUsage | undefined): LLMUsage | undefined {
  if (!base && !next) {
    return undefined;
  }

  const merged: LLMUsage = {};

  if (base?.inputTokens !== undefined || next?.inputTokens !== undefined) {
    merged.inputTokens = next?.inputTokens ?? base?.inputTokens;
  }
  if (base?.outputTokens !== undefined || next?.outputTokens !== undefined) {
    merged.outputTokens = next?.outputTokens ?? base?.outputTokens;
  }
  if (base?.totalTokens !== undefined || next?.totalTokens !== undefined) {
    merged.totalTokens = next?.totalTokens ?? base?.totalTokens;
  }
  if (base?.cost !== undefined || next?.cost !== undefined) {
    merged.cost = next?.cost ?? base?.cost;
  }

  return Object.keys(merged).length > 0 ? merged : undefined;
}

function addOptional(a: number | undefined, b: number | undefined): number | undefined {
  if (a === undefined && b === undefined) {
    return undefined;
  }

  return (a ?? 0) + (b ?? 0);
}

function mergePathnames(basePathname: string, pathPathname: string): string {
  const baseSegments = splitPathSegments(basePathname);
  const pathSegments = splitPathSegments(pathPathname);
  const overlap = findPathOverlap(baseSegments, pathSegments);
  const mergedSegments = [...baseSegments, ...pathSegments.slice(overlap)];

  if (mergedSegments.length === 0) {
    return "/";
  }

  const mergedPathname = `/${mergedSegments.join("/")}`;
  return pathPathname.endsWith("/") && pathPathname !== "/" ? `${mergedPathname}/` : mergedPathname;
}

function splitPathSegments(pathname: string): string[] {
  return pathname.split("/").filter((segment) => segment.length > 0);
}

function findPathOverlap(baseSegments: string[], pathSegments: string[]): number {
  const maxOverlap = Math.min(baseSegments.length, pathSegments.length);

  for (let size = maxOverlap; size > 0; size -= 1) {
    let matches = true;

    for (let index = 0; index < size; index += 1) {
      if (baseSegments[baseSegments.length - size + index] !== pathSegments[index]) {
        matches = false;
        break;
      }
    }

    if (matches) {
      return size;
    }
  }

  return 0;
}
