import type { LLMUsage, ReasoningBlock } from "../types";

const RE_THINK_TAG = /<\/?think\s*>/gi;

export function pushReasoningBlock(blocks: ReasoningBlock[], turnIndex: number, text: string | undefined): void {
  const clean = text?.replace(RE_THINK_TAG, "").trim();
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
  try {
    return new URL(path).toString();
  } catch {
    // Treat provider paths as relative to the configured base URL, even when they start with "/".
  }

  const base = new URL(normalizeBaseURL(baseURL));
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
