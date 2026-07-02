import type { ExtractionCandidate, ExtractionHeuristicsOptions } from "./types";

const RE_EMPTY_OBJECT = /^\{\s*\}$/;
const RE_EMPTY_ARRAY = /^\[\s*\]$/;
const RE_BOUNDARY_CHAR = /[\s,.;:!?`"'()\[\]{}<>]/;

export interface RankedCandidate extends ExtractionCandidate {
  shapeScore: number;
}

export function looksLikeJsonEnvelope(content: string, acceptArrays: boolean): boolean {
  const trimmed = content.trim();
  if (trimmed.startsWith("{")) {
    return true;
  }
  if (acceptArrays && trimmed.startsWith("[")) {
    return true;
  }
  return false;
}

export function languageBonus(language: string | null): number {
  if (!language) {
    return 40;
  }

  if (language === "json") {
    return 140;
  }

  if (language === "jsonc" || language === "json5") {
    return 100;
  }

  if (
    language === "javascript" ||
    language === "typescript" ||
    language === "js" ||
    language === "ts"
  ) {
    return 40;
  }

  return 0;
}

export function jsonShapeScore(
  content: string,
  acceptArrays: boolean,
): number {
  const trimmed = content.trim();
  if (!trimmed) {
    return -100;
  }

  const root = trimmed[0];
  const end = trimmed[trimmed.length - 1];
  if (root !== "{" && root !== "[") {
    return -100;
  }

  if (root === "[" && !acceptArrays) {
    return -100;
  }

  const expectedEnd = root === "{" ? "}" : "]";
  let score = 20;
  if (end === expectedEnd) {
    score += 20;
  } else {
    score -= 30;
  }

  score += hasBalancedJsonDelimiters(trimmed) ? 25 : -25;

  const colonCount = countChar(trimmed, ":");
  const commaCount = countChar(trimmed, ",");
  const quoteCount = countChar(trimmed, "\"");

  if (root === "{") {
    if (RE_EMPTY_OBJECT.test(trimmed)) {
      score += 12;
    } else if (colonCount > 0) {
      score += 22;
    } else {
      score -= 30;
    }

    if (quoteCount > 0) {
      score += quoteCount % 2 === 0 ? 8 : -8;
    }
  } else {
    score += RE_EMPTY_ARRAY.test(trimmed) ? 8 : 4;
    if (colonCount > 0) {
      score += 4;
    }
  }

  if (commaCount > 0) {
    score += 4;
  }

  const structural = countStructuralChars(trimmed);
  const ratio = structural / Math.max(trimmed.length, 1);
  if (ratio >= 0.04 && ratio <= 0.9) {
    score += 8;
  } else {
    score -= 5;
  }

  return score;
}

export function boundaryScore(input: string, start: number, end: number): number {
  const before = start > 0 ? input[start - 1] : "";
  const after = end < input.length ? input[end] : "";
  let score = 0;

  if (isBoundary(before)) {
    score += 10;
  }

  if (isBoundary(after)) {
    score += 10;
  }

  return score;
}

function isBoundary(char: string | undefined): boolean {
  if (!char) {
    return true;
  }

  return RE_BOUNDARY_CHAR.test(char);
}

export function lengthScore(length: number): number {
  return Math.min(120, Math.floor(Math.sqrt(length) * 6));
}

export function passesShapeFilter(candidate: RankedCandidate): boolean {
  if (candidate.source === "raw") {
    return true;
  }

  if (candidate.source === "fenced") {
    return candidate.shapeScore >= 0;
  }

  return candidate.shapeScore >= 35;
}

function hasBalancedJsonDelimiters(input: string): boolean {
  const stack: Array<"{" | "["> = [];

  let inString = false;
  let quote: string | null = null;
  let escaped = false;

  for (let i = 0; i < input.length; i += 1) {
    const char = input[i];

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }

      if (char === "\\") {
        escaped = true;
        continue;
      }

      if (char === quote) {
        inString = false;
        quote = null;
      }

      continue;
    }

    if (char === '"' || char === "'" || char === "`") {
      inString = true;
      quote = char;
      continue;
    }

    if (char === "{" || char === "[") {
      stack.push(char);
      continue;
    }

    if (char === "}" || char === "]") {
      const expected = char === "}" ? "{" : "[";
      const top = stack.pop();
      if (top !== expected) {
        return false;
      }
    }
  }

  return !inString && stack.length === 0;
}

function countChar(input: string, expected: string): number {
  let count = 0;
  for (let i = 0; i < input.length; i += 1) {
    if (input[i] === expected) {
      count += 1;
    }
  }
  return count;
}

function countStructuralChars(input: string): number {
  let count = 0;
  for (let i = 0; i < input.length; i += 1) {
    const char = input[i];
    if (
      char === "{" ||
      char === "}" ||
      char === "[" ||
      char === "]" ||
      char === ":" ||
      char === "," ||
      char === '"'
    ) {
      count += 1;
    }
  }
  return count;
}

export function sortCandidates<T extends { score: number; start: number; end: number }>(
  candidates: T[],
): T[] {
  return [...candidates].sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    if (a.start !== b.start) {
      return a.start - b.start;
    }
    return a.end - b.end;
  });
}

export function dedupeCandidates<T extends { content: string }>(candidates: T[]): T[] {
  const seen = new Set<string>();
  return candidates.filter((candidate) => {
    const key = candidate.content.trim();
    if (!key || seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.floor(value)));
}

export function normalizeInteger(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }

  return Math.max(0, Math.floor(value));
}

export function resolveExtractionHeuristics(
  input: Partial<ExtractionHeuristicsOptions> | undefined,
  defaults: ExtractionHeuristicsOptions,
): ExtractionHeuristicsOptions {
  const merged = {
    ...defaults,
    ...input,
  };

  const firstPassMin = normalizeInteger(merged.firstPassMin, defaults.firstPassMin);
  const firstPassCap = Math.max(
    firstPassMin,
    normalizeInteger(merged.firstPassCap, defaults.firstPassCap),
  );
  const secondPassMin = normalizeInteger(merged.secondPassMin, defaults.secondPassMin);
  const secondPassCap = Math.max(
    secondPassMin,
    normalizeInteger(merged.secondPassCap, defaults.secondPassCap),
  );

  return {
    firstPassMin,
    firstPassCap,
    firstPassMultiplier: normalizeInteger(
      merged.firstPassMultiplier,
      defaults.firstPassMultiplier,
    ),
    secondPassMin,
    secondPassCap,
    secondPassMultiplier: normalizeInteger(
      merged.secondPassMultiplier,
      defaults.secondPassMultiplier,
    ),
    hintMaxLength: normalizeInteger(
      merged.hintMaxLength,
      defaults.hintMaxLength,
    ),
  };
}
