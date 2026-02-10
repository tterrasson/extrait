import { jsonrepair } from "jsonrepair";
import { extractMarkdownCodeBlocks } from "./markdown";
import { toErrorMessage } from "./utils/common";
import type {
  ExtractJsonCandidatesOptions,
  ExtractionCandidate,
  ExtractionHeuristicsOptions,
  ExtractionParseHint,
} from "./types";

interface StackItem {
  char: "{" | "[";
  index: number;
}

interface RankedCandidate extends ExtractionCandidate {
  shapeScore: number;
}

export const DEFAULT_EXTRACTION_HEURISTICS: ExtractionHeuristicsOptions = {
  firstPassMin: 12,
  firstPassCap: 30,
  firstPassMultiplier: 6,
  secondPassMin: 4,
  secondPassCap: 8,
  secondPassMultiplier: 2,
  hintMaxLength: 50_000,
};

export function extractJsonCandidates(
  input: string,
  options: ExtractJsonCandidatesOptions = {},
): ExtractionCandidate[] {
  const maxCandidates = options.maxCandidates ?? 5;
  const acceptArrays = options.acceptArrays ?? true;
  const allowRepairHints = options.allowRepairHints ?? true;
  const heuristics = resolveExtractionHeuristics(options.heuristics);
  const extractionInput = input;
  const candidates: ExtractionCandidate[] = [];

  candidates.push(...extractFromMarkdown(extractionInput, acceptArrays));
  candidates.push(...scanBalancedSegments(extractionInput, acceptArrays));

  if (candidates.length === 0 && extractionInput.trim()) {
    const content = extractionInput.trim();
    candidates.push({
      id: "raw:fallback",
      source: "raw",
      content,
      start: 0,
      end: extractionInput.length,
      score: 10 + Math.floor(lengthScore(content.length) / 3),
    });
  }

  const prefiltered = prefilterByJsonShape(candidates, acceptArrays);
  const firstPassLimit = clamp(
    maxCandidates * heuristics.firstPassMultiplier,
    heuristics.firstPassMin,
    heuristics.firstPassCap,
  );
  const firstPass = prefiltered.slice(0, firstPassLimit);

  const secondPassLimit = Math.min(
    firstPass.length,
    clamp(
      maxCandidates * heuristics.secondPassMultiplier,
      heuristics.secondPassMin,
      heuristics.secondPassCap,
    ),
  );

  for (let i = 0; i < secondPassLimit; i += 1) {
    const candidate = firstPass[i];
    if (!candidate) {
      continue;
    }

    const parseHint = buildParseHint(
      candidate.content,
      allowRepairHints,
      heuristics.hintMaxLength,
    );
    if (!parseHint) {
      continue;
    }

    candidate.parseHint = parseHint;
    candidate.score += parseHintBonus(parseHint);
  }

  const sorted = sortCandidates(firstPass);
  const deduped = dedupeCandidates(sorted);

  return deduped.slice(0, maxCandidates).map((candidate, index) => ({
    id: `${candidate.source}:${index}`,
    source: candidate.source,
    content: candidate.content,
    language: candidate.language,
    parseHint: candidate.parseHint,
    start: candidate.start,
    end: candidate.end,
    score: candidate.score,
  }));
}

function prefilterByJsonShape(
  candidates: ExtractionCandidate[],
  acceptArrays: boolean,
): RankedCandidate[] {
  const shaped = candidates.map((candidate) => {
    const shapeScore = jsonShapeScore(candidate.content, acceptArrays);
    return {
      ...candidate,
      shapeScore,
      score: candidate.score + shapeScore,
    } satisfies RankedCandidate;
  });

  const sorted = sortCandidates(shaped);
  const deduped = dedupeCandidates(sorted);
  const filtered = deduped.filter((candidate) => passesShapeFilter(candidate));
  if (filtered.length > 0) {
    const fallback = deduped.find((candidate) => !passesShapeFilter(candidate));
    if (fallback) {
      filtered.push(fallback);
    }
    return sortCandidates(filtered);
  }

  return deduped.slice(0, Math.min(1, deduped.length));
}

function extractFromMarkdown(
  input: string,
  acceptArrays: boolean,
): ExtractionCandidate[] {
  const blocks = extractMarkdownCodeBlocks(input);
  return blocks.flatMap((block, index) => {
    const language = block.language || null;
    const content = block.code.trim();
    if (!content) {
      return [];
    }

    if (!looksLikeJsonEnvelope(content, acceptArrays)) {
      return [];
    }

    const langBonus = languageBonus(language);

    return [
      {
        id: `fenced:${index}`,
        source: "fenced",
        language,
        content,
        start: block.start,
        end: block.end,
        score: 260 + langBonus + lengthScore(content.length),
      } satisfies ExtractionCandidate,
    ];
  });
}

function scanBalancedSegments(
  input: string,
  acceptArrays: boolean,
): ExtractionCandidate[] {
  const results: ExtractionCandidate[] = [];
  const stack: StackItem[] = [];

  let inString = false;
  let quote: string | null = null;
  let escaped = false;

  for (let i = 0; i < input.length; i += 1) {
    const char = input[i];
    if (!char) {
      continue;
    }

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

    const allowSingleQuoted = stack.length > 0;
    if (char === '"' || (allowSingleQuoted && (char === "'" || char === "`"))) {
      inString = true;
      quote = char;
      continue;
    }

    if (char === "{" || char === "[") {
      stack.push({ char, index: i });
      continue;
    }

    if (char !== "}" && char !== "]") {
      continue;
    }

    const expectedOpen = char === "}" ? "{" : "[";
    while (stack.length > 0 && stack[stack.length - 1]?.char !== expectedOpen) {
      stack.pop();
    }

    const opened = stack.pop();
    if (!opened) {
      continue;
    }

    if (stack.length > 0) {
      continue;
    }

    if (!acceptArrays && opened.char === "[") {
      continue;
    }

    const content = input.slice(opened.index, i + 1).trim();
    if (!content) {
      continue;
    }

    const rootBonus = opened.char === "{" ? 40 : 20;
    const boundaryBonus = boundaryScore(input, opened.index, i + 1);

    results.push({
      id: `scan:${results.length}`,
      source: "scan",
      content,
      start: opened.index,
      end: i + 1,
      score: 120 + rootBonus + boundaryBonus + lengthScore(content.length),
    });
  }

  return results;
}

function looksLikeJsonEnvelope(content: string, acceptArrays: boolean): boolean {
  const trimmed = content.trim();
  if (trimmed.startsWith("{")) {
    return true;
  }
  if (acceptArrays && trimmed.startsWith("[")) {
    return true;
  }
  return false;
}

function languageBonus(language: string | null): number {
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

function jsonShapeScore(
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
    if (/^\{\s*\}$/.test(trimmed)) {
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
    score += /^\[\s*\]$/.test(trimmed) ? 8 : 4;
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

function boundaryScore(input: string, start: number, end: number): number {
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

  return /[\s,.;:!?`"'()[\]{}<>]/.test(char);
}

function lengthScore(length: number): number {
  return Math.min(120, Math.floor(Math.sqrt(length) * 6));
}

function passesShapeFilter(candidate: RankedCandidate): boolean {
  if (candidate.source === "raw") {
    return true;
  }

  if (candidate.source === "fenced") {
    return candidate.shapeScore >= 0;
  }

  return candidate.shapeScore >= 35;
}

function buildParseHint(
  content: string,
  allowRepair: boolean,
  hintMaxLength: number,
): ExtractionParseHint | null {
  if (content.length > hintMaxLength) {
    return null;
  }

  try {
    return {
      success: true,
      parsed: JSON.parse(content),
      repaired: null,
      usedRepair: false,
      stage: "parse",
      error: "",
    };
  } catch (directError) {
    if (!allowRepair) {
      return {
        success: false,
        parsed: null,
        repaired: null,
        usedRepair: false,
        stage: "parse",
        error: toErrorMessage(directError),
      };
    }

    let repaired: string;
    try {
      repaired = jsonrepair(content);
    } catch (repairError) {
      return {
        success: false,
        parsed: null,
        repaired: null,
        usedRepair: true,
        stage: "repair",
        error: toErrorMessage(repairError),
      };
    }

    try {
      return {
        success: true,
        parsed: JSON.parse(repaired),
        repaired,
        usedRepair: true,
        stage: "parse",
        error: "",
      };
    } catch (parseError) {
      return {
        success: false,
        parsed: null,
        repaired,
        usedRepair: true,
        stage: "parse",
        error: toErrorMessage(parseError || directError),
      };
    }
  }
}

function resolveExtractionHeuristics(
  input: Partial<ExtractionHeuristicsOptions> | undefined,
): ExtractionHeuristicsOptions {
  const merged = {
    ...DEFAULT_EXTRACTION_HEURISTICS,
    ...input,
  };

  const firstPassMin = normalizeInteger(merged.firstPassMin, DEFAULT_EXTRACTION_HEURISTICS.firstPassMin);
  const firstPassCap = Math.max(
    firstPassMin,
    normalizeInteger(merged.firstPassCap, DEFAULT_EXTRACTION_HEURISTICS.firstPassCap),
  );
  const secondPassMin = normalizeInteger(merged.secondPassMin, DEFAULT_EXTRACTION_HEURISTICS.secondPassMin);
  const secondPassCap = Math.max(
    secondPassMin,
    normalizeInteger(merged.secondPassCap, DEFAULT_EXTRACTION_HEURISTICS.secondPassCap),
  );

  return {
    firstPassMin,
    firstPassCap,
    firstPassMultiplier: normalizeInteger(
      merged.firstPassMultiplier,
      DEFAULT_EXTRACTION_HEURISTICS.firstPassMultiplier,
    ),
    secondPassMin,
    secondPassCap,
    secondPassMultiplier: normalizeInteger(
      merged.secondPassMultiplier,
      DEFAULT_EXTRACTION_HEURISTICS.secondPassMultiplier,
    ),
    hintMaxLength: normalizeInteger(
      merged.hintMaxLength,
      DEFAULT_EXTRACTION_HEURISTICS.hintMaxLength,
    ),
  };
}

function normalizeInteger(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }

  return Math.max(0, Math.floor(value));
}

function parseHintBonus(hint: ExtractionParseHint): number {
  if (hint.success) {
    return hint.usedRepair ? 70 : 120;
  }

  return hint.usedRepair ? -20 : -10;
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

function sortCandidates<T extends { score: number; start: number; end: number }>(
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

function dedupeCandidates<T extends { content: string }>(candidates: T[]): T[] {
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

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.floor(value)));
}
