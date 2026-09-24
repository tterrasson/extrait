import { extractMarkdownCodeBlocks } from "./markdown";
import { buildParseHint, parseHintBonus } from "./extract-parse-hint";
import {
  boundaryScore,
  clamp,
  dedupeCandidates,
  jsonShapeScore,
  languageBonus,
  lengthScore,
  looksLikeJsonEnvelope,
  passesShapeFilter,
  resolveExtractionHeuristics,
  sortCandidates,
  type RankedCandidate,
} from "./extract-shape";
import type {
  ExtractJsonCandidatesOptions,
  ExtractionCandidate,
  ExtractionHeuristicsOptions,
} from "./types";

interface StackItem {
  char: "{" | "[";
  index: number;
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
  const heuristics = resolveExtractionHeuristics(options.heuristics, DEFAULT_EXTRACTION_HEURISTICS);
  const extractionInput = input;
  const candidates: ExtractionCandidate[] = [];

  candidates.push(...extractFromMarkdown(extractionInput, acceptArrays));
  candidates.push(...scanAllBalancedSegments(extractionInput, acceptArrays));

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

/**
 * Quotes in the prose around the payload are read as JSON strings, which keeps
 * a quoted decoy such as `"see {x}"` from being taken for the payload. An
 * unbalanced one, though (`a 12" pizza`), would swallow everything after it.
 * Ending inside such a string is the tell: the text is scanned again with
 * quotes outside any bracket ignored, and the segments only that pass finds are
 * added. Ranking and validation pick the payload among them.
 */
function scanAllBalancedSegments(input: string, acceptArrays: boolean): ExtractionCandidate[] {
  const quoteAware = scanBalancedSegments(input, acceptArrays, true);
  if (!quoteAware.unterminatedString) {
    return quoteAware.results;
  }

  const seen = new Set(quoteAware.results.map((candidate) => `${candidate.start}:${candidate.end}`));
  const results = [...quoteAware.results];
  for (const candidate of scanBalancedSegments(input, acceptArrays, false).results) {
    if (!seen.has(`${candidate.start}:${candidate.end}`)) {
      results.push({ ...candidate, id: `scan:${results.length}` });
    }
  }
  return results;
}

function scanBalancedSegments(
  input: string,
  acceptArrays: boolean,
  topLevelQuotes: boolean,
): { results: ExtractionCandidate[]; unterminatedString: boolean } {
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

    const insideContainer = stack.length > 0;
    if (
      (char === '"' && (insideContainer || topLevelQuotes)) ||
      (insideContainer && (char === "'" || char === "`"))
    ) {
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

  return { results, unterminatedString: inString && stack.length === 0 };
}
