import type { MarkdownCodeBlock, MarkdownCodeOptions } from "./types";
import { isWhitespace } from "./utils/common";

export function extractMarkdownCodeBlocks(
  input: string,
  options: MarkdownCodeOptions = {},
): MarkdownCodeBlock[] {
  const wantedLanguage = normalizeLanguage(options.language);
  const blocks: MarkdownCodeBlock[] = [];
  let cursor = 0;

  while (cursor < input.length) {
    const lineStart = cursor;
    const [lineEnd, nextLineStart] = nextLineBounds(input, lineStart);
    const line = input.slice(lineStart, lineEnd);
    const opening = parseOpeningFence(line);

    if (!opening) {
      cursor = nextLineStart;
      continue;
    }

    const rawLanguage = normalizeLanguage(extractInfoLanguage(opening.info));
    const contentStart = nextLineStart;
    let scanCursor = contentStart;
    let closed = false;

    while (scanCursor <= input.length) {
      const closingLineStart = scanCursor;
      const [closingLineEnd, afterClosingLine] = nextLineBounds(input, closingLineStart);
      const closingLine = input.slice(closingLineStart, closingLineEnd);
      const closing = parseClosingFence(closingLine, opening.marker, opening.length);
      const inlineClosing = parseInlineClosingFence(closingLine, opening.marker, opening.length);

      if (closing !== null) {
        if (!wantedLanguage || rawLanguage === wantedLanguage) {
          blocks.push({
            language: rawLanguage,
            code: input.slice(contentStart, closingLineStart).trim(),
            start: lineStart,
            end: closingLineStart + closing,
          });

          if (options.firstOnly) {
            return blocks;
          }
        }

        cursor = afterClosingLine;
        closed = true;
        break;
      }

      if (inlineClosing) {
        if (!wantedLanguage || rawLanguage === wantedLanguage) {
          blocks.push({
            language: rawLanguage,
            code: input.slice(contentStart, closingLineStart + inlineClosing.start).trim(),
            start: lineStart,
            end: closingLineStart + inlineClosing.end,
          });

          if (options.firstOnly) {
            return blocks;
          }
        }

        cursor = afterClosingLine;
        closed = true;
        break;
      }

      if (afterClosingLine <= scanCursor) {
        break;
      }

      scanCursor = afterClosingLine;
    }

    if (!closed) {
      cursor = nextLineStart;
    }
  }

  return blocks;
}

export function extractFirstMarkdownCode(
  input: string,
  options: Omit<MarkdownCodeOptions, "firstOnly"> = {},
): MarkdownCodeBlock | null {
  return extractMarkdownCodeBlocks(input, { ...options, firstOnly: true })[0] || null;
}

interface OpeningFence {
  marker: "`" | "~";
  length: number;
  info: string;
}

function nextLineBounds(input: string, start: number): [lineEnd: number, nextStart: number] {
  const newline = input.indexOf("\n", start);
  if (newline === -1) {
    return [input.length, input.length];
  }
  return [newline, newline + 1];
}

function parseOpeningFence(line: string): OpeningFence | null {
  const end = stripTrailingCarriageReturn(line);
  const indent = leadingSpaces(end);
  if (indent > 3 || indent >= end.length) {
    return null;
  }

  const marker = end[indent];
  if (marker !== "`" && marker !== "~") {
    return null;
  }

  let fenceEnd = indent;
  while (fenceEnd < end.length && end[fenceEnd] === marker) {
    fenceEnd += 1;
  }

  const length = fenceEnd - indent;
  if (length < 3) {
    return null;
  }

  const info = end.slice(fenceEnd).trim();
  if (marker === "`" && info.includes("`")) {
    return null;
  }

  return { marker, length, info };
}

function parseClosingFence(line: string, marker: "`" | "~", openingLength: number): number | null {
  const end = stripTrailingCarriageReturn(line);
  const indent = leadingSpaces(end);
  if (indent > 3 || indent >= end.length) {
    return null;
  }

  if (end[indent] !== marker) {
    return null;
  }

  let fenceEnd = indent;
  while (fenceEnd < end.length && end[fenceEnd] === marker) {
    fenceEnd += 1;
  }

  if (fenceEnd - indent < openingLength) {
    return null;
  }

  for (let i = fenceEnd; i < end.length; i += 1) {
    const char = end[i];
    if (char !== " " && char !== "\t") {
      return null;
    }
  }

  return fenceEnd;
}

function parseInlineClosingFence(
  line: string,
  marker: "`" | "~",
  openingLength: number,
): { start: number; end: number } | null {
  const end = stripTrailingCarriageReturn(line);
  const trimmedEnd = trimRightWhitespaceIndex(end);
  if (trimmedEnd <= 0) {
    return null;
  }

  let fenceStart = trimmedEnd;
  while (fenceStart > 0 && end[fenceStart - 1] === marker) {
    fenceStart -= 1;
  }

  const fenceLength = trimmedEnd - fenceStart;
  if (fenceLength < openingLength) {
    return null;
  }

  // Standalone closing fences are handled by parseClosingFence.
  if (fenceStart === 0 || isOnlyWhitespace(end.slice(0, fenceStart))) {
    return null;
  }

  return {
    start: fenceStart,
    end: trimmedEnd,
  };
}

function extractInfoLanguage(info: string): string | null {
  if (!info) {
    return null;
  }

  let start = 0;
  while (start < info.length && isWhitespace(info[start])) {
    start += 1;
  }

  if (start >= info.length) {
    return null;
  }

  let end = start;
  while (end < info.length && !isWhitespace(info[end])) {
    end += 1;
  }

  let language = info.slice(start, end);
  if (language.startsWith(".")) {
    language = language.slice(1);
  }

  return language || null;
}

function normalizeLanguage(value: string | null | undefined): string | null {
  if (!value) {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  return normalized || null;
}

function leadingSpaces(value: string): number {
  let count = 0;
  while (count < value.length && value[count] === " ") {
    count += 1;
  }
  return count;
}

function stripTrailingCarriageReturn(value: string): string {
  if (value.endsWith("\r")) {
    return value.slice(0, -1);
  }
  return value;
}

function trimRightWhitespaceIndex(value: string): number {
  let end = value.length;
  while (end > 0 && isWhitespace(value[end - 1])) {
    end -= 1;
  }
  return end;
}

function isOnlyWhitespace(value: string): boolean {
  for (let i = 0; i < value.length; i += 1) {
    if (!isWhitespace(value[i])) {
      return false;
    }
  }
  return true;
}
