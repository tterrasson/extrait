import type { ThinkBlock, ThinkDiagnostics } from "./types";
import { isWhitespace } from "./utils/common";

export interface SanitizeThinkResult {
  visibleText: string;
  thinkBlocks: ThinkBlock[];
  diagnostics: ThinkDiagnostics;
}

export interface ThinkScanResult {
  thinkBlocks: ThinkBlock[];
  diagnostics: ThinkDiagnostics;
}

interface ThinkTagToken {
  type: "open" | "close";
  start: number;
  end: number;
}

const THINK_TAG_NAME = "think";
const RE_IDENTIFIER_CHAR = /[a-zA-Z0-9:_-]/;
const RE_NON_LINE_BREAK = /[^\r\n]/g;

export function scanThinkBlocks(input: string): ThinkScanResult {
  const thinkBlocks: ThinkBlock[] = [];
  const diagnostics: ThinkDiagnostics = {
    unterminatedCount: 0,
    nestedCount: 0,
    hiddenChars: 0,
  };

  let searchFrom = 0;
  let depth = 0;
  let blockStart = -1;
  let blockContentStart = -1;

  while (searchFrom < input.length) {
    const token = findNextThinkTag(input, searchFrom);
    if (!token) {
      break;
    }
    searchFrom = token.end;

    if (token.type === "open") {
      if (depth === 0) {
        blockStart = token.start;
        blockContentStart = token.end;
      } else {
        diagnostics.nestedCount += 1;
      }
      depth += 1;
      continue;
    }

    if (depth === 0) {
      continue;
    }

    depth -= 1;
    if (depth > 0) {
      continue;
    }

    const raw = input.slice(blockStart, token.end);
    thinkBlocks.push({
      id: `think:${thinkBlocks.length}`,
      content: input.slice(blockContentStart, token.start).trim(),
      raw,
      start: blockStart,
      end: token.end,
    });
    diagnostics.hiddenChars += countHiddenChars(raw);
    blockStart = -1;
    blockContentStart = -1;
  }

  if (depth > 0 && blockStart >= 0) {
    const raw = input.slice(blockStart);
    thinkBlocks.push({
      id: `think:${thinkBlocks.length}`,
      content: input.slice(blockContentStart).trim(),
      raw,
      start: blockStart,
      end: input.length,
    });
    diagnostics.hiddenChars += countHiddenChars(raw);
    diagnostics.unterminatedCount += 1;
  }

  return {
    thinkBlocks,
    diagnostics,
  };
}

export function sanitizeThink(input: string): SanitizeThinkResult {
  const { thinkBlocks, diagnostics } = scanThinkBlocks(input);
  if (thinkBlocks.length === 0) {
    return { visibleText: input, thinkBlocks, diagnostics };
  }

  let visibleText = "";
  let cursor = 0;
  for (const block of thinkBlocks) {
    visibleText += input.slice(cursor, block.start) + maskKeepingLineBreaks(block.raw);
    cursor = block.end;
  }
  visibleText += input.slice(cursor);

  return {
    visibleText,
    thinkBlocks,
    diagnostics,
  };
}

/**
 * Removes every `<think>` / `</think>` tag token the scanner recognizes,
 * including variants with attributes (`<think foo>`) or inner whitespace
 * (`</ think >`), leaving the surrounding content untouched. Reasoning text
 * must go through this (not a literal-tag regex) before being re-wrapped in a
 * synthetic `<think>` block: a surviving variant tag would unbalance the
 * composed source and make sanitization swallow the visible text after it.
 */
export function stripThinkTags(input: string): string {
  let output = "";
  let cursor = 0;

  while (cursor < input.length) {
    const token = findNextThinkTag(input, cursor);
    if (!token) {
      break;
    }
    output += input.slice(cursor, token.start);
    cursor = token.end;
  }

  if (cursor === 0) {
    return input;
  }
  return output + input.slice(cursor);
}

function findNextThinkTag(input: string, from: number): ThinkTagToken | null {
  let cursor = from;
  while (cursor < input.length) {
    const lt = input.indexOf("<", cursor);
    if (lt < 0) {
      return null;
    }

    const token = parseThinkTagAt(input, lt);
    if (token) {
      return token;
    }

    cursor = lt + 1;
  }

  return null;
}

function parseThinkTagAt(input: string, index: number): ThinkTagToken | null {
  if (input[index] !== "<") {
    return null;
  }

  let cursor = index + 1;
  let closing = false;

  if (input[cursor] === "/") {
    closing = true;
    cursor += 1;
    while (cursor < input.length && isWhitespace(input[cursor])) {
      cursor += 1;
    }
  }

  if (!matchesIgnoreCase(input, cursor, THINK_TAG_NAME)) {
    return null;
  }
  cursor += THINK_TAG_NAME.length;

  const next = input[cursor];
  if (next && isIdentifierChar(next)) {
    return null;
  }

  if (closing) {
    while (cursor < input.length && isWhitespace(input[cursor])) {
      cursor += 1;
    }

    if (input[cursor] !== ">") {
      return null;
    }

    return {
      type: "close",
      start: index,
      end: cursor + 1,
    };
  }

  const tagEnd = findTagEnd(input, cursor);
  return {
    type: "open",
    start: index,
    end: tagEnd >= 0 ? tagEnd : input.length,
  };
}

function findTagEnd(input: string, from: number): number {
  let quote: '"' | "'" | null = null;

  for (let i = from; i < input.length; i += 1) {
    const char = input[i];
    if (quote) {
      if (char === quote && input[i - 1] !== "\\") {
        quote = null;
      }
      continue;
    }

    if (char === '"' || char === "'") {
      quote = char;
      continue;
    }

    if (char === ">") {
      return i + 1;
    }
  }

  return -1;
}

function matchesIgnoreCase(input: string, index: number, expected: string): boolean {
  if (index + expected.length > input.length) {
    return false;
  }

  return input.slice(index, index + expected.length).toLowerCase() === expected;
}

function isIdentifierChar(char: string): boolean {
  return RE_IDENTIFIER_CHAR.test(char);
}

function countHiddenChars(value: string): number {
  let count = 0;
  for (let i = 0; i < value.length; i += 1) {
    const char = value[i];
    if (char !== "\n" && char !== "\r") {
      count += 1;
    }
  }
  return count;
}

function maskKeepingLineBreaks(value: string): string {
  return value.replace(RE_NON_LINE_BREAK, " ");
}
