// Tolerant JSON parser for the live `snapshot.data` preview.
//
// It resumes where it stopped instead of re-reading the accumulated text, so a
// stream costs O(total output) rather than O(output) per event. Containers are
// built in place and only the still-open ones are copied on read, which keeps
// each snapshot immutable while completed subtrees keep their identity.
//
// This drives the preview only. The authoritative parse of the final output
// runs through `parseLLMOutput`, so anything the preview cannot render (or
// renders approximately) is still parsed strictly at the end.

type Frame =
  | { kind: "object"; value: Record<string, unknown>; key: string | null }
  | { kind: "array"; value: unknown[] };

type Mode = "value" | "afterValue" | "key" | "colon";

// Slot holding a value that may still grow (an unterminated string, a literal
// with no delimiter yet). It is withdrawn before each resume and re-derived.
type PendingSlot =
  | { kind: "root" }
  | { kind: "array"; target: unknown[] }
  | { kind: "object"; target: Record<string, unknown>; key: string }
  | null;

const CHAR_TAB = 9;
const CHAR_NEWLINE = 10;
const CHAR_RETURN = 13;
const CHAR_SPACE = 32;
const CHAR_QUOTE = 34;
const CHAR_COMMA = 44;
const CHAR_MINUS = 45;
const CHAR_COLON = 58;
const CHAR_OPEN_BRACKET = 91;
const CHAR_BACKSLASH = 92;
const CHAR_CLOSE_BRACKET = 93;
const CHAR_OPEN_BRACE = 123;
const CHAR_CLOSE_BRACE = 125;
const CHAR_ZERO = 48;
const CHAR_NINE = 57;
const CHAR_LOWER_T = 116;
const CHAR_LOWER_F = 102;
const CHAR_LOWER_N = 110;

export interface StreamingStructuredParser {
  /** Feeds the accumulated visible text; returns the preview, or null. */
  update(visibleText: string): unknown | null;
}

export function createStreamingStructuredParser(): StreamingStructuredParser {
  let text = "";

  // Root search. Only a bracket found outside a string counts, so a decoy like
  // `"draft: {not the payload}"` never starts a parse. The result never changes
  // once set, which is what lets the parse below run forward-only.
  let rootStart = -1;
  let scanPos = 0;
  let scanInString = false;
  let scanEscaped = false;

  let activeStart = -1;
  let pos = 0;
  let mode: Mode = "value";
  let stack: Frame[] = [];
  let root: unknown;
  let rooted = false;
  let complete = false;
  let pending: PendingSlot = null;
  // Set on syntax the parser cannot advance past; the preview then holds its
  // last good state instead of flickering.
  let stalled = false;

  // Scan state for the string currently open at the tail. A long text field
  // spans many events, so the search for its closing quote resumes instead of
  // restarting, and an escape-free body skips decoding entirely.
  let openStringAt = -1;
  let openStringScan = 0;
  let openStringHasEscape = false;

  function resetScanner(): void {
    rootStart = -1;
    scanPos = 0;
    scanInString = false;
    scanEscaped = false;
  }

  function resetParser(start: number): void {
    activeStart = start;
    pos = start;
    mode = "value";
    stack = [];
    root = undefined;
    rooted = false;
    complete = false;
    pending = null;
    stalled = false;
    openStringAt = -1;
    openStringScan = 0;
    openStringHasEscape = false;
  }

  function isWhitespace(code: number): boolean {
    return code === CHAR_SPACE || code === CHAR_TAB || code === CHAR_NEWLINE || code === CHAR_RETURN;
  }

  function isDelimiter(code: number): boolean {
    return (
      isWhitespace(code) ||
      code === CHAR_COMMA ||
      code === CHAR_CLOSE_BRACE ||
      code === CHAR_CLOSE_BRACKET
    );
  }

  function startsValue(code: number): boolean {
    return (
      code === CHAR_OPEN_BRACE ||
      code === CHAR_OPEN_BRACKET ||
      code === CHAR_QUOTE ||
      code === CHAR_MINUS ||
      (code >= CHAR_ZERO && code <= CHAR_NINE) ||
      code === CHAR_LOWER_T ||
      code === CHAR_LOWER_F ||
      code === CHAR_LOWER_N
    );
  }

  function withdrawPending(): void {
    if (!pending) {
      return;
    }
    if (pending.kind === "array") {
      pending.target.pop();
    } else if (pending.kind === "object") {
      delete pending.target[pending.key];
    } else {
      root = undefined;
      rooted = false;
    }
    pending = null;
  }

  function assign(value: unknown, isPending: boolean): void {
    const frame = stack[stack.length - 1];
    if (!frame) {
      root = value;
      rooted = true;
      if (isPending) {
        pending = { kind: "root" };
      }
      return;
    }
    if (frame.kind === "array") {
      frame.value.push(value);
      if (isPending) {
        pending = { kind: "array", target: frame.value };
      }
      return;
    }
    if (frame.key === null) {
      return;
    }
    frame.value[frame.key] = value;
    if (isPending) {
      pending = { kind: "object", target: frame.value, key: frame.key };
    }
  }

  function skipWhitespace(): boolean {
    while (pos < text.length && isWhitespace(text.charCodeAt(pos))) {
      pos += 1;
    }
    return pos < text.length;
  }

  /**
   * Index just past the closing quote, or -1 while the string is still open.
   * Resumes where the previous call stopped, so a string spanning many events
   * is walked once overall.
   */
  function findStringEnd(start: number): number {
    if (openStringAt !== start) {
      openStringAt = start;
      openStringScan = start + 1;
      openStringHasEscape = false;
    }
    let index = openStringScan;
    while (index < text.length) {
      const code = text.charCodeAt(index);
      if (code === CHAR_BACKSLASH) {
        openStringHasEscape = true;
        index += 2;
        continue;
      }
      if (code === CHAR_QUOTE) {
        openStringAt = -1;
        return index + 1;
      }
      index += 1;
    }
    openStringScan = index;
    return -1;
  }

  function decodeString(raw: string): string {
    try {
      return JSON.parse(raw) as string;
    } catch {
      return raw.slice(1, raw.endsWith('"') ? -1 : undefined);
    }
  }

  /** Renders an open string, dropping a half-written escape at its tail. */
  function decodePartialString(start: number): string {
    if (!openStringHasEscape) {
      return text.slice(start + 1);
    }
    let body = text.slice(start + 1);
    const trailingBackslashes = /\\*$/.exec(body)?.[0].length ?? 0;
    if (trailingBackslashes % 2 === 1) {
      body = body.slice(0, -1);
    }
    // Only a `\u` whose backslash is not itself escaped starts an escape.
    const partialUnicode = /(\\*)(\\u[0-9a-fA-F]{0,3})$/.exec(body);
    if (partialUnicode && (partialUnicode[1]?.length ?? 0) % 2 === 0) {
      body = body.slice(0, -(partialUnicode[2]?.length ?? 0));
    }
    return decodeString(`"${body}"`);
  }

  function parseScalar(token: string): { ok: boolean; value: unknown } {
    try {
      return { ok: true, value: JSON.parse(token) as unknown };
    } catch {
      return { ok: false, value: undefined };
    }
  }

  /** Renders an unfinished literal: `tr` -> true, `12.` -> 12. */
  function parsePartialScalar(token: string): { ok: boolean; value: unknown } {
    if (token.length === 0) {
      return { ok: false, value: undefined };
    }
    if ("true".startsWith(token)) {
      return { ok: true, value: true };
    }
    if ("false".startsWith(token)) {
      return { ok: true, value: false };
    }
    if ("null".startsWith(token)) {
      return { ok: true, value: null };
    }
    // Longest prefix that is a complete number per the JSON grammar.
    const number = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(token)?.[0];
    if (number) {
      return parseScalar(number);
    }
    return { ok: false, value: undefined };
  }

  function openContainer(frame: Frame): void {
    assign(frame.value, false);
    stack.push(frame);
  }

  function closeContainer(): void {
    stack.pop();
    mode = "afterValue";
    if (stack.length === 0) {
      complete = true;
    }
  }

  /** Closes the top container if it has the given kind, otherwise stalls. */
  function tryClose(kind: Frame["kind"]): boolean {
    if (stack[stack.length - 1]?.kind !== kind) {
      stalled = true;
      return false;
    }
    pos += 1;
    closeContainer();
    return true;
  }

  function commitValue(value: unknown, end: number): void {
    assign(value, false);
    pos = end;
    mode = "afterValue";
    if (stack.length === 0) {
      complete = true;
    }
  }

  /** Keeps a keyed slot visible as null while its value has not arrived. */
  function holdObjectSlot(): void {
    if (stack[stack.length - 1]?.kind === "object") {
      assign(null, true);
    }
  }

  /** One step in "value" mode; false when parsing must pause. */
  function stepValue(code: number): boolean {
    if (code === CHAR_OPEN_BRACE) {
      pos += 1;
      openContainer({ kind: "object", value: {}, key: null });
      mode = "key";
      return true;
    }
    if (code === CHAR_OPEN_BRACKET) {
      pos += 1;
      openContainer({ kind: "array", value: [] });
      mode = "value";
      return true;
    }
    if (code === CHAR_CLOSE_BRACKET) {
      return tryClose("array");
    }
    if (code === CHAR_QUOTE) {
      const end = findStringEnd(pos);
      if (end < 0) {
        assign(decodePartialString(pos), true);
        return false;
      }
      commitValue(decodeString(text.slice(pos, end)), end);
      return true;
    }

    let end = pos;
    while (end < text.length && !isDelimiter(text.charCodeAt(end))) {
      end += 1;
    }
    if (end >= text.length) {
      // No delimiter yet, so the literal may still grow.
      const partial = parsePartialScalar(text.slice(pos));
      if (partial.ok) {
        assign(partial.value, true);
      } else {
        holdObjectSlot();
      }
      return false;
    }
    const scalar = parseScalar(text.slice(pos, end));
    if (!scalar.ok) {
      stalled = true;
      return false;
    }
    commitValue(scalar.value, end);
    return true;
  }

  /** One step in "key" mode; false when parsing must pause. */
  function stepKey(code: number): boolean {
    if (code === CHAR_CLOSE_BRACE) {
      return tryClose("object");
    }
    if (code !== CHAR_QUOTE) {
      stalled = true;
      return false;
    }
    const end = findStringEnd(pos);
    if (end < 0) {
      return false;
    }
    const frame = stack[stack.length - 1];
    if (frame?.kind !== "object") {
      stalled = true;
      return false;
    }
    frame.key = decodeString(text.slice(pos, end));
    pos = end;
    mode = "colon";
    return true;
  }

  /** One step in "colon" mode; false when parsing must pause. */
  function stepColon(code: number): boolean {
    if (code !== CHAR_COLON) {
      stalled = true;
      return false;
    }
    pos += 1;
    mode = "value";
    return true;
  }

  /** One step in "afterValue" mode; false when parsing must pause. */
  function stepAfterValue(code: number): boolean {
    if (code === CHAR_COMMA) {
      pos += 1;
      const frame = stack[stack.length - 1];
      if (!frame) {
        complete = true;
        return false;
      }
      mode = frame.kind === "object" ? "key" : "value";
      return true;
    }
    if (code === CHAR_CLOSE_BRACE) {
      return tryClose("object");
    }
    if (code === CHAR_CLOSE_BRACKET) {
      return tryClose("array");
    }
    // A dropped comma is the common model slip; infer it rather than stall.
    const frame = stack[stack.length - 1];
    if (frame?.kind === "object" && code === CHAR_QUOTE) {
      mode = "key";
      return true;
    }
    if (frame?.kind === "array" && startsValue(code)) {
      mode = "value";
      return true;
    }
    stalled = true;
    return false;
  }

  function resume(): void {
    while (!complete && !stalled) {
      if (!skipWhitespace()) {
        if (mode === "value" || mode === "colon") {
          holdObjectSlot();
        }
        return;
      }
      const code = text.charCodeAt(pos);
      const advanced =
        mode === "value"
          ? stepValue(code)
          : mode === "key"
            ? stepKey(code)
            : mode === "colon"
              ? stepColon(code)
              : stepAfterValue(code);
      if (!advanced) {
        return;
      }
    }
  }

  /**
   * Copies the containers still open so callers get an immutable value; the
   * completed subtrees below them are shared, so references stay stable.
   */
  function snapshot(): unknown {
    if (stack.length === 0) {
      return root;
    }
    let child: unknown;
    let hasChild = false;
    for (let index = stack.length - 1; index >= 0; index -= 1) {
      const frame = stack[index];
      if (!frame) {
        continue;
      }
      if (frame.kind === "array") {
        const copy = frame.value.slice();
        if (hasChild) {
          copy[copy.length - 1] = child;
        }
        child = copy;
      } else {
        const copy = { ...frame.value };
        if (hasChild && frame.key !== null) {
          copy[frame.key] = child;
        }
        child = copy;
      }
      hasChild = true;
    }
    return child;
  }

  function advanceScanner(): void {
    while (rootStart < 0 && scanPos < text.length) {
      const char = text[scanPos];
      scanPos += 1;
      if (scanInString) {
        if (scanEscaped) {
          scanEscaped = false;
          continue;
        }
        if (char === "\\") {
          scanEscaped = true;
          continue;
        }
        if (char === '"') {
          scanInString = false;
        }
        continue;
      }
      if (char === '"') {
        scanInString = true;
        continue;
      }
      if (char === "{" || char === "[") {
        rootStart = scanPos - 1;
      }
    }
  }

  /**
   * Think-tag sanitization can rewrite earlier text, which invalidates the
   * state. Everything any cursor has consumed must be unchanged — including
   * the lookahead of the string open at the tail, which runs past `pos`.
   */
  function extendsPreviousText(next: string): boolean {
    // `openStringScan` may sit one past the end after a trailing backslash;
    // the character there was never read, so it does not need verification.
    const stringScan = openStringAt >= 0 ? Math.min(openStringScan, text.length) : 0;
    const verified = Math.max(pos, scanPos, stringScan);
    if (next.length < verified) {
      return false;
    }
    return next.startsWith(verified === text.length ? text : text.slice(0, verified));
  }

  return {
    update(visibleText: string): unknown | null {
      if (!extendsPreviousText(visibleText)) {
        resetScanner();
        resetParser(-1);
        text = "";
      }

      text = visibleText;
      advanceScanner();

      if (rootStart < 0) {
        return null;
      }
      if (rootStart !== activeStart) {
        resetParser(rootStart);
      }

      withdrawPending();
      resume();

      if (!rooted || typeof root !== "object" || root === null) {
        return null;
      }
      return snapshot();
    },
  };
}
