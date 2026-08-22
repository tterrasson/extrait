// Tolerant JSON parser for the live `snapshot.data` preview.
//
// It resumes where it stopped instead of re-reading the accumulated text, so a
// stream costs O(total output) rather than O(output) per event. Containers are
// built in place and only the still-open ones are copied on read, which keeps
// each snapshot immutable while completed subtrees keep their identity.
//
// Rendering is deliberately hysteretic: a value already shown survives a stall
// further along, so on malformed text the preview reflects what the parser
// managed to render and not the accumulated text alone.
//
// This drives the preview only. The authoritative parse of the final output
// runs through `parseLLMOutput`, so anything the preview cannot render (or
// renders approximately) is still parsed strictly at the end.

type Frame =
  | { kind: "object"; value: Record<string, unknown>; key: string | null }
  // `slot` is the index the value being read will fill. Writing there instead
  // of pushing is what makes a value that grows across events idempotent: the
  // rendering of a partial value is overwritten, never stacked.
  | { kind: "array"; value: unknown[]; slot: number };

type Mode = "value" | "afterValue" | "key" | "colon";

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

const HIGH_SURROGATE_START = 0xd800;
const HIGH_SURROGATE_END = 0xdbff;

// A bracket in the prose can look like the payload (`- [x] done`). When such a
// root turns out not to be JSON at all, the search moves past it rather than
// leaving the preview dead for the rest of the stream. The number of candidates
// dropped is capped, so a run of `[[[[` cannot make the rescan quadratic. Prose
// that happens to be valid JSON (`[1]`, `- [ ] plan`) is indistinguishable from
// the payload here and still wins; the authoritative parse ranks candidates.
const MAX_DROPPED_ROOTS = 8;

// Longest head of an unterminated literal worth rendering: past `-` plus 17
// significant digits, an exponent and a sign, a double holds nothing more.
const MAX_PARTIAL_LITERAL = 32;

const ESCAPES: Record<string, string | undefined> = {
  '"': '"',
  "\\": "\\",
  "/": "/",
  b: "\b",
  f: "\f",
  n: "\n",
  r: "\r",
  t: "\t",
};

export interface StreamingStructuredParser {
  /** Feeds the accumulated visible text; returns the preview, or null. */
  update(visibleText: string): unknown | null;
}

export function createStreamingStructuredParser(): StreamingStructuredParser {
  let text = "";

  // Root search. Only a bracket found outside a string counts, so a decoy like
  // `"draft: {not the payload}"` never starts a parse.
  let rootStart = -1;
  let scanPos = 0;
  let scanInString = false;
  let scanEscaped = false;
  let droppedRoots = 0;

  let activeStart = -1;
  let pos = 0;
  let mode: Mode = "value";
  let stack: Frame[] = [];
  let root: unknown;
  let rooted = false;
  let complete = false;
  // Whether this root ever rendered a leaf. One that stalls without having
  // shown a single value is the one worth trading for another candidate.
  let rendered = false;
  // Set once the search has moved past the active root, which then stays on
  // screen only until a better candidate arrives.
  let abandoned = false;
  // Set on syntax the parser cannot advance past; the preview then holds its
  // last good state instead of flickering.
  let stalled = false;

  // Scan state for the string currently open at the tail. A long text field
  // spans many events, so the search for its closing quote resumes instead of
  // restarting, and an escape-free body skips decoding entirely.
  let openStringAt = -1;
  let openStringScan = 0;
  let openStringHasEscape = false;

  function resetScanner(from: number): void {
    rootStart = -1;
    scanPos = from;
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
    rendered = false;
    abandoned = false;
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

  /**
   * Writes into the slot the value being read occupies. Re-reading a value that
   * has grown since the last event overwrites that slot, so nothing has to be
   * withdrawn first and a value already shown survives a stall further along.
   */
  function assign(value: unknown): void {
    const frame = stack[stack.length - 1];
    if (!frame) {
      root = value;
      rooted = true;
      return;
    }
    if (frame.kind === "array") {
      frame.value[frame.slot] = value;
      return;
    }
    if (frame.key !== null) {
      frame.value[frame.key] = value;
    }
  }

  /**
   * Empties that slot: a partial value that stopped rendering must not leave a
   * stale guess behind. An object keeps its key visible as null.
   */
  function holdSlot(): void {
    const frame = stack[stack.length - 1];
    if (!frame) {
      return;
    }
    if (frame.kind === "array") {
      frame.value.length = frame.slot;
      return;
    }
    if (frame.key !== null) {
      frame.value[frame.key] = null;
    }
  }

  /** The value in the slot is final: move on to the next one. */
  function closeSlot(): void {
    const frame = stack[stack.length - 1];
    if (frame?.kind === "array") {
      frame.slot += 1;
    }
    mode = "afterValue";
    if (stack.length === 0) {
      complete = true;
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

  /**
   * Unescapes a body the strict decode rejected — a raw control character, an
   * escape JSON does not define. An undefined escape renders as its own
   * character, which is what the repairing parse settles on too.
   */
  function decodeLoosely(body: string): string {
    let out = "";
    for (let index = 0; index < body.length; index += 1) {
      const char = body[index] as string;
      if (char !== "\\") {
        out += char;
        continue;
      }
      const next = body[index + 1];
      if (next === undefined) {
        break;
      }
      if (next === "u") {
        const hex = body.slice(index + 2, index + 6);
        if (/^[0-9a-fA-F]{4}$/.test(hex)) {
          out += String.fromCharCode(Number.parseInt(hex, 16));
          index += 5;
          continue;
        }
      }
      out += ESCAPES[next] ?? next;
      index += 1;
    }
    return out;
  }

  /** `raw` always carries its two quotes. */
  function decodeString(raw: string): string {
    try {
      return JSON.parse(raw) as string;
    } catch {
      return decodeLoosely(raw.slice(1, -1));
    }
  }

  /** Drops a surrogate whose other half has not arrived yet. */
  function trimLoneSurrogate(value: string): string {
    const last = value.charCodeAt(value.length - 1);
    return last >= HIGH_SURROGATE_START && last <= HIGH_SURROGATE_END ? value.slice(0, -1) : value;
  }

  /** Renders an open string, dropping a half-written escape at its tail. */
  function decodePartialString(start: number): string {
    if (!openStringHasEscape) {
      return trimLoneSurrogate(text.slice(start + 1));
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
    return trimLoneSurrogate(decodeString(`"${body}"`));
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
    assign(frame.value);
    stack.push(frame);
  }

  /** Closes the top container if it has the given kind, otherwise stalls. */
  function tryClose(kind: Frame["kind"]): boolean {
    if (stack[stack.length - 1]?.kind !== kind) {
      stalled = true;
      return false;
    }
    pos += 1;
    stack.pop();
    closeSlot();
    return true;
  }

  function commitValue(value: unknown, end: number): void {
    assign(value);
    pos = end;
    rendered = true;
    closeSlot();
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
      openContainer({ kind: "array", value: [], slot: 0 });
      mode = "value";
      return true;
    }
    if (code === CHAR_CLOSE_BRACKET) {
      return tryClose("array");
    }
    // The value never came (`{"a": , ...}`): leave the slot empty and let the
    // delimiter be read where it belongs.
    if (code === CHAR_COMMA || code === CHAR_CLOSE_BRACE) {
      holdSlot();
      mode = "afterValue";
      return true;
    }
    if (code === CHAR_QUOTE) {
      const end = findStringEnd(pos);
      if (end < 0) {
        assign(decodePartialString(pos));
        rendered = true;
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
      // No delimiter yet, so the literal may still grow. Only the head of it
      // can carry a renderable value, and reading no further keeps a literal
      // that never terminates from costing its length on every event.
      const partial = parsePartialScalar(text.slice(pos, pos + MAX_PARTIAL_LITERAL));
      if (partial.ok) {
        assign(partial.value);
        rendered = true;
      } else {
        holdSlot();
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
    const frame = stack[stack.length - 1];
    if (code === CHAR_COMMA) {
      pos += 1;
      mode = frame?.kind === "array" ? "value" : "key";
      return true;
    }
    if (code === CHAR_CLOSE_BRACE) {
      return tryClose("object");
    }
    if (code === CHAR_CLOSE_BRACKET) {
      return tryClose("array");
    }
    // A dropped comma is the common model slip; infer it rather than stall.
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
          holdSlot();
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
          copy[frame.slot] = child;
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
      const code = text.charCodeAt(scanPos);
      scanPos += 1;
      if (scanInString) {
        if (scanEscaped) {
          scanEscaped = false;
        } else if (code === CHAR_BACKSLASH) {
          scanEscaped = true;
        } else if (code === CHAR_QUOTE) {
          scanInString = false;
        }
        continue;
      }
      if (code === CHAR_QUOTE) {
        scanInString = true;
        continue;
      }
      if (code === CHAR_OPEN_BRACE || code === CHAR_OPEN_BRACKET) {
        rootStart = scanPos - 1;
      }
    }
  }

  /**
   * Moves the root search past a bracket that turned out not to open JSON. Only
   * a root that stalled without ever rendering a value is dropped, so a preview
   * the caller has already seen is never traded away, and the dropped root
   * keeps being shown until a better candidate actually arrives.
   */
  function dropDecoyRoot(): void {
    while (stalled && !rendered && droppedRoots < MAX_DROPPED_ROOTS) {
      if (!abandoned) {
        droppedRoots += 1;
        abandoned = true;
        resetScanner(activeStart + 1);
      }
      advanceScanner();
      if (rootStart < 0) {
        // Nothing better in sight; the scanner picks up where it stopped on the
        // next event, without rewinding or spending the budget again.
        return;
      }
      resetParser(rootStart);
      resume();
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
        resetScanner(0);
        resetParser(-1);
        text = "";
      }

      text = visibleText;
      advanceScanner();

      if (rootStart >= 0 && rootStart !== activeStart) {
        resetParser(rootStart);
      }
      if (activeStart >= 0) {
        resume();
        dropDecoyRoot();
      }

      if (!rooted || typeof root !== "object" || root === null) {
        return null;
      }
      return snapshot();
    },
  };
}
