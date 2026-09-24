// Incremental counterpart of `normalizeModelOutput` for streams.
//
// `normalizeModelOutput(text, reasoning)` hides `<think>` blocks from the text,
// gathers their content and the dedicated reasoning into one reasoning string,
// and callers diff successive results into deltas. Doing that on the
// accumulated output costs the whole output on every chunk. This module reads
// each chunk once instead:
//
// - A tag scanner turns chunks into characters and think tags. A trailing `<`
//   fragment that could still become a tag (`<thi`) is held back until the
//   next chunk decides it.
// - The rendered text and reasoning are append-only strings (engine ropes,
//   never read here), plus a provisional tail: the held-back fragment rendered
//   as if the stream ended now, which is what `normalizeModelOutput` would do.
// - Deltas and equality are decided on a small window at the end of each
//   field, since everything before it is shared and unchanged.
//
// Every update renders exactly what `normalizeModelOutput` renders for the
// accumulated input, and folding the deltas (appending, or replacing on
// `resync`) always yields its stable part (see tests/streaming-normalizer.test.ts). A rare shape, reasoning arriving on the
// dedicated channel after an inline think block, changes the middle of the
// reasoning; that update falls back to comparing full strings.

import { normalizeModelOutput, withoutTrailingThinkTagPrefix } from "./generate-output";

// Characters kept at the end of each field for the delta window. Must exceed
// the longest partial think tag a stable snapshot withholds (7).
const WINDOW = 16;

const SEPARATOR = "\n\n";
const RE_NON_SPACE = /\S/;
const RE_IDENTIFIER_CHAR = /[a-zA-Z0-9:_-]/;
const TAG_NAME = "think";

type TagClass =
  | { kind: "none" }
  | { kind: "undecided" }
  | { kind: "open"; end: number }
  | { kind: "close"; end: number };

function isTagSpace(char: string | undefined): boolean {
  return char === " " || char === "\t" || char === "\n" || char === "\r";
}

/**
 * Decides whether the `<` at `start` opens a think tag, mirroring
 * `parseThinkTagAt` in think.ts. While more input may follow, running out of
 * characters before the answer is known yields "undecided"; at the end of the
 * input it yields what the batch scanner concludes on the same characters.
 */
function classifyTag(input: string, start: number, final: boolean): TagClass {
  const unknown: TagClass = final ? { kind: "none" } : { kind: "undecided" };
  let cursor = start + 1;
  if (cursor >= input.length) {
    return unknown;
  }

  const closing = input[cursor] === "/";
  if (closing) {
    cursor += 1;
    while (cursor < input.length && isTagSpace(input[cursor])) {
      cursor += 1;
    }
  }

  for (let index = 0; index < TAG_NAME.length; index += 1) {
    const char = input[cursor + index];
    if (char === undefined) {
      return unknown;
    }
    if (char.toLowerCase() !== TAG_NAME[index]) {
      return { kind: "none" };
    }
  }
  cursor += TAG_NAME.length;

  const next = input[cursor];
  if (next === undefined) {
    // `<think` at the very end is an unterminated open tag for the batch
    // scanner, but `<thinking` is no tag at all: only the end decides.
    return closing || !final ? unknown : { kind: "open", end: cursor };
  }
  if (RE_IDENTIFIER_CHAR.test(next)) {
    return { kind: "none" };
  }

  if (!closing) {
    return { kind: "open", end: cursor };
  }

  while (cursor < input.length && isTagSpace(input[cursor])) {
    cursor += 1;
  }
  if (cursor >= input.length) {
    return unknown;
  }
  return input[cursor] === ">" ? { kind: "close", end: cursor + 1 } : { kind: "none" };
}

interface TagSink {
  chars(text: string): void;
  /** An open tag was recognized; `raw` runs up to the end of its name. */
  openTag(raw: string): void;
  /** More raw characters of the open tag being read, up to its `>`. */
  openTagChars(raw: string): void;
  openTagEnd(): void;
  closeTag(raw: string): void;
}

/** Splits a character stream into text and think tags; see `classifyTag`. */
class TagScanner {
  private pending = "";
  private inOpenTag = false;
  private quote: string | null = null;
  private previousChar = "";

  get hasPending(): boolean {
    return this.pending.length > 0;
  }

  clone(): TagScanner {
    return Object.assign(new TagScanner(), this);
  }

  feed(chunk: string, sink: TagSink): void {
    const input = this.pending + chunk;
    this.pending = "";
    this.scan(input, sink, false);
  }

  finish(sink: TagSink): void {
    const input = this.pending;
    this.pending = "";
    this.scan(input, sink, true);
  }

  private scan(input: string, sink: TagSink, final: boolean): void {
    let cursor = 0;
    while (cursor < input.length) {
      if (this.inOpenTag) {
        cursor = this.scanOpenTag(input, cursor, sink);
        continue;
      }

      const lt = input.indexOf("<", cursor);
      if (lt < 0) {
        sink.chars(input.slice(cursor));
        return;
      }
      if (lt > cursor) {
        sink.chars(input.slice(cursor, lt));
      }

      const tag = classifyTag(input, lt, final);
      if (tag.kind === "undecided") {
        this.pending = input.slice(lt);
        return;
      }
      if (tag.kind === "none") {
        sink.chars("<");
        cursor = lt + 1;
        continue;
      }
      if (tag.kind === "close") {
        sink.closeTag(input.slice(lt, tag.end));
        cursor = tag.end;
        continue;
      }

      sink.openTag(input.slice(lt, tag.end));
      this.inOpenTag = true;
      this.quote = null;
      this.previousChar = input[tag.end - 1] ?? "";
      cursor = tag.end;
    }
  }

  /** Reads the open tag up to its `>`, skipping quoted attribute values. */
  private scanOpenTag(input: string, from: number, sink: TagSink): number {
    for (let index = from; index < input.length; index += 1) {
      const char = input[index] as string;
      if (this.quote) {
        if (char === this.quote && this.previousChar !== "\\") {
          this.quote = null;
        }
      } else if (char === '"' || char === "'") {
        this.quote = char;
      } else if (char === ">") {
        sink.openTagChars(input.slice(from, index + 1));
        this.inOpenTag = false;
        this.previousChar = char;
        sink.openTagEnd();
        return index + 1;
      }
      this.previousChar = char;
    }

    sink.openTagChars(input.slice(from));
    return input.length;
  }
}

/** Append-only string plus the bookkeeping the delta window needs. */
class Field {
  value = "";
  /** The last `WINDOW` characters of `value`. */
  tail = "";
  /** What was appended since the last mark. */
  appended = "";
  /** Set when `value` changed other than by appending since the last mark. */
  replaced = false;

  clone(): Field {
    return Object.assign(new Field(), this);
  }

  append(text: string): void {
    if (text.length === 0) {
      return;
    }
    this.value += text;
    this.appended += text;
    this.tail = (this.tail + text).slice(-WINDOW);
  }

  replace(value: string): void {
    this.value = value;
    this.tail = value.slice(-WINDOW);
    this.replaced = true;
  }

  mark(): void {
    this.appended = "";
    this.replaced = false;
  }
}

/**
 * A string kept trimmed as it grows, the way `.trim()` would leave the whole:
 * whitespace after the last visible character waits in `pending` until more
 * visible text follows it.
 */
class TrimmedText {
  body = "";
  pending = "";

  clone(): TrimmedText {
    return Object.assign(new TrimmedText(), this);
  }

  /** Appends `text` and returns what that added to `body`. */
  append(text: string): string {
    const first = text.search(RE_NON_SPACE);
    if (first < 0) {
      if (this.body.length > 0) {
        this.pending += text;
      }
      return "";
    }

    let last = text.length - 1;
    while (!RE_NON_SPACE.test(text[last] as string)) {
      last -= 1;
    }

    const added = this.body.length === 0 ? text.slice(first, last + 1) : this.pending + text.slice(0, last + 1);
    this.body += added;
    this.pending = text.slice(last + 1);
    return added;
  }
}

/**
 * The normalized output built from scanner events: visible text, and the
 * reasoning joined from the dedicated channel and every think block, in that
 * order, each trimmed and separated by a blank line.
 */
class OutputState {
  readonly text = new Field();
  readonly reasoning = new Field();
  readonly textScanner = new TagScanner();
  readonly reasoningScanner = new TagScanner();
  dedicated = new TrimmedText();
  /** Completed think blocks with content, already joined. */
  blocks = "";
  block: TrimmedText | undefined;
  depth = 0;
  inBlockOpenTag = false;

  readonly textSink: TagSink = {
    chars: (text) => {
      if (this.depth === 0) {
        this.text.append(text);
      } else {
        this.appendToBlock(text);
      }
    },
    openTag: (raw) => {
      if (this.depth === 0) {
        this.block = new TrimmedText();
        this.inBlockOpenTag = true;
      } else {
        this.appendToBlock(raw);
      }
      this.depth += 1;
    },
    openTagChars: (raw) => {
      if (!this.inBlockOpenTag) {
        this.appendToBlock(raw);
      }
    },
    openTagEnd: () => {
      this.inBlockOpenTag = false;
    },
    closeTag: (raw) => {
      if (this.depth === 0) {
        this.text.append(raw);
        return;
      }
      this.depth -= 1;
      if (this.depth > 0) {
        this.appendToBlock(raw);
        return;
      }
      const content = this.block?.body ?? "";
      if (content.length > 0) {
        this.blocks = this.blocks.length > 0 ? this.blocks + SEPARATOR + content : content;
      }
      this.block = undefined;
    },
  };

  // `stripThinkTags` semantics: every tag goes, whatever the nesting.
  readonly reasoningSink: TagSink = {
    chars: (text) => this.appendToDedicated(text),
    openTag: () => {},
    openTagChars: () => {},
    openTagEnd: () => {},
    closeTag: () => {},
  };

  clone(): OutputState {
    const copy = new OutputState();
    Object.assign(copy.text, this.text);
    Object.assign(copy.reasoning, this.reasoning);
    Object.assign(copy.textScanner, this.textScanner);
    Object.assign(copy.reasoningScanner, this.reasoningScanner);
    copy.dedicated = this.dedicated.clone();
    copy.blocks = this.blocks;
    copy.block = this.block?.clone();
    copy.depth = this.depth;
    copy.inBlockOpenTag = this.inBlockOpenTag;
    return copy;
  }

  private appendToBlock(text: string): void {
    const block = this.block;
    if (!block) {
      return;
    }
    const wasEmpty = block.body.length === 0;
    const added = block.append(text);
    if (added.length === 0) {
      return;
    }
    // The open block is the last reasoning segment, so its growth is an append.
    const hasEarlier = this.dedicated.body.length > 0 || this.blocks.length > 0;
    this.reasoning.append(wasEmpty && hasEarlier ? SEPARATOR + added : added);
  }

  private appendToDedicated(text: string): void {
    const added = this.dedicated.append(text);
    if (added.length === 0) {
      return;
    }
    const hasLater = this.blocks.length > 0 || (this.block?.body.length ?? 0) > 0;
    if (!hasLater) {
      this.reasoning.append(added);
      return;
    }
    // The dedicated reasoning comes first: growing it now edits the middle.
    this.reasoning.replace(
      [this.dedicated.body, this.blocks, this.block?.body ?? ""].filter((part) => part.length > 0).join(SEPARATOR),
    );
  }
}

/** Tracks what was last reported for one field, as a window at its end. */
class FieldReport {
  // Position in the field where the window starts; everything before it was
  // reported and can no longer change while the field only grows.
  private windowStart = 0;
  private baseWindow = "";
  private renderedWindow = "";
  private stableWindow = "";
  private rendered = "";
  private stableLength = 0;

  /**
   * Reports `field` rendered with `provisional` appended: whether it changed,
   * whether it extends the previous report, and the newly stable delta. The
   * stable form withholds a trailing partial think tag unless `done`.
   */
  update(field: Field, provisional: Field | undefined, done: boolean): {
    value: string;
    changed: boolean;
    extends: boolean;
    delta: string;
    resync: boolean;
  } {
    const current = provisional ?? field;
    const value = current.value;
    let result: { changed: boolean; extends: boolean; delta: string; resync: boolean };

    if (!done && !field.replaced && !current.replaced) {
      const window = this.baseWindow + field.appended + (provisional ? provisional.appended : "");
      const stable = withoutTrailingThinkTagPrefix(window);
      const resync = !stable.startsWith(this.stableWindow);
      result = {
        changed: window !== this.renderedWindow,
        extends: window.startsWith(this.renderedWindow),
        // Rare: what was reported no longer holds, so the whole stable text
        // is reported again, as a replacement.
        delta: resync ? value.slice(0, this.windowStart) + stable : stable.slice(this.stableWindow.length),
        resync,
      };
      this.moveWindow(field, window, stable);
    } else {
      const stable = done ? value : withoutTrailingThinkTagPrefix(value);
      const previousStable = this.rendered.slice(0, this.stableLength);
      const resync = !stable.startsWith(previousStable);
      result = {
        changed: value !== this.rendered,
        extends: value.startsWith(this.rendered),
        delta: resync ? stable : stable.slice(previousStable.length),
        resync,
      };
      this.windowStart = 0;
      this.moveWindow(field, value, stable);
    }

    this.rendered = value;
    field.mark();
    return { value, ...result };
  }

  /** `window` and `stable` start at the current `windowStart`. */
  private moveWindow(field: Field, window: string, stable: string): void {
    const start = Math.max(0, field.value.length - WINDOW);
    const shift = start - this.windowStart;
    this.windowStart = start;
    this.baseWindow = field.tail.slice(field.tail.length - (field.value.length - start));
    this.renderedWindow = window.slice(shift);
    this.stableWindow = stable.slice(shift);
    this.stableLength = start + this.stableWindow.length;
  }
}

export interface StreamNormalizerUpdate {
  /** The normalized text so far, as `normalizeModelOutput` renders it. */
  text: string;
  /** The normalized reasoning so far, as `normalizeModelOutput` renders it. */
  reasoning: string;
  /**
   * What became stable since the previous update, to append to what the
   * previous deltas built. A trailing fragment that may still turn into a think
   * tag is withheld until it is decided. When `resync` is set for a field, its
   * delta is instead the whole stable value, replacing what was built.
   */
  delta: { text: string; reasoning: string };
  /** Whether `text` / `reasoning` differ from the previous update. */
  changed: { text: boolean; reasoning: boolean };
  /** Whether `text` starts with the text of the previous update. */
  textExtends: boolean;
  /**
   * Set for a field whose previously reported deltas no longer add up to it:
   * text already reported was withdrawn (a late think tag hid it, or the final
   * input differs from the chunks). That field's delta then replaces what the
   * deltas built instead of extending it: `view = resync ? delta : view + delta`.
   */
  resync: { text: boolean; reasoning: boolean };
}

export interface StreamNormalizer {
  /** Feeds the next chunk of raw text and/or dedicated reasoning. */
  push(chunk: { text?: string; reasoning?: string }): StreamNormalizerUpdate;
  /**
   * Ends the stream and releases anything withheld. Pass the complete final
   * text/reasoning when the provider reports them: they are authoritative over
   * the accumulated chunks.
   */
  finish(final?: { text?: string; reasoning?: string }): StreamNormalizerUpdate;
}

/**
 * Normalizes a streamed model output chunk by chunk, at a cost proportional to
 * the chunks rather than to the accumulated output. The strings it returns are
 * built by appending and are only copied if the caller reads them, so a
 * consumer that only uses `delta` stays linear over the whole stream.
 *
 * @example
 * const stream = createStreamNormalizer();
 * for await (const chunk of chunks) {
 *   const { delta } = stream.push({ text: chunk });
 *   process.stdout.write(delta.text);
 * }
 * process.stdout.write(stream.finish().delta.text);
 */
export function createStreamNormalizer(): StreamNormalizer {
  const state = new OutputState();
  const textReport = new FieldReport();
  const reasoningReport = new FieldReport();
  let rawText = "";
  let rawReasoning = "";
  let finished = false;

  const report = (): StreamNormalizerUpdate => {
    // The held-back fragments, rendered as if the stream ended here.
    let provisional: OutputState | undefined;
    if (state.textScanner.hasPending || state.reasoningScanner.hasPending) {
      provisional = state.clone();
      provisional.text.mark();
      provisional.reasoning.mark();
      provisional.textScanner.finish(provisional.textSink);
      provisional.reasoningScanner.finish(provisional.reasoningSink);
    }

    const text = textReport.update(state.text, provisional?.text, false);
    const reasoning = reasoningReport.update(state.reasoning, provisional?.reasoning, false);
    return {
      text: text.value,
      reasoning: reasoning.value,
      delta: { text: text.delta, reasoning: reasoning.delta },
      changed: { text: text.changed, reasoning: reasoning.changed },
      textExtends: text.extends,
      resync: { text: text.resync, reasoning: reasoning.resync },
    };
  };

  return {
    push(chunk) {
      if (finished) {
        throw new Error("The stream normalizer has already finished.");
      }
      if (chunk.text) {
        rawText += chunk.text;
        state.textScanner.feed(chunk.text, state.textSink);
      }
      if (chunk.reasoning) {
        rawReasoning += chunk.reasoning;
        state.reasoningScanner.feed(chunk.reasoning, state.reasoningSink);
      }
      return report();
    },

    finish(final) {
      finished = true;
      // The final render is the batch one, on the authoritative input: it is
      // computed once per stream, so its cost does not compound.
      const normalized = normalizeModelOutput(final?.text ?? rawText, final?.reasoning ?? rawReasoning);
      const textField = new Field();
      textField.replace(normalized.text);
      const reasoningField = new Field();
      reasoningField.replace(normalized.reasoning);
      const text = textReport.update(textField, undefined, true);
      const reasoning = reasoningReport.update(reasoningField, undefined, true);
      return {
        text: text.value,
        reasoning: reasoning.value,
        delta: { text: text.delta, reasoning: reasoning.delta },
        changed: { text: text.changed, reasoning: reasoning.changed },
        textExtends: text.extends,
        resync: { text: text.resync, reasoning: reasoning.resync },
      };
    },
  };
}
