export interface OutdentOptions {
  trimLeadingNewline?: boolean;
  trimTrailingNewline?: boolean;
  newline?: string | null;
}

export interface OutdentTag {
  (strings: TemplateStringsArray, ...values: unknown[]): string;
  string(input: string): string;
}

const DEFAULT_OPTIONS: Required<Pick<OutdentOptions, "trimLeadingNewline" | "trimTrailingNewline">> &
  Pick<OutdentOptions, "newline"> = {
  trimLeadingNewline: true,
  trimTrailingNewline: true,
  newline: null,
};

function isIndentChar(char: string): boolean {
  return char === " " || char === "\t";
}

function normalizeNewlines(text: string, newline: string): string {
  let result = "";
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (char === "\r") {
      if (text[index + 1] === "\n") {
        index += 1;
      }
      result += newline;
      continue;
    }
    if (char === "\n") {
      result += newline;
      continue;
    }
    result += char;
  }
  return result;
}

function detectIndentationFromFirstSegment(segment: string): number {
  for (let index = 0; index < segment.length; index += 1) {
    if (segment.charAt(index) !== "\n") {
      continue;
    }

    let cursor = index + 1;
    let indentation = 0;
    while (cursor < segment.length && isIndentChar(segment.charAt(cursor))) {
      indentation += 1;
      cursor += 1;
    }

    if (cursor === segment.length) {
      return indentation;
    }

    if (segment.charAt(cursor) !== "\n") {
      return indentation;
    }
  }

  return 0;
}

function removeIndentAfterNewlines(segment: string, indentation: number): string {
  if (indentation <= 0) {
    return segment;
  }

  let result = "";
  for (let index = 0; index < segment.length; index += 1) {
    const char = segment.charAt(index);
    result += char;

    if (char !== "\n") {
      continue;
    }

    let removed = 0;
    while (
      index + 1 < segment.length &&
      removed < indentation &&
      isIndentChar(segment.charAt(index + 1))
    ) {
      index += 1;
      removed += 1;
    }
  }

  return result;
}

function trimLeadingNewline(segment: string): string {
  let index = 0;
  while (index < segment.length && isIndentChar(segment.charAt(index))) {
    index += 1;
  }

  if (segment.charAt(index) === "\n") {
    return segment.slice(index + 1);
  }

  return segment;
}

function trimTrailingNewline(segment: string): string {
  let end = segment.length;
  while (end > 0 && isIndentChar(segment.charAt(end - 1))) {
    end -= 1;
  }

  if (end > 0 && segment.charAt(end - 1) === "\n") {
    return segment.slice(0, end - 1);
  }

  return segment;
}

function processSegments(
  inputSegments: ReadonlyArray<string>,
  options: Required<Pick<OutdentOptions, "trimLeadingNewline" | "trimTrailingNewline">> &
    Pick<OutdentOptions, "newline">,
): string[] {
  const segments = inputSegments.map((segment) => {
    if (typeof options.newline === "string") {
      return normalizeNewlines(segment, options.newline);
    }
    return segment;
  });

  const indentation = detectIndentationFromFirstSegment(segments[0] ?? "");
  const outdented = segments.map((segment) => removeIndentAfterNewlines(segment, indentation));

  if (options.trimLeadingNewline && outdented.length > 0) {
    outdented[0] = trimLeadingNewline(outdented[0] ?? "");
  }

  if (options.trimTrailingNewline && outdented.length > 0) {
    const lastIndex = outdented.length - 1;
    outdented[lastIndex] = trimTrailingNewline(outdented[lastIndex] ?? "");
  }

  return outdented;
}

function concatTemplate(strings: ReadonlyArray<string>, values: ReadonlyArray<unknown>): string {
  let output = "";
  for (let index = 0; index < strings.length; index += 1) {
    output += strings[index] ?? "";
    if (index < values.length) {
      output += String(values[index]);
    }
  }
  return output;
}

export function createOutdent(options: OutdentOptions = {}): OutdentTag {
  const resolvedOptions = {
    ...DEFAULT_OPTIONS,
    ...options,
  };

  const outdent = ((strings: TemplateStringsArray, ...values: unknown[]): string => {
    const processed = processSegments(strings, resolvedOptions);
    return concatTemplate(processed, values);
  }) as OutdentTag;

  outdent.string = (input: string): string => {
    return processSegments([input], resolvedOptions)[0] ?? "";
  };

  return outdent;
}
