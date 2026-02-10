import type { StructuredPromptContext, StructuredPromptPayload, StructuredPromptResolver } from "./types";
import { createOutdent } from "./outdent";

function toPromptString(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }

  if (typeof value === "string") {
    return value;
  }

  if (
    typeof value === "number" ||
    typeof value === "boolean" ||
    typeof value === "bigint"
  ) {
    return String(value);
  }

  try {
    return JSON.stringify(value, null, 2) ?? "";
  } catch {
    return String(value);
  }
}

const dedent = createOutdent({
  trimLeadingNewline: true,
  trimTrailingNewline: true,
  newline: "\n",
});

function isBlankLine(line: string): boolean {
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char !== " " && char !== "\t") {
      return false;
    }
  }
  return true;
}

function stripOuterBlankLines(text: string): string {
  const lines = text.split("\n");
  let start = 0;
  while (start < lines.length && isBlankLine(lines[start] ?? "")) {
    start += 1;
  }

  let end = lines.length - 1;
  while (end >= start && isBlankLine(lines[end] ?? "")) {
    end -= 1;
  }

  if (start > end) {
    return "";
  }

  return lines.slice(start, end + 1).join("\n");
}

function renderPromptTemplate(strings: TemplateStringsArray, values: unknown[]): string {
  return stripOuterBlankLines(dedent(strings, ...values.map(toPromptString)));
}

function isTemplateStringsArray(value: unknown): value is TemplateStringsArray {
  return Array.isArray(value) && "raw" in value;
}

function toPromptMessage(input: string | TemplateStringsArray, values: unknown[]): string {
  if (typeof input === "string") {
    return input;
  }

  return renderPromptTemplate(input, values);
}

function joinMessages(messages: string[]): string {
  return messages.join("\n\n");
}

export interface PromptMessageBuilder extends StructuredPromptResolver {
  system(input: string): PromptMessageBuilder;
  system(strings: TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder;
  user(input: string): PromptMessageBuilder;
  user(strings: TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder;
  build(): StructuredPromptPayload;
}

class PromptMessageBuilderImpl implements PromptMessageBuilder {
  private readonly systemMessages: string[] = [];
  private readonly userMessages: string[] = [];

  system(input: string | TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder {
    const message = toPromptMessage(input, values);
    if (message.length > 0) {
      this.systemMessages.push(message);
    }
    return this;
  }

  user(input: string | TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder {
    const message = toPromptMessage(input, values);
    if (message.length > 0) {
      this.userMessages.push(message);
    }
    return this;
  }

  build(): StructuredPromptPayload {
    const prompt = joinMessages(this.userMessages);
    const systemPrompt = joinMessages(this.systemMessages);
    return {
      prompt,
      systemPrompt: systemPrompt.length > 0 ? systemPrompt : undefined,
    };
  }

  resolvePrompt(_context: StructuredPromptContext): StructuredPromptPayload {
    return this.build();
  }
}

function createPromptMessageBuilder(): PromptMessageBuilder {
  return new PromptMessageBuilderImpl();
}

export function prompt(strings: TemplateStringsArray, ...values: unknown[]): string;
export function prompt(): PromptMessageBuilder;
export function prompt(
  input?: TemplateStringsArray,
  ...values: unknown[]
): string | PromptMessageBuilder {
  if (isTemplateStringsArray(input)) {
    return renderPromptTemplate(input, values);
  }

  return createPromptMessageBuilder();
}
