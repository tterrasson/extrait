import type {
  LLMMessage,
  StructuredPromptContext,
  StructuredPromptPayload,
  StructuredPromptResolver
} from "./types";
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

export interface PromptMessageBuilder extends StructuredPromptResolver {
  system(input: string): PromptMessageBuilder;
  system(strings: TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder;
  user(input: string): PromptMessageBuilder;
  user(strings: TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder;
  assistant(input: string): PromptMessageBuilder;
  assistant(strings: TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder;
  build(): StructuredPromptPayload;
}

class PromptMessageBuilderImpl implements PromptMessageBuilder {
  private readonly messages: LLMMessage[] = [];

  system(input: string | TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder {
    return this.pushMessage("system", input, values);
  }

  user(input: string | TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder {
    return this.pushMessage("user", input, values);
  }

  assistant(input: string | TemplateStringsArray, ...values: unknown[]): PromptMessageBuilder {
    return this.pushMessage("assistant", input, values);
  }

  private pushMessage(
    role: LLMMessage["role"],
    input: string | TemplateStringsArray,
    values: unknown[],
  ): PromptMessageBuilder {
    const message = toPromptMessage(input, values);
    if (message.length > 0) {
      this.messages.push({ role, content: message });
    }
    return this;
  }

  build(): StructuredPromptPayload {
    return {
      messages: this.messages.map((message) => ({ ...message })),
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
