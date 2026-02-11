import type { z } from "zod";
import { formatZodSchemaLikeTypeScript } from "./schema";

export interface WithFormatOptions {
  schemaInstruction?: string;
}

export const DEFAULT_SCHEMA_INSTRUCTION = "Strictly follow this JSON schema:";

export function withFormat(schema: z.ZodTypeAny, options: WithFormatOptions = {}): string {
  const schemaType = formatZodSchemaLikeTypeScript(schema);
  const instruction = resolveSchemaInstruction(options.schemaInstruction);

  return [instruction, schemaType].join("\n");
}

export function formatPrompt(schema: z.ZodTypeAny, task: string, options: WithFormatOptions = {}): string {
  const trimmedTask = task.trim();
  if (trimmedTask.length === 0) {
    return withFormat(schema, options);
  }

  return [withFormat(schema, options), "", trimmedTask].join("\n");
}

export function resolveSchemaInstruction(instruction: string | undefined): string {
  const trimmed = instruction?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : DEFAULT_SCHEMA_INSTRUCTION;
}
