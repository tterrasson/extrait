import type { z } from "zod";

export type StructuredMode = "loose" | "strict";

export interface ExtractionCandidate {
  id: string;
  source: "fenced" | "scan" | "raw";
  content: string;
  language?: string | null;
  parseHint?: ExtractionParseHint;
  start: number;
  end: number;
  score: number;
}

export interface ExtractionParseHint {
  success: boolean;
  parsed: unknown | null;
  repaired: string | null;
  usedRepair: boolean;
  stage: "parse" | "repair";
  error: string;
}

export interface ExtractionHeuristicsOptions {
  firstPassMin: number;
  firstPassCap: number;
  firstPassMultiplier: number;
  secondPassMin: number;
  secondPassCap: number;
  secondPassMultiplier: number;
  hintMaxLength: number;
}

export interface ExtractJsonCandidatesOptions {
  maxCandidates?: number;
  acceptArrays?: boolean;
  allowRepairHints?: boolean;
  heuristics?: Partial<ExtractionHeuristicsOptions>;
}

export interface ParseTraceEvent {
  stage: "extract" | "repair" | "parse" | "validate" | "result";
  level: "info" | "error";
  message: string;
  candidateId?: string;
  details?: unknown;
}

export interface ParseLLMOutputOptions {
  repair?: boolean;
  maxCandidates?: number;
  acceptArrays?: boolean;
  extraction?: Partial<ExtractionHeuristicsOptions>;
  onTrace?: (event: ParseTraceEvent) => void;
}

export interface PipelineError {
  stage: "extract" | "repair" | "parse" | "validate" | "llm" | "self-heal";
  message: string;
  candidateId?: string;
  details?: unknown;
}

export interface CandidateDiagnostics {
  candidateId: string;
  source: ExtractionCandidate["source"];
  usedRepair: boolean;
  parseSuccess: boolean;
  validationSuccess: boolean;
  selected: boolean;
  stage: "repair" | "parse" | "validate" | "success";
  message?: string;
  zodIssues?: z.core.$ZodIssue[];
}

export interface ThinkBlock {
  id: string;
  content: string;
  raw: string;
  start: number;
  end: number;
}

export interface ThinkDiagnostics {
  unterminatedCount: number;
  nestedCount: number;
  hiddenChars: number;
}

export interface ParseLLMOutputResult<T> {
  success: boolean;
  data: T | null;
  raw: string;
  sanitizedRaw: string;
  thinkBlocks: ThinkBlock[];
  thinkDiagnostics: ThinkDiagnostics;
  parsed: unknown | null;
  candidate: ExtractionCandidate | null;
  repaired: string | null;
  candidates: ExtractionCandidate[];
  diagnostics: CandidateDiagnostics[];
  errors: PipelineError[];
  zodIssues: z.core.$ZodIssue[];
}
