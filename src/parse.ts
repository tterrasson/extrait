import { jsonrepair } from "jsonrepair";
import type { z } from "zod";
import { extractJsonCandidates } from "./extract";
import { sanitizeThink } from "./think";
import { toErrorMessage } from "./utils/common";
import type {
  CandidateDiagnostics,
  ExtractionParseHint,
  ParseLLMOutputOptions,
  ParseLLMOutputResult,
  ParseTraceEvent,
  PipelineError,
} from "./types";

export function parseLLMOutput<TSchema extends z.ZodTypeAny>(
  output: string,
  schema: TSchema,
  options: ParseLLMOutputOptions = {},
): ParseLLMOutputResult<z.infer<TSchema>> {
  const sanitized = sanitizeThink(output);

  const parseOptions = {
    repair: options.repair ?? true,
    maxCandidates: options.maxCandidates ?? 5,
    acceptArrays: options.acceptArrays ?? true,
    extraction: options.extraction,
    onTrace: options.onTrace,
  };

  const candidates = extractJsonCandidates(sanitized.visibleText, {
    maxCandidates: parseOptions.maxCandidates,
    acceptArrays: parseOptions.acceptArrays,
    allowRepairHints: parseOptions.repair,
    heuristics: parseOptions.extraction,
  });

  emitTrace(parseOptions.onTrace, {
    stage: "extract",
    level: "info",
    message: `Extracted ${candidates.length} candidate(s).`,
    details: {
      maxCandidates: parseOptions.maxCandidates,
      thinkBlocks: sanitized.thinkBlocks.length,
      thinkDiagnostics: sanitized.diagnostics,
    },
  });

  const errors: PipelineError[] = [];
  const diagnostics: CandidateDiagnostics[] = [];
  let bestIssues: z.ZodIssue[] = [];
  let bestCandidate = candidates[0] ?? null;
  let bestParsed: unknown | null = null;
  let bestRepaired: string | null = null;

  for (const candidate of candidates) {
    const parseAttempt =
      parseAttemptFromHint(candidate.parseHint, parseOptions.repair) ??
      tryParseJsonCandidate(candidate.content, parseOptions.repair);

    if (!parseAttempt.success) {
      const diagnostic: CandidateDiagnostics = {
        candidateId: candidate.id,
        source: candidate.source,
        usedRepair: parseAttempt.usedRepair,
        parseSuccess: false,
        validationSuccess: false,
        selected: false,
        stage: parseAttempt.stage,
        message: parseAttempt.error,
      };
      diagnostics.push(diagnostic);

      errors.push({
        stage: parseAttempt.stage,
        message: parseAttempt.error,
        candidateId: candidate.id,
      });

      emitTrace(parseOptions.onTrace, {
        stage: parseAttempt.stage,
        level: "error",
        message: parseAttempt.error,
        candidateId: candidate.id,
      });

      continue;
    }

    emitTrace(parseOptions.onTrace, {
      stage: "parse",
      level: "info",
      message: parseAttempt.usedRepair
        ? "Candidate parsed after repair."
        : "Candidate parsed without repair.",
      candidateId: candidate.id,
      details: {
        usedRepair: parseAttempt.usedRepair,
      },
    });

    const validated = schema.safeParse(parseAttempt.parsed);
    if (validated.success) {
      const selectedDiagnostic: CandidateDiagnostics = {
        candidateId: candidate.id,
        source: candidate.source,
        usedRepair: parseAttempt.usedRepair,
        parseSuccess: true,
        validationSuccess: true,
        selected: true,
        stage: "success",
      };
      diagnostics.push(selectedDiagnostic);

      emitTrace(parseOptions.onTrace, {
        stage: "result",
        level: "info",
        message: `Validation succeeded on candidate ${candidate.id}.`,
        candidateId: candidate.id,
      });

      return {
        success: true,
        data: validated.data,
        raw: output,
        sanitizedRaw: sanitized.visibleText,
        thinkBlocks: sanitized.thinkBlocks,
        thinkDiagnostics: sanitized.diagnostics,
        parsed: parseAttempt.parsed,
        candidate,
        repaired: parseAttempt.repaired,
        candidates,
        diagnostics,
        errors,
        zodIssues: [],
      };
    }

    const issues = validated.error.issues;
    const message = formatZodIssues(issues);
    const validationDiagnostic: CandidateDiagnostics = {
      candidateId: candidate.id,
      source: candidate.source,
      usedRepair: parseAttempt.usedRepair,
      parseSuccess: true,
      validationSuccess: false,
      selected: false,
      stage: "validate",
      message,
      zodIssues: issues,
    };
    diagnostics.push(validationDiagnostic);

    if (bestIssues.length === 0 || issues.length < bestIssues.length) {
      bestIssues = issues;
      bestCandidate = candidate;
      bestParsed = parseAttempt.parsed;
      bestRepaired = parseAttempt.repaired;
    }

    errors.push({
      stage: "validate",
      message,
      candidateId: candidate.id,
      details: issues,
    });

    emitTrace(parseOptions.onTrace, {
      stage: "validate",
      level: "error",
      message: `Validation failed on candidate ${candidate.id}.`,
      candidateId: candidate.id,
      details: {
        issuesCount: issues.length,
      },
    });
  }

  if (candidates.length === 0) {
    const message = "No JSON candidate was extracted.";
    errors.push({
      stage: "extract",
      message,
    });

    emitTrace(parseOptions.onTrace, {
      stage: "extract",
      level: "error",
      message,
    });
  }

  markSelectedDiagnostic(diagnostics, bestCandidate?.id ?? null);

  emitTrace(parseOptions.onTrace, {
    stage: "result",
    level: "error",
    message: "No candidate could be validated.",
    details: {
      candidateCount: candidates.length,
      errors: errors.length,
    },
  });

  return {
    success: false,
    data: null,
    raw: output,
    sanitizedRaw: sanitized.visibleText,
    thinkBlocks: sanitized.thinkBlocks,
    thinkDiagnostics: sanitized.diagnostics,
    parsed: bestParsed,
    candidate: bestCandidate,
    repaired: bestRepaired,
    candidates,
    diagnostics,
    errors,
    zodIssues: bestIssues,
  };
}

interface ParseAttempt {
  success: boolean;
  parsed: unknown | null;
  repaired: string | null;
  usedRepair: boolean;
  stage: "parse" | "repair";
  error: string;
}

function tryParseJsonCandidate(input: string, allowRepair: boolean): ParseAttempt {
  try {
    return {
      success: true,
      parsed: JSON.parse(input),
      repaired: null,
      usedRepair: false,
      stage: "parse",
      error: "",
    };
  } catch (directError) {
    if (!allowRepair) {
      return {
        success: false,
        parsed: null,
        repaired: null,
        usedRepair: false,
        stage: "parse",
        error: toErrorMessage(directError),
      };
    }
  }

  let repaired: string;
  try {
    repaired = jsonrepair(input);
  } catch (repairError) {
    return {
      success: false,
      parsed: null,
      repaired: null,
      usedRepair: true,
      stage: "repair",
      error: toErrorMessage(repairError),
    };
  }

  try {
    return {
      success: true,
      parsed: JSON.parse(repaired),
      repaired,
      usedRepair: true,
      stage: "parse",
      error: "",
    };
  } catch (parseError) {
    return {
      success: false,
      parsed: null,
      repaired,
      usedRepair: true,
      stage: "parse",
      error: toErrorMessage(parseError),
    };
  }
}

function parseAttemptFromHint(
  hint: ExtractionParseHint | undefined,
  allowRepair: boolean,
): ParseAttempt | null {
  if (!hint) {
    return null;
  }

  if (hint.success) {
    if (hint.usedRepair && !allowRepair) {
      return null;
    }

    return {
      success: true,
      parsed: hint.parsed,
      repaired: hint.repaired,
      usedRepair: hint.usedRepair,
      stage: "parse",
      error: "",
    };
  }

  if (hint.usedRepair) {
    if (!allowRepair) {
      return null;
    }

    return {
      success: false,
      parsed: null,
      repaired: hint.repaired,
      usedRepair: true,
      stage: hint.stage,
      error: hint.error,
    };
  }

  if (allowRepair) {
    return null;
  }

  return {
    success: false,
    parsed: null,
    repaired: null,
    usedRepair: false,
    stage: "parse",
    error: hint.error,
  };
}

function markSelectedDiagnostic(
  diagnostics: CandidateDiagnostics[],
  selectedCandidateId: string | null,
): void {
  if (!selectedCandidateId) {
    return;
  }

  for (const diagnostic of diagnostics) {
    if (diagnostic.candidateId === selectedCandidateId) {
      diagnostic.selected = true;
      return;
    }
  }
}

function emitTrace(
  onTrace: ParseLLMOutputOptions["onTrace"],
  event: ParseTraceEvent,
): void {
  onTrace?.(event);
}

export function formatZodIssues(issues: z.ZodIssue[]): string {
  if (issues.length === 0) {
    return "Validation failed without details.";
  }

  return issues
    .map((issue) => {
      const path = issue.path.length > 0 ? issue.path.join(".") : "<root>";
      return `${path}: ${issue.message}`;
    })
    .join("\n");
}
