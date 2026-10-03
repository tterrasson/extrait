import type {
  DecisionAnswer,
  DecisionQuestion,
  DecisionQuestions,
  DecisionRequest,
  DecisionResult,
  LLMImageContent,
} from "../types";
import { buildHeaders, pickUsage, type OpenAICompatibleAdapterOptions } from "./openai-compatible-common";
import { buildURL, cleanUndefined, isRecord, pickString, readErrorBody, toFiniteNumber } from "./utils";

export const DEFAULT_DECISION_PATH = "/v1/systemone";

const MIN_SCORE_LEVELS = 2;
const MAX_SCORE_LEVELS = 10;

/** Calls llama.cpp's decision-model endpoint (`POST /v1/systemone`). */
export async function decideSystemOne(
  options: OpenAICompatibleAdapterOptions,
  fetcher: typeof fetch,
  path: string,
  request: DecisionRequest,
): Promise<DecisionResult> {
  const questions = buildQuestionsBody(request.questions);
  const body = cleanUndefined({
    ...options.defaultBody,
    ...request.body,
    model: request.model ?? options.model,
    state: request.state,
    images: request.images?.length ? request.images.map(toImageURL) : undefined,
    questions,
  });

  const response = await fetcher(buildURL(options.baseURL, path), {
    method: "POST",
    headers: buildHeaders(options),
    body: JSON.stringify(body),
    signal: request.signal,
  });

  if (!response.ok) {
    const message = await readErrorBody(response);
    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  const json: unknown = await response.json();
  if (!isRecord(json) || !isRecord(json.answers)) {
    throw new Error("Unexpected decision response: missing answers object");
  }

  const rawAnswers = json.answers;
  const answers = Object.fromEntries(
    Object.entries(request.questions).map(([key, question]) => [
      key,
      parseAnswer(key, question, rawAnswers[key]),
    ]),
  );

  return {
    answers,
    model: pickString(json.model) ?? (body.model as string),
    usage: pickUsage(json),
    raw: json,
  };
}

function buildQuestionsBody(questions: DecisionQuestions): Record<string, Record<string, unknown>> {
  const entries = Object.entries(questions);
  if (entries.length === 0) {
    throw new RangeError("decide() needs at least one question.");
  }

  return Object.fromEntries(entries.map(([key, question]) => [key, buildQuestionBody(key, question)]));
}

function buildQuestionBody(key: string, question: DecisionQuestion): Record<string, unknown> {
  switch (question.type) {
    case "choice": {
      // A list of names is shorthand for options without descriptions.
      const criteria = Array.isArray(question.criteria)
        ? Object.fromEntries(question.criteria.map((option) => [option, null]))
        : question.criteria;
      if (Object.keys(criteria).length === 0) {
        throw new RangeError(`Question "${key}": a choice needs at least one option.`);
      }
      return { type: "choice", instructions: question.instructions, criteria };
    }
    case "score": {
      const levels = question.criteria.length;
      if (levels < MIN_SCORE_LEVELS || levels > MAX_SCORE_LEVELS) {
        throw new RangeError(
          `Question "${key}": a score needs ${MIN_SCORE_LEVELS} to ${MAX_SCORE_LEVELS} levels, got ${levels}.`,
        );
      }
      return { type: "score", instructions: question.instructions, criteria: [...question.criteria] };
    }
    case "noul":
      return cleanUndefined({
        type: "noul",
        instructions: question.instructions,
        criteria: question.criteria,
      });
    default:
      throw new TypeError(
        `Question "${key}": unknown type "${(question as { type?: unknown }).type}" (expected choice, score or noul).`,
      );
  }
}

function toImageURL(image: string | LLMImageContent): string {
  return typeof image === "string" ? image : image.image_url.url;
}

function parseAnswer(key: string, question: DecisionQuestion, value: unknown): DecisionAnswer {
  if (!isRecord(value)) {
    throw new Error(`Unexpected decision response: no answer for question "${key}"`);
  }
  if (value.type !== question.type) {
    throw new Error(
      `Unexpected decision response: question "${key}" has type "${value.type}" (expected ${question.type})`,
    );
  }

  switch (question.type) {
    case "choice": {
      const options = Array.isArray(question.criteria)
        ? [...new Set(question.criteria)]
        : Object.keys(question.criteria);
      const choice = requireString(key, value, "choice");
      if (!options.includes(choice)) {
        throw new Error(`Unexpected decision response: question "${key}" has unknown choice "${choice}"`);
      }
      return {
        type: "choice",
        choice,
        probabilities: requireProbabilities(key, value.probabilities, options),
        confidence: requireBoundedNumber(key, value, "confidence", 1),
      };
    }
    case "score": {
      const levels = question.criteria.map((_, index) => String(index));
      requireKeys(key, "legend", value.legend, levels);
      const legend = Object.fromEntries(
        levels.map((level, index) => {
          const label = (value.legend as Record<string, unknown>)[level];
          if (label !== question.criteria[index]) {
            throw new Error(
              `Unexpected decision response: question "${key}" has an invalid legend for level "${level}"`,
            );
          }
          return [level, label as string];
        }),
      );
      return {
        type: "score",
        score: requireBoundedNumber(key, value, "score", levels.length - 1),
        legend,
        probabilities: requireProbabilities(key, value.probabilities, levels),
        confidence: requireBoundedNumber(key, value, "confidence", 1),
      };
    }
    case "noul":
      return { type: "noul", noul: requireBoundedNumber(key, value, "noul", 1) };
  }
}

function requireString(key: string, value: Record<string, unknown>, field: string): string {
  const result = value[field];
  if (typeof result !== "string") {
    throw new Error(`Unexpected decision response: question "${key}" has no "${field}"`);
  }
  return result;
}

function requireNumber(key: string, value: Record<string, unknown>, field: string): number {
  const result = toFiniteNumber(value[field]);
  if (result === undefined) {
    throw new Error(`Unexpected decision response: question "${key}" has no numeric "${field}"`);
  }
  return result;
}

function requireBoundedNumber(
  key: string,
  value: Record<string, unknown>,
  field: string,
  max: number,
): number {
  const number = requireNumber(key, value, field);
  if (number < 0 || number > max) {
    throw new Error(`Unexpected decision response: question "${key}" has "${field}" outside [0, ${max}]`);
  }
  return number;
}

function requireKeys(
  key: string,
  field: string,
  value: unknown,
  expected: readonly string[],
): asserts value is Record<string, unknown> {
  if (
    !isRecord(value) ||
    Object.keys(value).length !== expected.length ||
    expected.some((name) => !Object.hasOwn(value, name))
  ) {
    throw new Error(`Unexpected decision response: question "${key}" has invalid "${field}" keys`);
  }
}

function requireProbabilities(
  key: string,
  value: unknown,
  expected: readonly string[],
): Record<string, number> {
  requireKeys(key, "probabilities", value, expected);
  return Object.fromEntries(expected.map((name) => [name, requireBoundedNumber(key, value, name, 1)]));
}
