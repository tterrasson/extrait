import type { LLMImageContent, LLMMessage, LLMUsage } from "./llm";

/**
 * Decision models (llama.cpp `/v1/systemone`) answer by scoring the options
 * you give them instead of generating text: one forward pass, one probability
 * per option, no output tokens.
 */

/** Pick one option. `criteria` maps each option to an optional description. */
export interface DecisionChoiceQuestion {
  type: "choice";
  instructions: string;
  /**
   * Options keyed by name, with an optional description (`null` for none).
   * A plain list of names is accepted as shorthand for options without descriptions.
   */
  criteria: Readonly<Record<string, string | null>> | readonly string[];
}

/** Rate on an ordered scale. `criteria` lists 2 to 10 level labels, lowest first. */
export interface DecisionScoreQuestion {
  type: "score";
  instructions: string;
  criteria: readonly string[];
}

/** Yes/no question. */
export interface DecisionNoulQuestion {
  type: "noul";
  instructions: string;
  /** Optional descriptions of the yes (`true`) and no (`false`) alternatives. */
  criteria?: Readonly<{ true?: string | null; false?: string | null }>;
}

export type DecisionQuestion = DecisionChoiceQuestion | DecisionScoreQuestion | DecisionNoulQuestion;

export type DecisionQuestions = Readonly<Record<string, DecisionQuestion>>;

export interface DecisionChoiceAnswer<TOption extends string = string> {
  type: "choice";
  /** The most likely option. */
  choice: TOption;
  probabilities: Record<TOption, number>;
  confidence: number;
}

export interface DecisionScoreAnswer {
  type: "score";
  /** Expected level: a fractional index into `legend` (0 = lowest level). */
  score: number;
  /** Level index (as a string) to its label. */
  legend: Record<string, string>;
  /** Level index (as a string) to its probability. */
  probabilities: Record<string, number>;
  confidence: number;
}

export interface DecisionNoulAnswer {
  type: "noul";
  /** Probability of "yes", between 0 and 1. */
  noul: number;
}

export type DecisionAnswer = DecisionChoiceAnswer | DecisionScoreAnswer | DecisionNoulAnswer;

type DecisionChoiceOptions<TCriteria> = TCriteria extends readonly (infer TOption extends string)[]
  ? TOption
  : Extract<keyof TCriteria, string>;

export type DecisionAnswerFor<TQuestion> = TQuestion extends { type: "choice"; criteria: infer TCriteria }
  ? DecisionChoiceAnswer<DecisionChoiceOptions<TCriteria>>
  : TQuestion extends { type: "score" }
    ? DecisionScoreAnswer
    : TQuestion extends { type: "noul" }
      ? DecisionNoulAnswer
      : DecisionAnswer;

/** Answers keyed like the questions, each typed after its question. */
export type DecisionAnswers<TQuestions extends DecisionQuestions = DecisionQuestions> = {
  [K in keyof TQuestions]: DecisionAnswerFor<TQuestions[K]>;
};

/** JSON content given to a decision model, serialized as text by the server. */
export type DecisionJSONValue = string | number | boolean | null
  | readonly DecisionJSONValue[]
  | { readonly [key: string]: DecisionJSONValue };

/** Non-null JSON content, or chat messages with optional inline images. */
export type DecisionState = Exclude<DecisionJSONValue, null> | readonly LLMMessage[];

export interface DecisionRequest<TQuestions extends DecisionQuestions = DecisionQuestions> {
  /** Plain text, JSON content, or chat messages (`image_url` data URLs are read as images). */
  state: DecisionState;
  questions: TQuestions;
  /** Images as data URLs, or the parts returned by `images()` / `loadImages()`. */
  images?: (string | LLMImageContent)[];
  /** Overrides the client model, e.g. to pick a model in llama.cpp router mode. */
  model?: string;
  body?: Record<string, unknown>;
  signal?: AbortSignal;
}

export interface DecisionResult<TQuestions extends DecisionQuestions = DecisionQuestions> {
  answers: DecisionAnswers<TQuestions>;
  model: string;
  usage?: LLMUsage;
  raw?: unknown;
}
