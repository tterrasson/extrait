import { describe, expect, test } from "bun:test";
import { createOpenAICompatibleAdapter } from "@/providers/openai-compatible";
import { createOpenAICompatibleLegacyAdapter } from "@/providers/openai-compatible-legacy";
import { createAnthropicCompatibleAdapter } from "@/providers/anthropic-compatible";
import { createLLM } from "@/llm";
import { images } from "@/image";
import type { DecisionState } from "@/index";

// Request and response taken from the llama.cpp decision-models announcement.
const QUESTIONS = {
  route: {
    type: "choice",
    instructions: "Which team should handle this?",
    criteria: {
      billing: "payments, charges, refunds, invoices",
      shipping: "delivery, tracking, lost or late parcels",
      technical: "bugs, errors, login problems",
    },
  },
  angry: {
    type: "noul",
    instructions: "Is the customer angry?",
  },
  urgency: {
    type: "score",
    instructions: "How urgent is this?",
    criteria: ["can wait", "this week", "today", "right now"],
  },
} as const;

const RESPONSE = {
  model: "ggml-org/Kev-4B-GGUF",
  answers: {
    route: {
      type: "choice",
      choice: "billing",
      probabilities: { billing: 0.9049, shipping: 0.0275, technical: 0.0676 },
      confidence: 0.8574,
    },
    angry: { type: "noul", noul: 0.8208 },
    urgency: {
      type: "score",
      score: 2.2821,
      legend: { "0": "can wait", "1": "this week", "2": "today", "3": "right now" },
      probabilities: { "0": 0.036, "1": 0.1937, "2": 0.2225, "3": 0.5478 },
      confidence: 0.2821,
    },
  },
  usage: { input_tokens: 130, output_tokens: 0 },
};

const STATE = "Customer message: I was charged twice for my order last week and nobody has replied.";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function capturingFetcher(payload: unknown = RESPONSE) {
  const calls: { url: string; init: RequestInit; body: Record<string, unknown> }[] = [];
  const fetcher = (async (url: string, init: RequestInit) => {
    calls.push({ url, init, body: JSON.parse(init.body as string) });
    return jsonResponse(payload);
  }) as unknown as typeof fetch;
  return { calls, fetcher };
}

function createClient(fetcher: typeof fetch, transport: Record<string, unknown> = {}) {
  return createLLM({
    provider: "openai-compatible",
    baseURL: "http://localhost:8080",
    model: "ggml-org/Kev-4B-GGUF",
    transport: { fetcher, ...transport },
  });
}

describe("LLMClient.decide()", () => {
  test("posts state and questions to /v1/systemone and returns typed answers", async () => {
    const { calls, fetcher } = capturingFetcher();
    const llm = createClient(fetcher);

    const result = await llm.decide(STATE, QUESTIONS);

    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe("http://localhost:8080/v1/systemone");
    expect(calls[0]!.init.method).toBe("POST");
    expect(calls[0]!.body).toEqual({
      model: "ggml-org/Kev-4B-GGUF",
      state: STATE,
      questions: QUESTIONS,
    });

    const route: "billing" | "shipping" | "technical" = result.answers.route.choice;
    expect(route).toBe("billing");
    expect(result.answers.route.probabilities.shipping).toBe(0.0275);
    expect(result.answers.route.confidence).toBe(0.8574);
    expect(result.answers.angry.noul).toBe(0.8208);
    expect(result.answers.urgency.score).toBe(2.2821);
    expect(result.answers.urgency.legend["3"]).toBe("right now");
    expect(result.answers.urgency.probabilities["3"]).toBe(0.5478);
    expect(result.model).toBe("ggml-org/Kev-4B-GGUF");
    expect(result.usage).toEqual({ inputTokens: 130, outputTokens: 0, totalTokens: undefined });
    expect(result.raw).toEqual(RESPONSE);
  });

  test("does not duplicate /v1 when the baseURL already ends with it", async () => {
    const { calls, fetcher } = capturingFetcher();
    const llm = createLLM({
      provider: "openai-compatible",
      baseURL: "http://localhost:8080/v1",
      model: "m",
      transport: { fetcher },
    });

    await llm.decide(STATE, QUESTIONS);

    expect(calls[0]!.url).toBe("http://localhost:8080/v1/systemone");
  });

  test("honors transport.decisionPath", async () => {
    const { calls, fetcher } = capturingFetcher();
    const llm = createClient(fetcher, { decisionPath: "/custom/decide" });

    await llm.decide(STATE, QUESTIONS);

    expect(calls[0]!.url).toBe("http://localhost:8080/custom/decide");
  });

  test("works with the legacy chat-completions adapter", async () => {
    const { calls, fetcher } = capturingFetcher();
    const adapter = createOpenAICompatibleLegacyAdapter({
      baseURL: "http://localhost:8080",
      model: "m",
      fetcher,
    });

    const result = await adapter.decide!({ state: STATE, questions: QUESTIONS });

    expect(calls[0]!.url).toBe("http://localhost:8080/v1/systemone");
    expect(result.answers.angry).toEqual({ type: "noul", noul: 0.8208 });
  });

  test("overrides the model per call (router mode)", async () => {
    const { calls, fetcher } = capturingFetcher({ ...RESPONSE, model: "ggml-org/Julia-1-GGUF:Q8_0" });
    const llm = createClient(fetcher);

    const result = await llm.decide(STATE, QUESTIONS, { model: "ggml-org/Julia-1-GGUF:Q8_0" });

    expect(calls[0]!.body.model).toBe("ggml-org/Julia-1-GGUF:Q8_0");
    expect(result.model).toBe("ggml-org/Julia-1-GGUF:Q8_0");
  });

  test("expands a list of choice options into options without descriptions", async () => {
    const { calls, fetcher } = capturingFetcher({
      model: "m",
      answers: {
        kind: {
          type: "choice",
          choice: "invoice",
          probabilities: { invoice: 0.7, receipt: 0.2, other: 0.1 },
          confidence: 0.5,
        },
      },
    });
    const llm = createClient(fetcher);

    const result = await llm.decide("A file uploaded by a customer.", {
      kind: {
        type: "choice",
        instructions: "What kind of document is this?",
        criteria: ["invoice", "receipt", "other"],
      },
    });

    expect(calls[0]!.body.questions).toEqual({
      kind: {
        type: "choice",
        instructions: "What kind of document is this?",
        criteria: { invoice: null, receipt: null, other: null },
      },
    });
    const kind: "invoice" | "receipt" | "other" = result.answers.kind.choice;
    expect(kind).toBe("invoice");
  });

  test("sends images as data URLs, accepting images() parts", async () => {
    const { calls, fetcher } = capturingFetcher({
      model: "m",
      answers: { blank: { type: "noul", noul: 0.1 } },
    });
    const llm = createClient(fetcher);
    const png = "data:image/png;base64,iVBORw0KGgo=";

    await llm.decide(
      "A file uploaded by a customer.",
      { blank: { type: "noul", instructions: "Is the page blank?" } },
      { images: [png, ...images({ base64: "AAAA", mimeType: "image/jpeg" })] },
    );

    expect(calls[0]!.body.images).toEqual([png, "data:image/jpeg;base64,AAAA"]);
  });

  test("passes chat messages through as state", async () => {
    const { calls, fetcher } = capturingFetcher({
      model: "m",
      answers: { angry: { type: "noul", noul: 0.9 } },
    });
    const llm = createClient(fetcher);
    const messages = [
      { role: "user" as const, content: "Where is my parcel?!" },
      { role: "assistant" as const, content: "Let me check." },
    ];

    await llm.decide(messages, { angry: { type: "noul", instructions: "Is the customer angry?" } });

    expect(calls[0]!.body.state).toEqual(messages);
    expect(calls[0]!.body.images).toBeUndefined();
  });

  test("merges defaultBody and per-call body, and forwards the signal", async () => {
    const { calls, fetcher } = capturingFetcher();
    const llm = createClient(fetcher, { defaultBody: { a: 1, b: 1 } });
    const controller = new AbortController();

    await llm.decide(STATE, QUESTIONS, { body: { b: 2 }, signal: controller.signal });

    expect(calls[0]!.body.a).toBe(1);
    expect(calls[0]!.body.b).toBe(2);
    expect(calls[0]!.init.signal).toBe(controller.signal);
  });

  const jsonStates: DecisionState[] = [
    { message: "Please refund my order", order: { id: 4471, paid: true, note: null }, tags: ["billing"] },
    [{ id: 1 }, { id: 2 }],
    false,
    0,
  ];
  test.each(jsonStates.map((state) => ({ state })))(
    "passes JSON state and noul descriptions through (%#)",
    async ({ state }) => {
      const { calls, fetcher } = capturingFetcher({ answers: { refund: { type: "noul", noul: 1 } } });
      const question = {
        type: "noul" as const,
        instructions: "Is a refund requested?",
        criteria: { true: "money back is asked", false: "no money back is asked" },
      };
      const result = await createClient(fetcher).decide(state, { refund: question });
      expect(calls[0]!.body.state).toEqual(state);
      expect(calls[0]!.body.questions).toEqual({ refund: question });
      expect(result.answers.refund.noul).toBe(1);
    },
  );

  test("preserves prototype-like question and option names", async () => {
    const probabilities = JSON.parse('{"__proto__":0.7,"constructor":0.3}');
    const answer = { type: "choice", choice: "__proto__", probabilities, confidence: 0.4 };
    const { calls, fetcher } = capturingFetcher({ answers: Object.fromEntries([["__proto__", answer]]) });
    const result = await createClient(fetcher).decide("state", {
      ["__proto__"]: { type: "choice", instructions: "?", criteria: ["__proto__", "constructor"] },
    });
    expect(result.answers.__proto__.probabilities.__proto__).toBe(0.7);
    expect(Object.hasOwn(result.answers.__proto__.probabilities, "__proto__")).toBe(true);
    expect(Object.getPrototypeOf(result.answers.__proto__.probabilities)).toBe(Object.prototype);
    expect((calls[0]!.body.questions as Record<string, unknown>).__proto__).toEqual({
      type: "choice",
      instructions: "?",
      criteria: Object.fromEntries([
        ["__proto__", null],
        ["constructor", null],
      ]),
    });
  });

  test.each([
    ["route", { ...RESPONSE.answers.route, type: "score" }, "expected choice"],
    ["route", { ...RESPONSE.answers.route, choice: "other" }, "unknown choice"],
    ["route", { ...RESPONSE.answers.route, probabilities: undefined }, 'invalid "probabilities" keys'],
    ["route", { ...RESPONSE.answers.route, probabilities: { billing: 1 } }, 'invalid "probabilities" keys'],
    [
      "route",
      { ...RESPONSE.answers.route, probabilities: { ...RESPONSE.answers.route.probabilities, other: 0 } },
      'invalid "probabilities" keys',
    ],
    [
      "route",
      { ...RESPONSE.answers.route, probabilities: { billing: "0.9", shipping: 0.03, technical: 0.07 } },
      'no numeric "billing"',
    ],
    [
      "route",
      { ...RESPONSE.answers.route, probabilities: { billing: -0.1, shipping: 0.1, technical: 1 } },
      '"billing" outside',
    ],
    [
      "route",
      { ...RESPONSE.answers.route, probabilities: { billing: 1.1, shipping: 0, technical: 0 } },
      '"billing" outside',
    ],
    ["route", { ...RESPONSE.answers.route, confidence: 42 }, '"confidence" outside'],
    ["route", { ...RESPONSE.answers.route, confidence: -1 }, '"confidence" outside'],
    ["angry", { type: "choice", noul: 0.5 }, "expected noul"],
    ["angry", { type: "noul", noul: -2 }, '"noul" outside'],
    ["angry", { type: "noul", noul: 2 }, '"noul" outside'],
    ["angry", { type: "noul", noul: NaN }, 'no numeric "noul"'],
    ["urgency", { ...RESPONSE.answers.urgency, type: "noul" }, "expected score"],
    ["urgency", { ...RESPONSE.answers.urgency, score: 99 }, '"score" outside'],
    ["urgency", { ...RESPONSE.answers.urgency, score: -1 }, '"score" outside'],
    ["urgency", { ...RESPONSE.answers.urgency, confidence: 2 }, '"confidence" outside'],
    ["urgency", { ...RESPONSE.answers.urgency, legend: undefined }, 'invalid "legend" keys'],
    ["urgency", { ...RESPONSE.answers.urgency, legend: { "0": "can wait" } }, 'invalid "legend" keys'],
    [
      "urgency",
      { ...RESPONSE.answers.urgency, legend: { ...RESPONSE.answers.urgency.legend, "0": "wrong" } },
      "invalid legend",
    ],
    ["urgency", { ...RESPONSE.answers.urgency, probabilities: {} }, 'invalid "probabilities" keys'],
  ] as const)("rejects malformed %s answers (%#)", async (key, answer, message) => {
    const payload = { ...RESPONSE, answers: { ...RESPONSE.answers, [key]: answer } };
    await expect(createClient(capturingFetcher(payload).fetcher).decide(STATE, QUESTIONS)).rejects.toThrow(
      message,
    );
  });

  test.each([[null], [[]], ["unexpected"], [{ answers: null }]])(
    "rejects invalid response envelopes (%#)",
    async (payload) => {
      await expect(createClient(capturingFetcher(payload).fetcher).decide(STATE, QUESTIONS)).rejects.toThrow(
        "missing answers object",
      );
    },
  );

  test("accepts endpoints of the probability and score ranges", async () => {
    const payload = {
      answers: {
        route: {
          ...RESPONSE.answers.route,
          probabilities: { billing: 1, shipping: 0, technical: 0 },
          confidence: 1,
        },
        angry: { type: "noul", noul: 0 },
        urgency: {
          ...RESPONSE.answers.urgency,
          score: 3,
          confidence: 0,
          probabilities: { "0": 0, "1": 0, "2": 0, "3": 1 },
        },
      },
    };
    const result = await createClient(capturingFetcher(payload).fetcher).decide(STATE, QUESTIONS);
    expect(result.answers.angry.noul).toBe(0);
    expect(result.answers.urgency.score).toBe(3);
  });

  test("sends the api key as a bearer token", async () => {
    const { calls, fetcher } = capturingFetcher();
    const llm = createLLM({
      provider: "openai-compatible",
      baseURL: "http://localhost:8080",
      model: "m",
      apiKey: "secret",
      transport: { fetcher },
    });

    await llm.decide(STATE, QUESTIONS);

    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBe("Bearer secret");
  });

  test("rejects invalid questions before sending anything", async () => {
    const { calls, fetcher } = capturingFetcher();
    const llm = createClient(fetcher);

    await expect(llm.decide(STATE, {})).rejects.toThrow("at least one question");
    await expect(
      llm.decide(STATE, { q: { type: "score", instructions: "?", criteria: ["only one"] } }),
    ).rejects.toThrow('Question "q": a score needs 2 to 10 levels, got 1.');
    await expect(
      llm.decide(STATE, {
        q: { type: "score", instructions: "?", criteria: Array.from({ length: 11 }, (_, i) => `l${i}`) },
      }),
    ).rejects.toThrow("got 11");
    await expect(
      llm.decide(STATE, { q: { type: "choice", instructions: "?", criteria: {} } }),
    ).rejects.toThrow("at least one option");
    await expect(llm.decide(STATE, { q: { type: "rank", instructions: "?" } } as never)).rejects.toThrow(
      'unknown type "rank"',
    );
    expect(calls).toHaveLength(0);
  });

  test("throws on non-ok responses with the error body", async () => {
    const fetcher = (async () =>
      new Response("model does not support decisions", { status: 400 })) as unknown as typeof fetch;
    const llm = createClient(fetcher);

    await expect(llm.decide(STATE, QUESTIONS)).rejects.toThrow("HTTP 400: model does not support decisions");
  });

  test("throws on malformed responses", async () => {
    await expect(
      createClient(capturingFetcher({ model: "m" }).fetcher).decide(STATE, QUESTIONS),
    ).rejects.toThrow("missing answers object");

    const { angry: _angry, ...partial } = RESPONSE.answers;
    await expect(
      createClient(capturingFetcher({ ...RESPONSE, answers: partial }).fetcher).decide(STATE, QUESTIONS),
    ).rejects.toThrow('no answer for question "angry"');

    await expect(
      createClient(
        capturingFetcher({ ...RESPONSE, answers: { ...RESPONSE.answers, angry: { type: "noul" } } }).fetcher,
      ).decide(STATE, QUESTIONS),
    ).rejects.toThrow('question "angry" has no numeric "noul"');
  });

  test("throws a descriptive error on providers without decision models", async () => {
    const llm = createLLM({
      provider: "anthropic-compatible",
      baseURL: "https://api.anthropic.com",
      model: "claude-haiku-4-5-20251001",
    });

    expect(
      createAnthropicCompatibleAdapter({ baseURL: "https://x.test", model: "m" }).decide,
    ).toBeUndefined();
    await expect(llm.decide(STATE, QUESTIONS)).rejects.toThrow(
      'Provider "anthropic-compatible" does not support decision models',
    );
  });
});
