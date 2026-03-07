/**
 * Multi-Step Reasoning Example
 *
 * Demonstrates:
 * - Chaining multiple structured LLM calls
 * - Passing data between calls
 * - Breaking down complex problems into steps
 * - Different schemas for different stages
 *
 * Usage: bun run dev multi-step-reasoning [your question]
 */

import { z } from "zod";
import { createLLM, prompt, s, StructuredParseError } from "@/index";

const provider = (process.env.LLM_PROVIDER ?? "openai-compatible") as
  | "openai-compatible"
  | "anthropic-compatible";

const model = process.env.LLM_MODEL ?? "gpt-5-nano";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;

const llm = createLLM({
  provider,
  model,
  transport: {
    baseURL,
    apiKey,
  },
  defaults: {
    mode: "loose",
    selfHeal: 1,
  },
});

// Step 1: Break down the question into sub-questions
const DecompositionSchema = s.schema(
  "QuestionDecomposition",
  z.object({
    subQuestions: s
      .array(
        z.object({
          id: s.number().int().describe("Sub-question ID"),
          question: s.string().describe("The sub-question to answer"),
          reasoning: s.string().describe("Why this sub-question is needed"),
        })
      )
      .min(1)
      .max(4)
      .describe("Break the question into 1-4 sub-questions"),
  })
);

// Step 2: Answer each sub-question
const AnswerSchema = s.schema(
  "SubAnswer",
  z.object({
    questionId: s.number().int().describe("ID of the question being answered"),
    answer: s.string().min(1).describe("The answer to this sub-question"),
    confidence: s.number().min(0).max(1).describe("Confidence in this answer (0-1)"),
  })
);

// Step 3: Synthesize final answer
const SynthesisSchema = s.schema(
  "FinalAnswer",
  z.object({
    answer: s.string().min(1).describe("The comprehensive final answer"),
    summary: s.string().describe("Brief summary of the reasoning process"),
    confidence: s.number().min(0).max(1).describe("Overall confidence in the final answer"),
  })
);

const userInput = process.argv.slice(3).join(" ").trim();
const question =
  userInput ||
  "How does photosynthesis work and why is it important for life on Earth?";

console.log("🧠 Multi-Step Reasoning Process\n");
console.log("Question:", question);
console.log("\n" + "━".repeat(60) + "\n");

try {
  // STEP 1: Decompose the question
  console.log("📋 Step 1: Breaking down the question...\n");

  const decomposition = await llm.structured(
    DecompositionSchema,
    prompt`
      Break down this complex question into simpler sub-questions that, when answered together,
      will provide a comprehensive answer to the main question.

      Question: """${question}"""

      Generate 2-4 sub-questions that cover all aspects needed to answer the main question.
    `
  );

  console.log("Sub-questions identified:");
  decomposition.data.subQuestions.forEach((sq) => {
    console.log(`  ${sq.id}. ${sq.question}`);
    console.log(`     → ${sq.reasoning}`);
  });

  console.log("\n" + "━".repeat(60) + "\n");

  // STEP 2: Answer each sub-question
  console.log("💡 Step 2: Answering each sub-question...\n");

  const answers: z.infer<typeof AnswerSchema>[] = [];
  for (const subQ of decomposition.data.subQuestions) {
    console.log(`Answering ${subQ.id}: ${subQ.question}`);

    const answer = await llm.structured(
      AnswerSchema,
      prompt`
        Answer this specific question concisely and accurately:

        Question: ${subQ.question}

        Context: This is part ${subQ.id} of answering the larger question: "${question}"

        Provide a clear, factual answer.
      `
    );

    answers.push(answer.data);
    console.log(`  ✓ Confidence: ${(answer.data.confidence * 100).toFixed(0)}%\n`);
  }

  console.log("━".repeat(60) + "\n");

  // STEP 3: Synthesize final answer
  console.log("🎯 Step 3: Synthesizing final answer...\n");

  const answersText = answers
    .map((a) => `Q${a.questionId}: ${a.answer}`)
    .join("\n\n");

  const synthesis = await llm.structured(
    SynthesisSchema,
    prompt`
      Based on these sub-question answers, provide a comprehensive final answer to the original question.

      Original question: """${question}"""

      Sub-question answers:
      ${answersText}

      Synthesize a complete, coherent answer that integrates all the information from the sub-answers.
    `
  );

  console.log("━".repeat(60));
  console.log("\n✅ FINAL ANSWER:\n");
  console.log(synthesis.data.answer);
  console.log(`\n📝 Summary: ${synthesis.data.summary}`);
  console.log(`📊 Confidence: ${(synthesis.data.confidence * 100).toFixed(0)}%`);
  console.log("\n" + "━".repeat(60) + "\n");

  // Show all sub-answers for reference
  console.log("📚 Detailed Sub-Answers:\n");
  decomposition.data.subQuestions.forEach((sq, idx) => {
    console.log(`${sq.id}. ${sq.question}`);
    console.log(`   ${answers[idx]?.answer ?? "No answer available"}\n`);
  });
} catch (error) {
  if (error instanceof StructuredParseError) {
    console.error("\n❌ Multi-step reasoning failed.");
    console.error("Stage:", error.attempt);
    console.error("Validation issues:", error.zodIssues ?? []);
    process.exit(1);
  }

  throw error;
}
