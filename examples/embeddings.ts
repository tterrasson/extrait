/**
 * Embeddings Example
 *
 * Demonstrates:
 * - Generating vector embeddings with llm.embed()
 * - Embedding multiple texts in a single call
 * - Computing cosine similarity between embeddings
 *
 * Uses the same LLM_BASE_URL / LLM_API_KEY as other examples.
 * Set EMBED_MODEL to override the embedding model (e.g. "text-embedding-3-small").
 * For Anthropic users: set LLM_BASE_URL=https://api.voyageai.com and EMBED_MODEL=voyage-3.
 *
 * Usage: bun run dev embeddings [text1] [text2]
 * Example: bun run dev embeddings "the cat sat on the mat" "a feline rested on the rug"
 */

import { createLLM } from "@/index";

const embedder = createLLM({
  provider: "openai-compatible",
  model: process.env.EMBED_MODEL ?? process.env.LLM_MODEL ?? "text-embedding-3-small",
  transport: {
    baseURL: process.env.LLM_BASE_URL,
    apiKey: process.env.LLM_API_KEY,
  },
});

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, val, i) => sum + val * b[i]!, 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

const args = process.argv.slice(3);
const textA = args[0] ?? "the cat sat on the mat";
const textB = args[1] ?? "a feline rested on the rug";
const textC = args[2] ?? "the stock market closed higher today";

console.log("Texts to embed:");
console.log(`  A: "${textA}"`);
console.log(`  B: "${textB}"`);
console.log(`  C: "${textC}"`);
console.log();

const { embeddings, model, usage } = await embedder.embed([textA, textB, textC]);

console.log(`Model: ${model}`);
console.log(`Dimensions: ${embeddings[0]!.length}`);
console.log(`Tokens used: ${usage?.totalTokens ?? "n/a"}`);
console.log();

const simAB = cosineSimilarity(embeddings[0]!, embeddings[1]!);
const simAC = cosineSimilarity(embeddings[0]!, embeddings[2]!);

console.log(`Cosine similarity A↔B: ${simAB.toFixed(4)} (semantically related — expected HIGH)`);
console.log(`Cosine similarity A↔C: ${simAC.toFixed(4)} (unrelated topic — expected LOW)`);
console.log();

if (simAB > simAC) {
  console.log("✓ A and B are more similar to each other than A and C — as expected.");
} else {
  console.log("Unexpected: A↔C similarity is higher than A↔B.");
}
