#!/usr/bin/env bun
/**
 * Example runner for extrait library
 * Usage: bun run examples/runner.ts [example-name]
 * Default: simple
 */

const availableExamples = [
  "generate",
  "logprobs",
  "streaming",
  "streaming-with-tools",
  "streaming-turns-with-tools",
  "abort-signal",
  "timeout",
  "simple",
  "data-extraction",
  "sentiment-analysis",
  "multi-step-reasoning",
  "calculator-tool",
  "image-analysis",
  "conversation",
  "simulated-tools",
  "embeddings"
] as const;

const exampleName = process.argv[2] || "simple";

if (!availableExamples.includes(exampleName as any)) {
  console.error(`❌ Example "${exampleName}" not found.\n`);
  console.log("Available examples:");
  availableExamples.forEach((name) => {
    console.log(`  - ${name}`);
  });
  console.log(`\nUsage: bun run dev <example-name>`);
  console.log(`Example: bun run dev ${availableExamples[0]}`);
  process.exit(1);
}

console.log(`\n🚀 Running example: ${exampleName}\n`);

// Dynamically import and run the example
try {
  await import(`./${exampleName}.ts`);
} catch (error) {
  console.error(`\n❌ Failed to run example "${exampleName}":`, error);
  process.exit(1);
}
