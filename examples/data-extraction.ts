/**
 * Data Extraction Example
 *
 * Demonstrates:
 * - Complex nested schemas with arrays and objects
 * - Self-healing for validation errors
 * - Debug mode to see repair attempts
 * - Extracting structured data from unstructured text
 *
 * Usage: bun run dev data-extraction
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
    selfHeal: 2, // Allow up to 2 self-heal attempts
    debug: true, // Enable debug to see repair attempts
  },
});

// Complex nested schema for recipe extraction
const RecipeSchema = s.schema(
  "Recipe",
  z.object({
    title: s.string().min(1).describe("The recipe title"),
    description: s.string().optional().describe("Brief description of the dish"),
    prepTime: s.number().min(0).describe("Preparation time in minutes"),
    cookTime: s.number().min(0).describe("Cooking time in minutes"),
    servings: s.number().int().min(1).describe("Number of servings"),
    difficulty: s.string().describe("Difficulty level: easy, medium, or hard"),
    ingredients: s
      .array(
        z.object({
          item: s.string().describe("Ingredient name"),
          quantity: s.string().describe("Amount needed (e.g., '2 cups', '1 tsp')"),
          optional: s.boolean().default(false).describe("Whether this ingredient is optional"),
        })
      )
      .min(1)
      .describe("List of ingredients"),
    steps: s
      .array(s.string().min(1))
      .min(1)
      .describe("Cooking steps in order"),
    tags: s.array(s.string()).default([]).describe("Recipe tags (e.g., 'vegan', 'gluten-free')"),
  })
);

// Unstructured recipe text
const recipeText = `
Classic Chocolate Chip Cookies

These soft and chewy cookies are a family favorite! Perfect for any occasion.

You'll need about 15 minutes to prep and 12 minutes to bake. Makes 24 cookies.
This is a medium difficulty recipe.

What you need:
- 2 1/4 cups all-purpose flour
- 1 teaspoon baking soda
- 1 teaspoon salt
- 1 cup (2 sticks) butter, softened
- 3/4 cup granulated sugar
- 3/4 cup packed brown sugar
- 2 large eggs
- 2 teaspoons vanilla extract
- 2 cups chocolate chips
- 1 cup chopped walnuts (if you like nuts)

How to make it:
1. Preheat your oven to 375°F (190°C).
2. Mix the flour, baking soda, and salt in a bowl.
3. In another bowl, beat the butter with both sugars until creamy.
4. Add eggs and vanilla to the butter mixture and beat well.
5. Gradually stir in the flour mixture.
6. Fold in the chocolate chips and walnuts.
7. Drop rounded tablespoons of dough onto ungreased baking sheets.
8. Bake for 9 to 11 minutes or until golden brown.
9. Cool on baking sheets for 2 minutes, then move to wire racks.

Tags: dessert, baking, chocolate, comfort-food
`;

console.log("📝 Extracting structured data from recipe text...\n");

try {
  const result = await llm.structured(
    RecipeSchema,
    prompt`
      Extract the recipe information from this text and return it as structured data.

      Text:
      """
      ${recipeText}
      """

      Important:
      - Convert time descriptions to numbers in minutes
      - Mark walnuts as optional since the text says "if you like nuts"
      - Set difficulty based on the description
      - Extract all ingredients with their quantities
      - List all cooking steps in order
    `,
    {
      // Self-healing is enabled in defaults, will automatically retry if validation fails
    }
  );

  console.log("\n✅ Successfully extracted recipe data!\n");
  console.log("Recipe:", result.data.title);
  console.log("Servings:", result.data.servings);
  console.log("Total time:", result.data.prepTime + result.data.cookTime, "minutes");
  console.log("Ingredients:", result.data.ingredients.length);
  console.log("Steps:", result.data.steps.length);
  console.log("\nFull structured data:");
  console.log(JSON.stringify(result.data, null, 2));
  console.log("\n📊 Usage:", result.usage ?? {});

  if (result.attempts && result.attempts.length > 1) {
    console.log(`\n🔄 Self-healing was used: ${result.attempts.length} attempts total`);
  }
} catch (error) {
  if (error instanceof StructuredParseError) {
    console.error("\n❌ Failed to extract recipe data after all attempts.");
    console.error("Attempt:", error.attempt);
    console.error("Zod validation issues:", error.zodIssues ?? []);
    console.error("\nRepair log:", error.repairLog ?? []);
    process.exit(1);
  }

  throw error;
}
