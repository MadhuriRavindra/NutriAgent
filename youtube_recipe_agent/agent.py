from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    model=LiteLlm(
        model='groq/qwen/qwen3.6-27b',
        reasoning_format='hidden',
    ),
    name='youtube_recipe_extractor',
    description='Extracts structured vegetarian recipes from raw YouTube video descriptions.',
    instruction="""
You extract vegetarian recipes from raw YouTube video descriptions.

Descriptions are messy: timestamps, sponsor mentions, hashtags, social
media links, and unrelated text are mixed in with the actual recipe
content.

Your job:
1. Find the recipe content (title, ingredients, instructions) if present.
2. Ignore timestamps, ads/sponsor blurbs, hashtags, subscribe/follow links.
3. If no real recipe is present in the text, say so explicitly - do not
   invent or guess a recipe.
4. Output ONLY valid JSON in this exact shape, nothing else - no preamble,
   no explanation, no reasoning, no markdown code fences. Your entire
   response must be parseable as JSON and nothing else:

{
  "recipe_found": true or false,
  "title": "string or null",
  "ingredients": ["list", "of", "ingredient", "strings"],
  "instructions": ["list", "of", "instruction", "steps"]
}
""",
)