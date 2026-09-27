"""
rag/organic_tagger.py
Scans each recipe's ingredients + instructions for non-organic / chemical
signals (artificial color, MSG, synthetic essence, packaged/processed items).
For each flag, also attaches a suggested healthier substitute.
Reads and updates: data/raw_recipes_clean.json
  (adds is_organic_safe, organic_flags, health_suggestions)
"""

import json
import re

# Keyword groups - matched against ingredients + instructions text (lowercased).
FLAG_RULES = {
    "artificial_color": [
        "food colour", "food color", "orange colour", "red colour",
        "yellow colour", "gel colour", "edible colour",
    ],
    "msg_ajinomoto": [
        "ajinomoto", "msg", "monosodium glutamate", "chinese salt",
    ],
    "synthetic_essence": [
        "essence",
    ],
    "packaged_processed": [
        "store bought", "store-bought", "readymade", "instant noodles",
        "maggi", "packet mix", "instant mix", "soft drink", "cola",
    ],
    "refined_flour": [
        "maida", "all purpose flour", "all-purpose flour", "plain flour",
    ],
    "refined_sugar": [
        "sugar", "powdered sugar", "castor sugar", "caster sugar",
        "icing sugar",
    ],
}

SUBSTITUTIONS = {
    "artificial_color": (
        "Skip the artificial colour, or use a natural alternative - "
        "saffron/kesar or turmeric for yellow-orange, beetroot juice for "
        "red-pink, spinach juice for green."
    ),
    "msg_ajinomoto": (
        "Drop the ajinomoto/MSG - a bit of extra garlic, white pepper, "
        "or a splash of soy sauce gives a similar savoury depth naturally."
    ),
    "synthetic_essence": (
        "Replace synthetic essence with real extract or the actual "
        "ingredient - e.g. real vanilla extract, or grated/crushed fruit "
        "instead of fruit essence."
    ),
    "packaged_processed": (
        "Swap the packaged/store-bought item for a homemade version where "
        "possible - e.g. homemade sauce or dough instead of a ready-made mix."
    ),
    "refined_flour": (
        "Maida (refined flour) can be replaced with whole wheat flour or "
        "a whole wheat + besan (gram flour) mix for more fibre and nutrients. "
        "Note: this WILL change the texture - whole wheat gives a denser, "
        "less airy/crisp result than maida, so expect a slightly heavier, "
        "less flaky or less puffed-up final dish."
    ),
    "refined_sugar": (
        "White sugar can be replaced with jaggery, honey, or dates paste "
        "for a less processed sweetener. Note: this WILL change the "
        "consistency and colour - jaggery/dates add moisture and a darker "
        "colour, so the batter/dough may need less added liquid, and the "
        "final texture may be softer or slightly denser than with sugar."
    ),
}


def tag_recipe_organic_status(recipe: dict) -> dict:
    ingredients_text = " | ".join(recipe.get("ingredients", [])).lower()
    instructions_text = " | ".join(recipe.get("instructions", [])).lower()
    full_text = ingredients_text + " | " + instructions_text

    flags = []
    reasons_seen = set()
    for reason, keywords in FLAG_RULES.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", full_text):
                flags.append({"reason": reason, "matched_keyword": kw})
                reasons_seen.add(reason)

    # One suggestion per distinct reason (not per keyword match, so a
    # recipe with 3 different "artificial_color" keyword hits still gets
    # ONE clear substitution tip, not three repeats of the same advice).
    suggestions = [SUBSTITUTIONS[reason] for reason in reasons_seen]

    recipe["is_organic_safe"] = len(flags) == 0
    recipe["organic_flags"] = flags
    recipe["health_suggestions"] = suggestions
    return recipe


def tag_dataset(path: str = "data/raw_recipes_clean.json") -> None:
    with open(path, encoding="utf-8") as f:
        recipes = json.load(f)

    for r in recipes:
        tag_recipe_organic_status(r)

    flagged = [r for r in recipes if not r["is_organic_safe"]]
    print(f"Tagged {len(recipes)} recipes")
    print(f"  clean (organic-safe): {len(recipes) - len(flagged)}")
    print(f"  flagged:              {len(flagged)}")

    if flagged:
        print("\nFlagged recipes and suggestions:")
        for r in flagged:
            reasons = ", ".join(f["reason"] for f in r["organic_flags"])
            print(f"\n  {r['title']}  [{reasons}]")
            for s in r["health_suggestions"]:
                print(f"    -> {s}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated {path} in place")


if __name__ == "__main__":
    tag_dataset()