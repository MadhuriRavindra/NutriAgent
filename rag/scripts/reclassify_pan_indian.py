"""
rag/reclassify_pan_indian.py
Re-examines recipes currently tagged 'Pan-Indian' and reclassifies them
as South Indian / North Indian when ingredients or instructions carry a
clear regional signal that the title alone didn't reveal.
Reads and overwrites: data/raw_recipes_clean.json
"""

import json

# Ingredient/instruction-level signals - checked across the FULL recipe
# text (ingredients + instructions), not just the title. These are things
# that reliably mark a dish as South or North Indian style cooking, even
# when the dish name itself is generic (e.g. "vegetable curry").
SOUTH_INDIAN_SIGNALS = [
    "curry leaves", "curry leaf", "coconut", "tamarind", "urad dal",
    "mustard seeds", "asafoetida", "hing", "rice flour", "toor dal",
    "gunpowder", "podi", "coconut oil", "sambar powder", "rasam powder",
    "idli", "dosa", "vada", "appalam", "papad roasted",
]

NORTH_INDIAN_SIGNALS = [
    "wheat flour", "atta", "paneer", "ghee", "garam masala", "cream",
    "kasuri methi", "cashew paste", "tandoor", "naan", "roti", "chapati",
    "rajma", "chole", "makhani", "malai", "dhaba",
]

# Minimum signal-count margin required before we override "Pan-Indian".
# A tie or a single stray match isn't enough - avoids over-classifying
# genuinely neutral recipes.
MIN_SIGNAL_MARGIN = 2


def score_signals(text: str, signal_list: list[str]) -> int:
    return sum(1 for signal in signal_list if signal in text)


def reclassify_recipe(recipe: dict) -> str:
    """
    Returns the recipe's cuisine - either its existing value (if not
    Pan-Indian, or if no clear signal is found) or a reclassified value.
    """
    if recipe.get("cuisine") != "Pan-Indian":
        return recipe["cuisine"]  # leave anything already classified alone

    full_text = " | ".join([
        recipe.get("title", ""),
        " ".join(recipe.get("ingredients", [])),
        " ".join(recipe.get("instructions", [])),
    ]).lower()

    south_score = score_signals(full_text, SOUTH_INDIAN_SIGNALS)
    north_score = score_signals(full_text, NORTH_INDIAN_SIGNALS)

    if south_score - north_score >= MIN_SIGNAL_MARGIN:
        return "South Indian"
    if north_score - south_score >= MIN_SIGNAL_MARGIN:
        return "North Indian"

    return "Pan-Indian"  # genuinely ambiguous or neutral - stays as is


def reclassify_dataset(path: str = "data/raw_recipes_clean.json") -> None:
    with open(path, encoding="utf-8") as f:
        recipes = json.load(f)

    before_counts = {}
    for r in recipes:
        before_counts[r["cuisine"]] = before_counts.get(r["cuisine"], 0) + 1

    changed = 0
    for r in recipes:
        old_cuisine = r["cuisine"]
        new_cuisine = reclassify_recipe(r)
        if new_cuisine != old_cuisine:
            r["cuisine"] = new_cuisine
            changed += 1

    after_counts = {}
    for r in recipes:
        after_counts[r["cuisine"]] = after_counts.get(r["cuisine"], 0) + 1

    print(f"Reclassified {changed} recipes out of {len(recipes)}\n")
    print("Before:", before_counts)
    print("After: ", after_counts)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated {path} in place")


if __name__ == "__main__":
    reclassify_dataset()