"""
rag/dedupe_and_tag_cuisine.py
One-time post-processing on data/raw_recipes.json:
  - dedupes recipes that appeared under multiple categories (merges their sections)
  - adds a 'cuisine' field (South Indian / North Indian / Hyderabadi-Andhra /
    Gujarati / Maharashtrian / Bengali / Fusion-Continental / Pan-Indian)
Output: data/raw_recipes_clean.json
"""

import json

# Tier 1: explicit region words mentioned directly in the title.
# Checked FIRST because they're the most reliable signal.
REGION_DIRECT_MENTIONS = {
    "south indian": "South Indian",
    "north indian": "North Indian",
    "karnataka": "South Indian",
    "udupi": "South Indian",
    "tamil": "South Indian",
    "kerala": "South Indian",
    "andhra": "Hyderabadi/Andhra",
    "telangana": "Hyderabadi/Andhra",
    "hyderabadi": "Hyderabadi/Andhra",
    "punjabi": "North Indian",
    "kashmiri": "Kashmiri",
    "rajasthani": "North Indian",
    "gujarati": "Gujarati",
    "maharashtrian": "Maharashtrian",
    "bengali": "Bengali",
    "goan": "Goan",
}

# Tier 2: dish-name / cooking-style keywords, used when no direct region
# mention exists in the title.
DISH_KEYWORDS = {
    "South Indian": [
        "dosa", "idli", "sambar", "rasam", "vada", "uttapam", "appe",
        "paddu", "chutney", "upma", "pongal", "palkova", "pappu",
        "rice bath", "payasam",
    ],
    "North Indian": [
        "paratha", "chole", "rajma", "paneer", "naan", "kadhi",
        "dal makhani", "roti", "puri", "dhaba style", "tadka",
        "makhani", "chokha",
    ],
    "Gujarati": ["handvo", "dhokla", "thepla", "khandvi"],
    "Maharashtrian": ["poha", "misal", "sabudana", "puran poli"],
    "Bengali": ["luchi", "sandesh", "mishti"],
    "Fusion/Continental": [
        "cake", "pasta", "sandwich", "toast", "omelette", "mayonnaise",
        "choco bar", "chocobar", "pizza",
    ],
}


def classify_cuisine_fallback(title: str) -> str:
    """
    Guesses cuisine from the recipe title when the source page didn't
    supply recipeCuisine data. Checked in order:
      1. Direct region name mentioned in the title (most reliable)
      2. Dish/cooking-style keyword match
      3. Pan-Indian - honest fallback, not a wrong guess
    """
    title_lower = title.lower()

    for phrase, cuisine in REGION_DIRECT_MENTIONS.items():
        if phrase in title_lower:
            return cuisine

    for cuisine, keywords in DISH_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return cuisine

    return "Pan-Indian"


def dedupe_and_tag(in_path: str = "data/raw_recipes.json",
                    out_path: str = "data/raw_recipes_clean.json") -> None:
    with open(in_path, encoding="utf-8") as f:
        recipes = json.load(f)

    # Dedupe by source_url. When the same recipe appears again under a
    # different category page, merge the section into a 'sections' list
    # instead of keeping two separate entries.
    deduped = {}
    for r in recipes:
        url = r["source_url"]
        if url not in deduped:
            r["sections"] = [r.pop("section")]
            deduped[url] = r
        else:
            existing_section = r.get("section")
            if existing_section and existing_section not in deduped[url]["sections"]:
                deduped[url]["sections"].append(existing_section)

    result = list(deduped.values())

    # Add cuisine - prefer real data from the page (recipeCuisine via
    # JSON-LD, if scraper.py has been updated to capture it), fall back
    # to the keyword classifier otherwise.
    for r in result:
        cuisine = r.get("cuisine", "")
        if not cuisine or not cuisine.strip():
            cuisine = classify_cuisine_fallback(r["title"])
        r["cuisine"] = cuisine

    print(f"Before dedupe: {len(recipes)} entries")
    print(f"After dedupe:  {len(result)} unique recipes")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    dedupe_and_tag()