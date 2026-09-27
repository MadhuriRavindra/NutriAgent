"""
rag/section_corrector.py
Fixes sections that got polluted by sidebar "Popular Recipes" widget links
during scraping (e.g. gulab jamun picking up breakfast/curry/dal from
widget appearances on those category pages, before the scraper ever
reached the real sweets category page - by which point max_legit_sections
had already been hit and the true section was silently dropped).

Approach: if a recipe's TITLE strongly matches a section's keyword list,
that's treated as ground truth and REPLACES whatever sections were
scraped - title-based section identity is far more reliable than
"which category page happened to link to it first".

Recipes whose titles don't match any keyword are left untouched.

Reads and updates: data/raw_recipes_clean.json
"""

import json

# Keyword -> section. Keep each list to strong, low-ambiguity dish names/
# words - a title match here OVERRIDES scraped sections, so false positives
# are costly. When a title matches keywords from more than one section,
# we skip it (ambiguous) rather than guess.
SECTION_KEYWORDS = {
    "sweets": [
        "gulab jamun", "jamun", "halwa", "kheer", "barfi", "burfi",
        "ladoo", "laddu", "rasgulla", "rasmalai", "peda", "jalebi",
        "payasam", "kulfi", "shrikhand", "mysore pak", "gujiya",
        "modak", "sandesh", "mishti", "basundi", "phirni",
    ],
    "breakfast": [
        "poha", "upma", "idli", "dosa", "uttapam", "paratha",
        "cheela", "pongal", "appe", "paddu", "thepla",
    ],
    "dal": [
        "dal tadka", "dal fry", "dal makhani", "sambar", "dal palak",
        "panchmel dal", "dal dhokli",
    ],
    "curry": [
        "curry", "sabzi", "sabji", "kadhi", "korma",
    ],
    "rice": [
        "pulao", "pulav", "biryani", "fried rice", "rice bath",
        "curd rice", "lemon rice", "tamarind rice",
    ],
    "snacks": [
        "pakora", "bhaji", "samosa", "chaat", "vada pav", "dhokla",
        "kachori", "bonda", "cutlet", "tikki",
    ],
    "roti": [
        "roti", "naan", "puri", "chapati", "kulcha", "bhatura",
    ],
}


def find_title_section_matches(title: str) -> list[str]:
    """Returns every section whose keyword list matches this title."""
    title_lower = title.lower()
    matches = []
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            matches.append(section)
    return matches


def correct_sections(path: str = "data/raw_recipes_clean.json") -> None:
    with open(path, encoding="utf-8") as f:
        recipes = json.load(f)

    corrected = 0
    skipped_ambiguous = 0

    for r in recipes:
        title_matches = find_title_section_matches(r["title"])

        if len(title_matches) == 0:
            continue  # no strong title signal, leave scraped sections as-is

        if len(title_matches) > 1:
            # title itself is ambiguous across sections (rare with this
            # keyword list, but possible) - don't guess, leave untouched
            skipped_ambiguous += 1
            continue

        correct_section = title_matches[0]
        current_sections = r.get("sections", [])

        if current_sections != [correct_section]:
            print(f"  {r['title']!r}: {current_sections} -> ['{correct_section}']")
            r["sections"] = [correct_section]
            corrected += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)

    print(f"\nCorrected {corrected} recipes' sections based on title keywords")
    print(f"Skipped {skipped_ambiguous} recipes with ambiguous title matches (left untouched)")
    print(f"Wrote {len(recipes)} recipes back to {path}")


if __name__ == "__main__":
    correct_sections()