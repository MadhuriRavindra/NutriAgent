"""
rag/merge_new_sources.py
Merges newly scraped recipes into data/raw_recipes_clean.json, deduped by
source_url. Existing recipes are kept in their original order and new ones
are appended at the end - this matters because vector_store.py assigns
Chroma IDs by list position (recipe_0, recipe_1, ...), so preserving order
keeps existing embeddings stable.
"""

import json

EXISTING_PATH = "data/raw_recipes_clean.json"
NEW_PATH = "data/new_sources_raw.json"  # update to your actual new-scrape output filename

def merge():
    with open(EXISTING_PATH, encoding="utf-8") as f:
        existing = json.load(f)

    with open(NEW_PATH, encoding="utf-8") as f:
        new_recipes = json.load(f)

    existing_urls = {r["source_url"] for r in existing}
    truly_new = [r for r in new_recipes if r["source_url"] not in existing_urls]

    print(f"Existing: {len(existing)}")
    print(f"Newly scraped: {len(new_recipes)}")
    print(f"Actually new (deduped by source_url): {len(truly_new)}")
    print(f"Skipped as duplicates: {len(new_recipes) - len(truly_new)}")

    merged = existing + truly_new  # order preserved - existing first, new appended

    with open(EXISTING_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(merged)} total recipes to {EXISTING_PATH}")

if __name__ == "__main__":
    merge()