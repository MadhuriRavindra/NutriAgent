"""
rag/scraper.py
Scrapes vegetarian recipes from multiple Indian recipe sites, organized by category.
Dedupes across categories and sites (by source_url) and classifies cuisine at scrape time.
Output: data/raw_recipes_clean.json
"""

import json
import time
import random
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SITES = {
    "hebbarskitchen": {
        "base": "https://hebbarskitchen.com",
        "veg_only": True,
        "categories": {
            "breakfast":  "/recipes/breakfast-recipes/",
            "dal":        "/recipes/indian-dal-recipes/",
            "curry":      "/recipes/indian-curry-recipes/",
            "rice":       "/recipes/indian-rice-recipes/",
            "paneer":     "/recipes/top-paneer-recipes-paneer-curries/",
            "snacks":     "/recipes/indian-snacks-recipes/",
            "sweets":     "/recipes/indian-sweets-recipes/",
            "roti":       "/recipes/indian-roti-recipes/",
        },
        "non_recipe_prefixes": (
            "/recipes/", "/cuisines/", "/ingredients/", "/groups/", "/cook-method/",
            "/recipe-time/", "/difficulty/", "/season/", "/author/", "/page/",
            "/hi/", "/kn/", "/tag/", "/wprm_print/", "/contact-us/", "/privacy-policy/",
            "/feed/", "/hebbars-kitchen-recipes-videos/",
        ),
    },
    "vegrecipesofindia": {
        "base": "https://www.vegrecipesofindia.com",
        "veg_only": True,
        "categories": {
            "breakfast": "/recipes/indian-breakfast-recipes/",
            "curry":     "/recipes/indian-curry-recipes/",
            "dal":       "/recipes/dal-recipes-indian-curries/",
            "paneer":    "/recipes/paneer/",
            "snacks":    "/recipes/indian-snacks-indian-starters/",
            "sweets":    "/recipes/desserts-recipes/",
            "rice":      "/recipes/rice-recipes/",
            "vegetable": "/recipes/indian-vegetable-recipes/",
        },
        "non_recipe_prefixes": (
            "/recipes/", "/cuisines/", "/ingredients/", "/tag/", "/author/",
            "/page/", "/about/", "/dassana-cookbook/", "/press-media/",
            "/wprm_print/", "/feed/",
        ),
    },
    "cookwithmanali": {
        "base": "https://www.cookwithmanali.com",
        "veg_only": True,
        "categories": {
            "street_food": "/category/recipes/indian-street-food/",
            "main_course": "/category/recipes/indian-main-course-vegetarian-recipes/",
            "breads":      "/category/recipes/indian-breads/",
            "sauces":      "/category/recipes/sauces-chutneys-dips/",
            "snacks":      "/category/recipes/snacks-appetizers-sides/",
            "cakes":       "/category/recipes/eggless-cakes-cookies/",
        },
        "non_recipe_prefixes": (
            "/category/", "/tag/", "/author/", "/page/", "/about/",
            "/meetme/", "/contact/", "/wprm_print/", "/feed/",
            "/download-my-ebook/", "/recipe-index/",
        ),
    },
}

session = requests.Session()
session.headers.update(HEADERS)


# ---------- JSON-LD field normalization ----------

def as_text(value) -> str:
    """
    JSON-LD fields like 'name' or 'recipeCuisine' are allowed by schema.org
    to be either a plain string or a list of strings. Some sites (e.g. a
    Hebbar's Kitchen page with multiple cuisine tags) use the list form,
    which breaks any code that calls .strip() assuming a string. This
    normalizes either shape down to a single clean string.
    """
    if isinstance(value, list):
        return value[0].strip() if value and isinstance(value[0], str) else ""
    if isinstance(value, str):
        return value.strip()
    return ""


# ---------- Cuisine classification (all 3 tiers, applied at scrape time) ----------

# Tier 1: explicit region words mentioned directly in the title.
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

# Tier 2: dish-name / cooking-style keywords in the title.
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

# Tier 3: ingredient/instruction-level signals, used only when title gives
# no clear cue. Needs a clear margin (not just one stray match) to override.
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
MIN_SIGNAL_MARGIN = 2


def classify_cuisine(title: str, ingredients: list[str], instructions: list[str],
                      recipe_cuisine_from_page: str = "") -> str:
    """
    Cuisine classification, in priority order:
      0. Real recipeCuisine data from the page's JSON-LD, if present
      1. Direct region name in the title
      2. Dish/cooking-style keyword in the title
      3. Ingredient/instruction signal scoring (needs clear margin)
      4. Pan-Indian - honest fallback, not a wrong guess

    recipe_cuisine_from_page is expected to already be a clean string
    (normalized via as_text() by the caller).
    """
    if recipe_cuisine_from_page:
        return recipe_cuisine_from_page.title()

    title_lower = title.lower()

    for phrase, cuisine in REGION_DIRECT_MENTIONS.items():
        if phrase in title_lower:
            return cuisine

    for cuisine, keywords in DISH_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return cuisine

    full_text = " | ".join([title_lower] + ingredients + instructions).lower()
    south_score = sum(1 for s in SOUTH_INDIAN_SIGNALS if s in full_text)
    north_score = sum(1 for s in NORTH_INDIAN_SIGNALS if s in full_text)

    if south_score - north_score >= MIN_SIGNAL_MARGIN:
        return "South Indian"
    if north_score - south_score >= MIN_SIGNAL_MARGIN:
        return "North Indian"

    return "Pan-Indian"


# ---------- Networking helper ----------

def fetch_with_retry(url: str, max_retries: int = 3):
    for attempt in range(1, max_retries + 1):
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            wait = attempt * 3 + random.uniform(0, 2)
            print(f"    got {resp.status_code} on attempt {attempt}, waiting {wait:.1f}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    return None


# ---------- Step A: find recipe URLs on a category page ----------

def is_recipe_link(href: str, site: dict) -> bool:
    base = site["base"]
    if not href or not href.startswith(base):
        return False
    path = href[len(base):]
    if not path.startswith("/") or path == "/":
        return False
    if any(path.startswith(p) for p in site["non_recipe_prefixes"]):
        return False
    parts = [p for p in path.split("/") if p]
    return len(parts) == 1


def get_recipe_links_from_category(category_url: str, site: dict, max_pages: int = 2) -> list[str]:
    links = set()
    for page_num in range(1, max_pages + 1):
        page_url = category_url if page_num == 1 else f"{category_url.rstrip('/')}/page/{page_num}/"
        resp = fetch_with_retry(page_url)
        if resp is None:
            print(f"    page {page_num}: giving up after retries")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        before = len(links)
        for a in soup.find_all("a", href=True):
            if is_recipe_link(a["href"], site):
                links.add(a["href"])

        added = len(links) - before
        print(f"    page {page_num}: +{added} recipes (total {len(links)})")
        if added == 0:
            break

        time.sleep(random.uniform(2, 3.5))

    return sorted(links)


# ---------- Step B: scrape one recipe page via JSON-LD ----------

def extract_recipe_jsonld(soup: BeautifulSoup) -> dict | None:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except (TypeError, json.JSONDecodeError):
            continue

        candidates = data if isinstance(data, list) else data.get("@graph", [data])
        for item in candidates:
            if isinstance(item, dict) and item.get("@type") == "Recipe":
                return item
    return None


def scrape_recipe(url: str) -> dict:
    resp = fetch_with_retry(url)
    if resp is None:
        raise RuntimeError(f"failed to fetch {url} after retries")

    soup = BeautifulSoup(resp.text, "html.parser")
    recipe_ld = extract_recipe_jsonld(soup)
    if recipe_ld is None:
        raise ValueError(f"no Recipe JSON-LD found on {url}")

    title = as_text(recipe_ld.get("name", ""))

    raw_ingredients = recipe_ld.get("recipeIngredient", [])
    ingredients = [as_text(i) for i in raw_ingredients]
    ingredients = [i for i in ingredients if i]

    instructions_raw = recipe_ld.get("recipeInstructions", [])
    instructions = []
    for step in instructions_raw:
        if isinstance(step, dict):
            instructions.append(as_text(step.get("text", "")))
        else:
            instructions.append(as_text(step))
    instructions = [i for i in instructions if i]

    cuisine = classify_cuisine(
        title, ingredients, instructions,
        recipe_cuisine_from_page=as_text(recipe_ld.get("recipeCuisine", "")),
    )

    return {
        "source_url": url,
        "title": title,
        "cuisine": cuisine,
        "ingredients": ingredients,
        "instructions": instructions,
    }


# ---------- Step C: orchestrate across sites and categories, with live dedupe ----------

def build_dataset(max_pages_per_category: int = 2,
                   out_path: str = "data/raw_recipes_clean.json",
                   max_legit_sections: int = 3) -> None:
    """
    max_legit_sections: a real recipe reasonably belongs to at most a
    few categories (e.g. dosa -> breakfast + snacks). Anything crossing
    this threshold is almost certainly a sidebar "Popular Recipes" widget
    link repeating across every category page, not a genuine multi-category
    recipe, so we scrape it once but only keep its first-seen section.

    Dedup is by source_url only, so the same dish scraped from two
    different sites (e.g. "Dal Tadka" on both vegrecipesofindia.com and
    cookwithmanali.com) is kept as two separate entries, tagged with
    source_site, rather than silently merged.
    """
    seen = {}

    for site_name, site in SITES.items():
        for section, path in site["categories"].items():
            category_url = site["base"] + path
            print(f"\n[{site_name}/{section}] collecting recipe links...")
            recipe_urls = get_recipe_links_from_category(category_url, site, max_pages=max_pages_per_category)

            for i, recipe_url in enumerate(recipe_urls, 1):
                if recipe_url in seen:
                    current_sections = seen[recipe_url]["sections"]
                    if len(current_sections) < max_legit_sections and section not in current_sections:
                        current_sections.append(section)
                    print(f"  ({i}/{len(recipe_urls)}) already scraped: {section}")
                    continue

                try:
                    data = scrape_recipe(recipe_url)
                    data["sections"] = [section]
                    data["source_site"] = site_name
                    seen[recipe_url] = data
                    print(f"  ({i}/{len(recipe_urls)}) scraped: {data['title']}  [{data['cuisine']}]")
                except Exception as e:
                    print(f"  ({i}/{len(recipe_urls)}) FAILED: {recipe_url} — {e}")
                time.sleep(random.uniform(2, 3.5))

    all_recipes = list(seen.values())

    Path("data").mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_recipes, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_recipes)} unique recipes saved to {out_path}")


if __name__ == "__main__":
    build_dataset(max_pages_per_category=1)  # start small