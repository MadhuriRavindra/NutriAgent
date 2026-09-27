"""
rag/vector_store.py
Builds embeddings from data/raw_recipes_clean.json and stores them in a
persistent ChromaDB collection, ready for semantic search with metadata
filtering (cuisine, organic-safe, section) and health substitution tips.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Small, fast, local - no API key needed. First run downloads (~80MB), then cached.
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

ALL_SECTIONS = ["breakfast", "dal", "curry", "rice", "paneer", "snacks", "sweets", "roti"]

def build_chunk_text(recipe: dict) -> str:
    """
    The text that actually gets embedded. Cuisine is included directly in
    the text (not just metadata) so semantic search itself is cuisine-aware.
    """
    ingredients_str = ", ".join(recipe.get("ingredients", []))
    instructions_str = " ".join(recipe.get("instructions", []))
    sections_str = ", ".join(recipe.get("sections", []))

    return (
        f"Recipe: {recipe['title']}\n"
        f"Cuisine: {recipe.get('cuisine', 'Pan-Indian')}\n"
        f"Meal sections: {sections_str}\n"
        f"Ingredients: {ingredients_str}\n"
        f"Instructions: {instructions_str}"
    )


def build_metadata(recipe: dict) -> dict:
    """
    Chroma metadata must be flat (str/int/float/bool only), so list fields
    get joined into strings. health_suggestions is included here (joined
    with ' || ' as a separator) so the agent can surface substitution tips
    when it returns a flagged recipe, without needing a second lookup.
    """
    sections = recipe.get("sections", [])

    metadata = {
        "title": recipe["title"],
        "source_url": recipe["source_url"],
        "sections": ", ".join(sections),  # kept for display purposes only
        "cuisine": recipe.get("cuisine", "Pan-Indian"),
        "is_organic_safe": recipe["is_organic_safe"],
        "num_flags": len(recipe.get("organic_flags", [])),
        "flag_reasons": ", ".join(f["reason"] for f in recipe.get("organic_flags", [])),
        "health_suggestions": " || ".join(recipe.get("health_suggestions", [])),
    }

    # One boolean field per section - this is what actually gets filtered on,
    # since Chroma's metadata filters only support exact match, not substring.
    for s in ALL_SECTIONS:
        metadata[f"is_{s}"] = s in sections

    return metadata


def get_chroma_collection(persist_path: str = "chroma_db", name: str = "recipes"):
    client = chromadb.PersistentClient(path=persist_path)
    return client.get_or_create_collection(name=name)


def build_vector_store(in_path: str = "data/raw_recipes_clean.json",
                        persist_path: str = "chroma_db") -> None:
    with open(in_path, encoding="utf-8") as f:
        recipes = json.load(f)

    collection = get_chroma_collection(persist_path)

    texts, metadatas, ids = [], [], []
    for i, recipe in enumerate(recipes):
        texts.append(build_chunk_text(recipe))
        metadatas.append(build_metadata(recipe))
        ids.append(f"recipe_{i}")

    print(f"Embedding {len(texts)} recipes...")
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=True).tolist()

    # upsert (not add) - safe to re-run after re-scraping/re-tagging
    # without duplicate entries, since ids are stable by list position.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Stored {collection.count()} recipes in ChromaDB at '{persist_path}'")


if __name__ == "__main__":
    build_vector_store()