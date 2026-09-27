"""
rag/vector_store.py
Builds embeddings from data/raw_recipes_clean.json and stores them in a
persistent ChromaDB collection, ready for semantic search with metadata
filtering (cuisine, organic-safe, section) and health substitution tips.

Also exposes ensure_vector_store(), which the Streamlit app calls on
startup: if the collection is missing or empty (e.g. a fresh Streamlit
Cloud container), it builds it once from the JSON data.
"""

import json
from pathlib import Path

import chromadb

# Absolute paths worked out from THIS file's location
# (rag/vector_store.py -> parent = rag/ -> parent.parent = project root),
# so they work no matter which folder the app is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_PATH = str(PROJECT_ROOT / "chroma_db")
DATA_PATH = str(PROJECT_ROOT / "data" / "raw_recipes_clean.json")
COLLECTION_NAME = "recipes"

# Small, fast, local - no API key needed. First run downloads (~80MB), then cached.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

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


def build_vector_store(in_path: str = DATA_PATH,
                       persist_path: str = PERSIST_PATH) -> None:
    """
    Full rebuild: drops the old collection and re-embeds every recipe.

    Why drop instead of upsert? ids are 'recipe_<position>', so if a re-scrape
    produces FEWER recipes than last time, upsert would overwrite the first N
    and leave stale entries beyond N sitting in the store. Deleting first
    guarantees the store exactly matches the current JSON.
    """
    # Imported here, not at module top, so ensure_vector_store() can check
    # the store without loading the model when no build is needed.
    from sentence_transformers import SentenceTransformer

    with open(in_path, encoding="utf-8") as f:
        recipes = json.load(f)

    client = chromadb.PersistentClient(path=persist_path)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)

    texts, metadatas, ids = [], [], []
    for i, recipe in enumerate(recipes):
        texts.append(build_chunk_text(recipe))
        metadatas.append(build_metadata(recipe))
        ids.append(f"recipe_{i}")

    print(f"Embedding {len(texts)} recipes...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Stored {collection.count()} recipes in ChromaDB at '{persist_path}'")


def ensure_vector_store(persist_path: str = PERSIST_PATH) -> int:
    """
    Called by the app on startup. Builds the store only if the collection
    is missing or empty; otherwise does nothing. Returns the recipe count.
    """
    client = chromadb.PersistentClient(path=persist_path)
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing:
        count = client.get_collection(COLLECTION_NAME).count()
        if count > 0:
            return count

    print("Recipe vector store missing or empty - building it now...")
    build_vector_store(persist_path=persist_path)
    return chromadb.PersistentClient(path=persist_path).get_collection(COLLECTION_NAME).count()


if __name__ == "__main__":
    build_vector_store()