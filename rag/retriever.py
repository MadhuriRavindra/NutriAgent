"""
rag/retriever.py
Runs a natural-language query against the ChromaDB recipe vector store,
combining semantic search (embedding similarity) with metadata filters
(cuisine, meal section, organic-safety).
"""

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Absolute path to <project root>/chroma_db, worked out from THIS file's
# location (rag/retriever.py -> parent = rag/ -> parent.parent = project root).
# A plain relative "chroma_db" depends on the folder the app was launched
# from, which differs between local runs and Streamlit Cloud.
PERSIST_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "recipes"


@lru_cache(maxsize=1)
def get_chroma_collection(persist_path: str = PERSIST_PATH,
                          name: str = COLLECTION_NAME):
    """
    Opens the persistent Chroma collection once and caches it, so every
    query reuses the same client instead of reopening the database.
    """
    client = chromadb.PersistentClient(path=persist_path)
    try:
        return client.get_collection(name)
    except Exception as e:
        existing = [c.name for c in client.list_collections()]
        raise RuntimeError(
            f"Chroma collection '{name}' not found at '{persist_path}'. "
            f"Collections present: {existing or 'none'}. "
            f"Run rag/vector_store.py to build it, or make sure chroma_db/ "
            f"is committed to the repo."
        ) from e


def build_where_filter(cuisine: str | None = None,
                       section: str | None = None,
                       organic_only: bool = False) -> dict | None:
    """
    Builds a ChromaDB 'where' clause from whichever filters are active.
    Each condition here is an EXACT match on pre-computed metadata -
    nothing gets re-analyzed at query time, it's all just reading fields
    that were set once during ingestion (organic_tagger.py, scraper.py).
    """
    conditions = []

    if cuisine:
        conditions.append({"cuisine": cuisine})

    if section:
        # Chroma metadata can't hold lists, and $contains only works on
        # document text - so each section was stored as its own boolean
        # field at ingestion (is_breakfast, is_dal, ...). Exact match here.
        conditions.append({f"is_{section}": True})

    if organic_only:
        conditions.append({"is_organic_safe": True})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search_recipes(query: str,
                   k: int = 3,
                   cuisine: str | None = None,
                   section: str | None = None,
                   organic_only: bool = False,
                   persist_path: str = PERSIST_PATH) -> list[dict]:
    """
    query        -> embedded and compared by meaning (semantic search)
    cuisine/section/organic_only -> applied as exact metadata filters,
                                     narrowing candidates BEFORE ranking
    Returns a clean list of dicts, ready for an agent/LLM to use.
    """
    collection = get_chroma_collection(persist_path)

    # Same embedding model as ingestion - critical. Query and stored
    # vectors must come from the identical model, or similarity scores
    # are meaningless (different models place text in different vector spaces).
    query_embedding = EMBED_MODEL.encode([query]).tolist()

    where_filter = build_where_filter(cuisine, section, organic_only)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where_filter,
    )

    # Reshape Chroma's nested list-of-lists response into a simple list
    # of recipe dicts - easier for the orchestrator/agent to consume.
    output = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "title": meta["title"],
            "cuisine": meta["cuisine"],
            "sections": meta["sections"],
            "source_url": meta["source_url"],
            "is_organic_safe": meta["is_organic_safe"],
            "flag_reasons": meta["flag_reasons"],
            "health_suggestions": meta["health_suggestions"].split(" || ") if meta["health_suggestions"] else [],
            "full_text": doc,
            "similarity_score": 1 - dist,  # Chroma returns distance; smaller = more similar
        })

    return output


if __name__ == "__main__":
    # Quick manual test - mirrors the interview example: "healthy south indian breakfast"
    print(f"Using Chroma at: {PERSIST_PATH}\n")
    results = search_recipes(
        query="healthy south indian breakfast",
        k=5,
        cuisine="South Indian",
        section="breakfast",
    )

    print(f"Found {len(results)} results:\n")
    for r in results:
        flag_note = "CLEAN" if r["is_organic_safe"] else f"FLAGGED ({r['flag_reasons']})"
        print(f"- {r['title']}")
        print(f"  score={r['similarity_score']:.3f} | {flag_note}")
        if r["health_suggestions"]:
            for s in r["health_suggestions"]:
                print(f"    -> {s}")
        print()