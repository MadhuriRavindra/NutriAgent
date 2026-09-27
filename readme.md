# 🥗 NutriRecipe Vegetarien

An AI-powered **vegetarian recipe assistant** for health-conscious people. Ask for a recipe in plain language ("a healthy South Indian breakfast", "a clean dal without refined sugar") and a LangGraph agent retrieves real recipes from a local vector store, flags unhealthy ingredients, and suggests healthier substitutes — without inventing recipes or nutrition numbers.

All recipes are strictly **vegetarian / vegan**.

---

## ✨ Features

- **Conversational recipe search** — natural-language chat powered by a LangGraph orchestrator agent with tool calling.
- **RAG over real recipes** — 640+ recipes scraped from Indian vegetarian recipe sites, embedded locally and searched semantically.
- **Health / organic flagging** — every recipe is scanned for artificial colour, MSG, synthetic essence, packaged/processed items, refined flour (maida) and refined sugar.
- **Healthier substitutes** — each flag comes with a suggested swap (e.g. jaggery/dates instead of sugar, whole wheat instead of maida) plus an honest note on how the swap changes texture or consistency.
- **Filters** — cuisine region (South Indian, North Indian, …), meal category, and organic-only.
- **Kids Recipe Finder** — kid-focused recipe suggestions.
- **Grounded answers** — the agent only uses retrieved data, says so when nothing matches, and does not estimate nutrition values it doesn't have.
- **Two interfaces** — a Streamlit chat UI and a FastAPI REST backend.

---

## 🏗️ Architecture

```
                ┌─────────────────────┐      ┌──────────────────┐
  User  ───────▶│  Streamlit (app.py) │      │ FastAPI (api.py) │◀─── REST clients
                └──────────┬──────────┘      └────────┬─────────┘
                           │   recipe_graph.invoke()  │
                           ▼                          ▼
                ┌─────────────────────────────────────────────┐
                │   LangGraph Orchestrator (agents/)          │
                │   Groq LLM  ⇄  find_recipes tool            │
                └──────────────────────┬──────────────────────┘
                                       ▼
                ┌─────────────────────────────────────────────┐
                │   rag/retriever.py                          │
                │   semantic search + metadata filters        │
                └──────────────────────┬──────────────────────┘
                                       ▼
                ┌─────────────────────────────────────────────┐
                │   ChromaDB (chroma_db/)                     │
                │   all-MiniLM-L6-v2 embeddings (local, CPU)  │
                └─────────────────────────────────────────────┘
```

### Offline data pipeline

```
rag/scraper.py  ──▶  rag/organic_tagger.py  ──▶  rag/vector_store.py  ──▶  rag/retriever.py
  (scrape +           (flag unhealthy            (embed + store            (query-time
   dedupe +            ingredients, add           in ChromaDB)              search)
   classify)           health suggestions)
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph, LangChain |
| LLM | Groq — `openai/gpt-oss-120b` |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, CPU) |
| Vector store | ChromaDB (persistent) |
| Scraping | requests + JSON-LD (schema.org/Recipe) parsing |
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn, Pydantic |
| YouTube agent (WIP) | Google ADK + LiteLLM (Groq) |
| Language | Python 3.13+ |

---

## 📁 Project Structure

```
.
├── agents/
│   ├── __init__.py
│   └── orchestrator.py        # LangGraph graph, find_recipes tool, system prompt
├── rag/
│   ├── scraper.py             # multi-site JSON-LD scraper
│   ├── organic_tagger.py      # health/organic flags + substitutions
│   ├── fix_section_overrides.py
│   ├── vector_store.py        # builds the ChromaDB collection
│   └── retriever.py           # search_recipes()
├── youtube_recipe_agent/      # Google ADK agent (in progress)
├── data/
│   └── raw_recipes_clean.json # scraped, de-duplicated recipes
├── chroma_db/                 # persistent vector store (generated)
├── .streamlit/config.toml
├── app.py                     # Streamlit UI
├── api.py                     # FastAPI backend
├── main.py                    # CLI chat loop
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd nutrirecipe-vegetarien
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

Install the **CPU-only** PyTorch build first — otherwise sentence-transformers pulls the full CUDA stack (several GB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Set your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

A free key is available at [console.groq.com](https://console.groq.com).

### 4. Build the recipe database

Run the pipeline once (or whenever you change scraping/tagging rules):

```bash
python rag/scraper.py          # → data/raw_recipes_clean.json
python rag/organic_tagger.py   # adds flags + health_suggestions
python rag/vector_store.py     # → chroma_db/
```

Optionally sanity-check retrieval:

```bash
python rag/retriever.py
```

### 5. Run the app

**Streamlit UI**

```bash
streamlit run app.py
```

> File watching is disabled in `.streamlit/config.toml` for faster startup, so restart Streamlit after code changes.

**FastAPI backend**

```bash
uvicorn api:app --reload
```

Then open the interactive docs at `http://localhost:8000/docs`.

**CLI**

```bash
python main.py
```

---

## 🔌 API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send a message with optional filters, get the agent's reply |

Example request:

```json
POST /chat
{
  "message": "Give me a healthy South Indian breakfast",
  "filters": {
    "cuisine": "South Indian",
    "meal_type": "breakfast",
    "organic_only": true
  }
}
```

**Meal categories** in the vector store: `breakfast`, `dal`, `curry`, `rice`, `paneer`, `snacks`, `sweets`, `roti`.

---

## 🔍 How Retrieval Works

1. The user query is embedded with the same `all-MiniLM-L6-v2` model used at indexing time.
2. ChromaDB returns the closest recipes by cosine similarity.
3. Exact-match metadata filters narrow the results:
   - `cuisine`
   - meal category — stored as one boolean per category (`is_breakfast`, `is_dal`, …) because Chroma's `$contains` only works on document text, not metadata lists
   - `is_organic_safe` for organic-only queries
4. The tool returns the full recipe text (ingredients + instructions) along with flags and health suggestions, and the LLM composes the answer.

### Cuisine classification

Assigned at scrape time using a 3-tier fallback:
1. the site's own `recipeCuisine` field
2. region/dish keywords in the title
3. ingredient-signal scoring

---

## 🌱 Data Sources

| Site | Status |
|---|---|
| hebbarskitchen.com | ✅ Scraped |
| vegrecipesofindia.com | ✅ Scraped |
| cookwithmanali.com | ✅ Scraped |
| indianhealthyrecipes.com | ⏳ Pending — needs a non-veg filter |
| archanaskitchen.com | ⏳ Pending — needs a non-veg filter |

Recipes are de-duplicated by source URL and tagged with `source_site`. Recipes with empty instructions are filtered out before tagging.

> Scraped content belongs to the respective recipe authors. This project is for personal learning; please respect each site's terms of use.

---

## 🧯 Known Issues

- Some vegrecipesofindia.com pages use nested `HowToSection` JSON-LD and currently come through without instructions (filtered out for now).
- Sidebar "popular recipes" links can occasionally add a wrong category to a recipe.
- No per-ingredient nutrition data exists yet, so the agent will not give calorie or macro numbers.

---

## 🗺️ Roadmap

- [x] RAG pipeline (scrape → tag → embed → retrieve)
- [x] LangGraph orchestrator with `find_recipes` tool
- [x] Streamlit chat UI with filters
- [x] FastAPI backend
- [x] Multi-source scraping (640+ recipes)
- [x] Kids Recipe Finder
- [ ] **User profile** in the sidebar (age, preferences) persisted to a database
- [ ] **Calorie calculation** per serving ("how many calories in 2 dosas?")
- [ ] **Food photo upload** → dish identification → calorie estimate
- [ ] **Weight-friendly suggestions** — filling, weight-maintaining meals
- [ ] YouTube description recipe extraction (Google ADK agent)
- [ ] More specialist agents: Meal Planner, Grocery, Nutrition, Meal Prep, Continental Food

---

## 📜 License

Personal learning project. Add a license of your choice (e.g. MIT) before publishing.