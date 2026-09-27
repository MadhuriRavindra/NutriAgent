"""
NutriRecipe Vegetarien - FastAPI backend
Exposes the LangGraph orchestrator (agents/orchestrator.py) over HTTP so any
frontend (Streamlit, a JS app, mobile, etc.) can talk to it without importing
the graph directly into the same process.
"""

from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from agents.orchestrator import app as recipe_graph

api = FastAPI(title="NutriRecipe Vegetarien API")


# ---------------------------------------------------------------------------
# Request / response shapes.
# Pydantic validates incoming JSON against these automatically - a malformed
# request gets a 422 error before your code even runs.
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Filters(BaseModel):
    cuisine: Optional[str] = None
    meal_type: Optional[str] = None
    organic_only: bool = False


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    filters: Filters = Filters()


class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# The actual endpoint. Same logic as run_orchestrator() in app.py, just
# taking its inputs from an HTTP request body instead of Streamlit widgets.
# ---------------------------------------------------------------------------
@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    filter_hint = (
        f"[User filter preferences - apply when relevant: "
        f"cuisine={req.filters.cuisine or 'any'}, "
        f"meal_type={req.filters.meal_type or 'any'}, "
        f"organic_only={req.filters.organic_only}]"
    )
    augmented_message = f"{filter_hint}\n{req.message}"

    lc_history = []
    for m in req.history:
        if m.role == "user":
            lc_history.append(HumanMessage(content=m.content))
        else:
            lc_history.append(AIMessage(content=m.content))

    state = {
        "messages": lc_history + [HumanMessage(content=augmented_message)],
    }

    result = recipe_graph.invoke(state)
    reply = result["messages"][-1].content

    return ChatResponse(reply=reply)


# Simple health check - handy for confirming the server is up before
# wiring a frontend to it.
@api.get("/health")
def health():
    return {"status": "ok"}