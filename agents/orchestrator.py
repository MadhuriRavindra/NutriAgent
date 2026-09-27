from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from rag.retriever import search_recipes as rag_search_recipes

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _format_recipe_block(r: dict, full: bool) -> str:
    block = f"### {r['title']} ({r['cuisine']}, sections: {r['sections']})\n"
    if full:
        block += r["full_text"] + "\n"
    else:
        preview = r["full_text"][:400]
        if len(r["full_text"]) > 400:
            preview += "... (call get_recipe_full_text with this title for full instructions)"
        block += preview + "\n"
    if r["is_organic_safe"]:
        block += "Organic-safe: yes\n"
    else:
        block += f"Organic-safe: NO - flagged for: {r['flag_reasons']}\n"
        if r["health_suggestions"]:
            block += "Healthier swap: " + "; ".join(r["health_suggestions"]) + "\n"
    return block


@tool
def find_recipes(query: str = "",
                  meal_type: Optional[str] = None,
                  cuisine: Optional[str] = None,
                  organic_only: bool = False,
                  exclude_titles: Optional[list[str]] = None) -> str:
    """Search the recipe database for vegetarian recipes (the whole database is vegetarian,
    so there's no need to filter for that separately).

    query: natural-language description of what the user wants (e.g. 'light and low-oil',
    'something with paneer', 'quick snack'). Leave empty ("") if the user only specified
    filters (meal_type/cuisine) with no other description.

    meal_type: one of 'breakfast', 'dal', 'curry', 'rice', 'paneer', 'snacks', 'sweets', 'roti'.
    Leave as None if not specified.

    cuisine: one of 'South Indian', 'North Indian', 'Gujarati', 'Maharashtrian', 'Bengali',
    'Hyderabadi/Andhra', 'Kashmiri', 'Goan', 'Fusion/Continental', 'Pan-Indian'. Leave as None
    if not specified.

    organic_only: set True ONLY if the user wants a filtered LIST of clean recipes in general.
    Never set this True when the user is asking about one specific named recipe - it will
    EXCLUDE that recipe from results if it's flagged, instead of returning it with substitution
    advice. For a named recipe, always search with organic_only=False so the recipe itself
    (plus its health_suggestions, if flagged) comes back.

    exclude_titles: exact titles of recipes ALREADY SHOWN to the user earlier in this
    conversation. Pass this when the user asks for "more", "other than these", or "something
    different" so you don't repeat recipes they've already seen. Leave as None on a
    first/fresh search.

    Returns a PREVIEW only (title, cuisine, partial ingredients, organic/health flags) -
    NOT full instructions. If the user wants the complete recipe for one specific result,
    call get_recipe_full_text(title) with that recipe's exact title next."""
    # Fetch a larger pool when excluding, so there's enough left after filtering
    fetch_k = 10 if exclude_titles else 3
    results = rag_search_recipes(
        query=query if query else "vegetarian indian recipe",
        k=fetch_k,
        cuisine=cuisine,
        section=meal_type,
        organic_only=organic_only,
    )

    if exclude_titles:
        excluded = {t.lower() for t in exclude_titles}
        results = [r for r in results if r["title"].lower() not in excluded]

    results = results[:3]

    if not results:
        if exclude_titles:
            return ("No additional matching recipes found beyond what's already been shown - "
                     "the database doesn't have more recipes matching this filter.")
        return "No matching recipes found."

    return "\n---\n".join(_format_recipe_block(r, full=False) for r in results)


@tool
def get_recipe_full_text(title: str) -> str:
    """Fetch the COMPLETE ingredients + step-by-step instructions for ONE specific recipe
    by its exact title (as shown in a previous find_recipes result).

    Use this when the user asks for the full recipe, wants to actually cook something,
    or asks for instructions/steps for a specific dish that was already shown to them.
    Do NOT use this for browsing/searching - use find_recipes for that.

    title: the exact recipe title as it appeared in a find_recipes result."""
    results = rag_search_recipes(query=title, k=1)
    if not results:
        return f"Could not find a recipe titled '{title}'."
    return _format_recipe_block(results[0], full=True)


tools = [find_recipes, get_recipe_full_text]
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content="""You are a vegetarian recipe assistant with access to
find_recipes and get_recipe_full_text tools, backed by a real recipe database (scraped from
Hebbar's Kitchen).

Tool usage:
- find_recipes returns PREVIEWS only (title, cuisine, partial ingredients, organic/health
  flags) - NOT full step-by-step instructions. Use it for browsing, listing options, comparing
  recipes, or answering general questions about a recipe's cuisine/flags.
- If the user wants to actually cook something (asks for the full recipe, steps, or
  instructions for one specific dish), call get_recipe_full_text with that recipe's exact
  title (as shown in a previous find_recipes result) to get the complete ingredients +
  instructions. Do not attempt to answer a "how do I make this" question from a preview alone.
- If the user asks for "more", "other than these", or "something different" after you've
  already shown recipes for a similar filter, call find_recipes again with the SAME filters
  but pass exclude_titles containing the exact titles already shown earlier in this
  conversation. If it returns no additional results, tell the user plainly that the database
  doesn't have more matches for this filter - don't just re-list what was already shown.

Rules you must follow:
- Only describe recipes, ingredients, cuisine, and organic/health flags that actually came back
  from find_recipes or get_recipe_full_text results. Do not invent recipe names, ingredient
  lists, or instructions.
- You do NOT have access to nutrition data (protein, calories, carbs, etc.) for any recipe.
  If the user asks for a nutrition target (e.g. "100g protein a day", "low calorie meal plan"),
  clearly tell them you don't have nutrition data for these recipes and can't calculate this
  accurately - do not estimate or guess numbers.
- If find_recipes or get_recipe_full_text returns no results, say so plainly rather than
  making something up.

Handling "healthy version" / "make it healthier" / "replace the unhealthy ingredient" requests:
- If the user asks this about a SPECIFIC recipe that was already returned earlier in this
  conversation, do NOT call find_recipes again with organic_only=True for it - that filter
  EXCLUDES flagged recipes from results, it does not fetch substitution advice for them.
  Instead, reuse the flag_reasons and health_suggestions already present in that earlier
  find_recipes or get_recipe_full_text result (visible in the conversation history) and
  answer directly from that.
- If the recipe hasn't been searched yet, call find_recipes with organic_only left as False
  and the recipe name as the query, so the flagged recipe itself comes back along with its
  health_suggestions - then relay those substitutions to the user.
- Only use organic_only=True when the user wants a filtered LIST of clean recipes in general
  (e.g. "show me only organic recipes"), never when they're asking about one named recipe.

  When the user asks for a recipe "with healthy substitutes applied" or similar (not just "what
should I swap"), rewrite the ingredient list showing the substitution directly in place of the
flagged ingredient (e.g. "whole wheat flour (instead of maida)"), and always keep the consistency/
texture caveat from health_suggestions visible - don't drop it for brevity. Never claim a
substitution is a perfect 1:1 swap if the source data says otherwise.
""")


def orchestrator_node(state: AgentState):
    trimmed = trim_messages(state["messages"], keep_last=10)
    messages = [SYSTEM_PROMPT] + trimmed
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def trim_messages(messages, keep_last=10):
    # Keep only the most recent N messages (human/AI/tool turns).
    # Adjust keep_last based on testing - too low and the LLM loses
    # context of what it already showed the user this session.
    return messages[-keep_last:]


def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    tools_by_name = {t.name: t for t in tools}
    for tool_call in last_message.tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        tool_result = selected_tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
        )
    return {"messages": tool_messages}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("orchestrator")
graph.add_conditional_edges("orchestrator", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "orchestrator")

app = graph.compile()