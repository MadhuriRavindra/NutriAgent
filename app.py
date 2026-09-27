"""
NutriRecipe Vegetarien - Streamlit frontend
Talks to the LangGraph orchestrator (agents/orchestrator.py) via the
run_orchestrator() wrapper below.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# 1. Import the COMPILED graph from orchestrator.py.
#    orchestrator.py's `app = graph.compile()` is the real entrypoint -
#    orchestrator_node is just one node inside it. We alias it to
#    recipe_graph so it doesn't collide with anything Streamlit-related.
# ---------------------------------------------------------------------------
from agents.orchestrator import app as recipe_graph


def run_orchestrator(user_message: str, chat_history: list, filters: dict) -> str:
    """
    Wraps a call to the compiled LangGraph app.

    user_message : the latest thing the user typed
    chat_history : list of {"role": "user"/"assistant", "content": str}
                    (everything said so far, oldest first - this is
                    Streamlit's storage format, we convert it below)
    filters      : {"cuisine": str|None, "meal_type": str|None, "organic_only": bool}

    Returns the assistant's reply as a plain string.
    """
    # Fold the sidebar filters into a short instruction so the LLM has
    # them as context when it decides how to call find_recipes.
    filter_hint = (
        f"[User filter preferences - apply when relevant: "
        f"cuisine={filters['cuisine'] or 'any'}, "
        f"meal_type={filters['meal_type'] or 'any'}, "
        f"organic_only={filters['organic_only']}]"
    )
    augmented_message = f"{filter_hint}\n{user_message}"

    # Convert Streamlit's plain-dict history into LangChain message objects,
    # since that's what AgentState / add_messages expects.
    lc_history = []
    for m in chat_history:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))

    state = {
        "messages": lc_history + [HumanMessage(content=augmented_message)],
    }

    # This runs the FULL graph: orchestrator -> (tools -> orchestrator)* -> END,
    # not just a single node - so tool calls are already resolved by the
    # time this returns.
    result = recipe_graph.invoke(state)

    # The last message is the final AIMessage once should_continue hits END.
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# 2. Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NutriRecipe Vegetarien", page_icon="🥗", layout="wide")

st.title("🥗 NutriRecipe Vegetarien")
st.caption("Vegetarian recipes, filtered for organic and clean-ingredient options.")

# ---------------------------------------------------------------------------
# 3. Session state - chat history and filters persist across reruns
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

# ---------------------------------------------------------------------------
# 4. Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    cuisine = st.selectbox(
        "Cuisine",
        options=[None, "South Indian", "North Indian", "Pan-Indian"],
        format_func=lambda x: "Any" if x is None else x,
    )

    meal_type = st.selectbox(
        "Meal type",
        options=[None, "breakfast", "dal", "curry", "rice", "paneer", "snacks", "sweets", "roti"],
        format_func=lambda x: "Any" if x is None else x.capitalize(),
    )

    organic_only = st.checkbox("Organic / clean ingredients only", value=False)

    filters = {"cuisine": cuisine, "meal_type": meal_type, "organic_only": organic_only}

    # Search button - builds a query straight from the selected filters and
    # queues it up to be sent, without the user needing to type anything.
    if st.button("🔍 Search", use_container_width=True):
        parts = []
        if filters["meal_type"]:
            parts.append(filters["meal_type"])
        if filters["cuisine"]:
            parts.append(filters["cuisine"])
        parts.append("recipes")
        if filters["organic_only"]:
            parts.append("(organic / clean ingredients only)")
        st.session_state.pending_query = "Show me " + " ".join(parts)

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# 5. Render existing chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# 6. Chat input - the main interaction loop
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask for a recipe...")

# A Search-button click sets pending_query on the PREVIOUS rerun; pop it here
# so it's only used once, and let it take priority over typed input in the
# unlikely case both happened at once.
pending_query = st.session_state.pop("pending_query", None)
effective_input = pending_query or user_input

if effective_input:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": effective_input})
    with st.chat_message("user"):
        st.markdown(effective_input)

    # Get and show the assistant's reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = run_orchestrator(
                effective_input, st.session_state.messages[:-1], filters
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})