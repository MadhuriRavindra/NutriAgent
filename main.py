from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agents.orchestrator import app

load_dotenv()


def run_query(user_input: str):
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    result = app.invoke(initial_state)
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    print("NutriRecipe Vegetarien - ask me about vegetarian/vegan recipes!")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        answer = run_query(user_input)
        print(f"\nAssistant: {answer}\n")