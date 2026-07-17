from unittest import result

from langchain_core.messages import HumanMessage, AIMessage
from graph.graph import build_graph
import os
from dotenv import load_dotenv

load_dotenv()  # MUST be first — before any langchain imports

graph = build_graph()

session_memory = {}


def run_agent(query: str, session_id: str = None):

    if session_id not in session_memory:
        session_memory[session_id] = {"messages": []}

    memory = session_memory[session_id]

    # Add the new user message to this session's history
    memory["messages"].append(HumanMessage(content=query))

    # Build the dict the graph expects, using the FULL history
    initial_state = {"messages": memory["messages"]}

    # Run the graph
    result = graph.invoke(initial_state, config={"recursion_limit": 10})

    for i, msg in enumerate(result["messages"]):
        print("=" * 50)
        print(i)
        print(type(msg).__name__)
        print(msg)

    # Last message = final answer
    final_answer = result["messages"][-1].content

    # Save the agent's reply into history too
    memory["messages"].append(AIMessage(content=final_answer))

    return final_answer


if __name__ == "__main__":
    session_id = "abc123"  # You can change this to manage different sessions
    while True:
        query = input("Ask ExamBuddy: ")
        result = run_agent(query, session_id)
        print(f"ExamBuddy: {result}")
