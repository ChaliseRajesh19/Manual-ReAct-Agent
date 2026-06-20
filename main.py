# main.py
import os
from dotenv import load_dotenv
load_dotenv()  # MUST be first — before any langchain imports

from langchain_core.messages import HumanMessage,AIMessage
from graph.graph import build_graph

def run_agent(query: str):
    # Build the graph
    graph = build_graph()
    
    # Put user query into the notebook as a HumanMessage
    initial_state = {"messages": [HumanMessage(content=query)]}
    
    # Run the graph
    result = graph.invoke(initial_state,config={"recursion_limit": 4})


    
    # Last message in notebook = final answer
    final_answer = result["messages"][-1].content
    print("\n--- Final Answer ---")
    print(final_answer)

if __name__ == "__main__":
    while True:
        query = input("Ask ExamBuddy: ")
        run_agent(query)

        