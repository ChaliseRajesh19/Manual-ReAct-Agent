# graph/graph.py
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from nodes.agent_node import agent_node
from nodes.tool_node import agent_tool_node


def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_graph():
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", agent_tool_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

    graph.add_edge("tools", "agent")

    return graph.compile()
