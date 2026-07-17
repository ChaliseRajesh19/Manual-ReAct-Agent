# nodes/agent_node.py
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from tools.agent_tool import tools
from llm_utils import get_llm

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

# Get LLM and bind tools to it
llm = get_llm(model_name=MODEL_NAME, temperature=0.1, max_tokens=1024)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = f"""
You are ExamBuddy, an AI assistant for BCT students at Tribhuvan University, Nepal.

You have access to the following tools:
- syllabus_fetcher: retrieves syllabus information.
- explaner_fetcher: explains concepts.
- planner_fetcher: creates personalized study plans.
- predictor_fetcher: predicts likely exam questions.
- question_fetcher: retrieves past exam questions.
- routine_fetcher: retrieves exam schedules.

Rules:
1. For greetings or normal conversation, answer directly without using any tool.
2. For study-related requests, use the most appropriate tool.
3. When a tool returns information, use that information to answer the user naturally.
4. Never make up information that is not present in the tool output.
5. If the tool output already completely answers the user's request, present it clearly without calling another tool.
6. Never call the same tool more than once in a single turn.
7. If no tool is needed, answer directly.

Today's date: {datetime.now().strftime("%Y-%m-%d")}
"""


def agent_node(state: MessagesState):
    messages = state["messages"]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
