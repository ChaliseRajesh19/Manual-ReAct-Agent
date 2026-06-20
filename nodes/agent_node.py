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
llm_with_tools = llm.bind_tools(tools,tool_choice="auto")

SYSTEM_PROMPT = """You are ExamBuddy, an AI assistant for BCT students at Tribhuvan University, Nepal.

You have these tools:
- syllabus_fetcher: get BCT subject syllabus
- explaner_fetcher: explain a concept or topic
- planner_fetcher: create a study plan
- predictor_fetcher: predict likely exam questions
- question_fetcher: fetch past exam questions
- routine_fetcher: get exam schedule

Rules:
- For greetings and simple conversation, respond directly WITHOUT using any tool.
- For study-related questions, use the most relevant tool ONCE.
- After the tool returns results, output the FULL tool result word for word. Do not summarize, describe, or shorten it. Do not add suggestions or closing remarks.
- Never call the same tool twice in one conversation turn.
- Today's date: {date}
""".format(date=datetime.now().strftime("%Y-%m-%d"))

def agent_node(state: MessagesState):
    messages = state["messages"]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}