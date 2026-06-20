# Manual-ReAct-Agent
# Manual ReAct Agent (StateGraph)

A hand-built ReAct agent for BCT exam preparation, using LangGraph's `StateGraph` instead of the prebuilt `create_react_agent`.

## What this is

Project 3 in my GenAI learning roadmap. Project 2 used `create_react_agent` (a black box). This project rebuilds the same agent loop manually to understand how it actually works.

## How it works

- `MessagesState` holds the conversation history
- `agent_node` calls the LLM, which decides: call a tool, or answer directly
- `should_continue` checks the LLM's response and routes to `tools` or `END`
- `tools` (`ToolNode`) runs the requested tool, then always goes back to `agent_node`

Reuses the same 6 tools from Project 2 (syllabus, explainer, planner, predictor, past questions, exam routine).

## Tech stack

LangGraph, LangChain, Groq (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`), FAISS, Tavily, LangSmith.

## What I learned

- `MessagesState` replaces manual conversation memory — no separate memory object needed
- LangSmith tracing is essential for debugging agent loops (used it to find quota errors, message-trimming bugs, and prompt issues)
- System prompt wording can cause infinite tool-call loops if not careful

## Known limitation

Groq's Llama models occasionally write tool calls as plain text instead of using proper function-calling, especially in longer conversations. Found this via LangSmith but haven't fully fixed it yet.