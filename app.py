from langchain_core.messages import AIMessage, HumanMessage
from graph.graph import build_graph

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("ExamBuddy 🎓")
st.caption("AI assistant for BCT students at Tribhuvan University")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []


if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
else:
    trimmed_history = st.session_state.conversation_history[-6:]
    st.session_state.conversation_history = trimmed_history

# Initialize graph in session state (build once, reuse)
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if query := st.chat_input("Ask ExamBuddy..."):
    # Show user message
    with st.chat_message("user"):
        st.write(query)

    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:

                message = st.session_state.conversation_history + [
                    HumanMessage(content=query)
                ]
                result = st.session_state.graph.invoke(
                    {"messages": message}, config={"recursion_limit": 10}
                )
                response = result["messages"][-1].content

                st.write(response)

                # Save assistant response
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                st.session_state.conversation_history.append(
                    HumanMessage(content=query)
                )
                st.session_state.conversation_history.append(
                    AIMessage(content=response)
                )

            except Exception as e:
                st.error(f"Error: {e}")
