import streamlit as st
import uuid
from client import ask_exam_buddy

st.set_page_config(page_title="ExamBuddy", page_icon=":robot_face:", layout="centered")

st.title("ExamBuddy")
st.caption("Your AI-powered exam preparation assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask any things my buddy:"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_exam_buddy(prompt, st.session_state.session_id)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
