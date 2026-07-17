import os
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.question_tool import load_index, load_question_info
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from llm_utils import get_llm

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME")


def question_predictor(query):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    question_index_dir = os.path.join(BASE_DIR, "faiss_index", "past_papers_index")
    question_store = load_index(question_index_dir)
    chain, docs_str = load_question_info(question_store, query)
    print("Relevant Documents:\n", docs_str)

    llm = get_llm(model_name=MODEL_NAME, temperature=0.7)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an Engineering Teacher who set the questions for the final board exam.Now generate the possible quesiton based on the provides information based on the question provided as context\n{context} and the output should be in the form of question only without any explanation and the question should be related to the provided context and the question should be of the same type as the provided question",
            ),
            ("human", "{input}"),
        ]
    )

    question_chain = prompt | llm | StrOutputParser()
    predicted_question = question_chain.invoke({"context": docs_str, "input": query})
    return predicted_question
