# tools/question_tool.py
from dotenv import load_dotenv

load_dotenv()

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from embedding_utils import get_embeddings
from llm_utils import get_llm

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")


# ✅ Reuse the load_index from syllabus_tool to avoid duplication
def load_index(index_path: str):
    embeddings = get_embeddings()
    store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def load_question_info(question_store, query: str):
    relevant_docs = question_store.invoke(query)

    # ✅ Trim each doc to 800 chars (was 1200) to stay within token budget
    docs_str = "\n\n".join(
        [
            f"Document {i+1}:\n'{doc.page_content[:800].strip()}'"
            for i, doc in enumerate(relevant_docs)
        ]
    )

    return docs_str
