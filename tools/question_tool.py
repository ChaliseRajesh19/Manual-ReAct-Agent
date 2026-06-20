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
    docs_str = "\n\n".join([
        f"Document {i+1}:\n'{doc.page_content[:800].strip()}'"
        for i, doc in enumerate(relevant_docs)
    ])

    # ✅ Print trimmed preview (helpful during dev, remove in production)
    print(f"\nRelevant Documents:\n {docs_str}\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are ExamBuddy, a TU past paper assistant for BCT students at Tribhuvan University.

STRICT RULES:
1. Extract and display the ACTUAL questions from the context below — word for word.
2. Do NOT add disclaimers, warnings, or notes of any kind.
3. Do NOT say "for demonstration purposes" — these are real past papers.
4. If marks are shown in brackets like [4] or [4+3], include them.
5. Group questions by exam year and semester.

Output format:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 [Subject Name] — Past Questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📅 [Year] [Semester/Month]
**Q1.** [exact question text] [marks]
**Q2.** [exact question text] [marks]
...

## 📅 [Next Year] ...

Context:
{context}"""),
        ("human", "{input}"),
    ])

    llm = get_llm(model_name=MODEL_NAME, temperature=0.0, max_tokens=1024)
    chain = prompt | llm | StrOutputParser()

    return chain, docs_str