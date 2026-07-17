from langchain_community.vectorstores import FAISS
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
import sys
import os
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_utils import get_embeddings
from llm_utils import get_llm



def load_index(index_path):
    embeddings = get_embeddings()
    
    store = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)
    syllabus_store = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return syllabus_store

def load_syllabus_info(syllabus_store, query):
    relevant_docs = syllabus_store.invoke(query)
    
    def format_docs(relevant_docs):
        docs_str = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:1200]}"
            for i, doc in enumerate(relevant_docs)
        ])
        return docs_str
    
    docs_str = format_docs(relevant_docs)
    return docs_str

