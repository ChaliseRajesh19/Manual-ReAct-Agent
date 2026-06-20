from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from tools.syllabus_tool import load_syllabus_info,load_index
from dotenv import load_dotenv
load_dotenv()
import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_utils import get_llm

MODEL_NAME = os.getenv("MODEL_NAME")

def load_planner_info(query):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(BASE_DIR, "faiss_index", "syllabus_index")
    syllabus_store = load_index(index_path)
    chain,syllabus_info = load_syllabus_info(syllabus_store, query)
    timess = datetime.now()


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful BCT exam assistant that creates detailed day-by-day study plans.
            Today's date is {current_date}.
            Use the following syllabus information to create the study plan:{syllabus_info}"""),
            ("human", "{input}"),
        ]
    )

    llm = get_llm(model_name=MODEL_NAME, temperature=0.7)
    
    planner_chain = prompt | llm | StrOutputParser()

    return planner_chain,syllabus_info


