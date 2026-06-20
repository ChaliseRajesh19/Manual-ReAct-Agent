from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os 
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils import get_llm
MODEL_NAME = os.getenv("MODEL_NAME")

def load_explaner_info():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that helps students understand complex concepts by providing clear and concise explanations."),
            ("human", "{input}"),
        ]
    )

    llm = get_llm(model_name=MODEL_NAME, temperature=0.7)
    
    explaner_chain = prompt | llm | StrOutputParser()

    return explaner_chain



