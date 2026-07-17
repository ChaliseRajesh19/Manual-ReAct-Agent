from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import sys
from functools import lru_cache

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = os.getenv("MODEL_NAME")


from llm_utils import get_llm


@lru_cache(maxsize=1)
def get_tavily_search():
    return TavilySearch(max_results=3)


def get_routine_results(query):
    tavily_search = get_tavily_search()
    specific_query = f"Tribhuvan University BCT {query} site:ioe.edu.np"
    results = tavily_search.invoke({"query": specific_query})
    return results
