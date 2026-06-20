
import os
from datetime import datetime
from langchain_core.tools import tool
from tools.explaner_tool import load_explaner_info
from tools.planner_tool import load_planner_info
from tools.predictor_tool import question_predictor
from tools.question_tool import load_question_info, load_index as load_question_index
from tools.syllabus_tool import load_syllabus_info, load_index as load_syllabus_index
from tools.routine_tool import get_routine_results, routine_to_ai

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
syllabus_index_path = os.path.join(BASE_DIR, "faiss_index", "syllabus_index")
question_index_path = os.path.join(BASE_DIR, "faiss_index", "past_papers_index")


@tool
def syllabus_fetcher(query: str) -> str:
    """Fetch BCT syllabus information for a specific subject.
    Input: subject name and what you need, e.g. 'Engineering Economics topics'"""
    syllabus_store = load_syllabus_index(syllabus_index_path)
    chain, docs_str = load_syllabus_info(syllabus_store, query)
    return chain.invoke({
        "context": docs_str,
        "input": query
    })

@tool
def explaner_fetcher(query: str) -> str:
    """Explain a BCT concept or topic in detail.You may have to use the another tool result as input to this tool.
    Input: topic name, e.g. 'Explain demand and supply in Engineering Economics'"""
    chain = load_explaner_info()
    return chain.invoke({"input": query})

@tool
def planner_fetcher(query: str) -> str:
    """Create a day-by-day study plan for a BCT subject before an exam.
    Input MUST include subject name and exam date, e.g. 'Engineering Economics exam on 2026-07-19'"""
    chain, syllabus_info = load_planner_info(query)
    return chain.invoke({
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "syllabus_info": syllabus_info,
        "chat_history": memory,
        "input": query
    })

@tool
def predictor_fetcher(query: str) -> str:
    """Predict likely exam questions for a BCT subject based on past patterns.
    Input: subject name, e.g. 'predict questions for Engineering Economics'"""
    return question_predictor(query)

@tool
def question_fetcher(query: str) -> str:
    """Fetch past exam questions for a BCT subject from the database.
    Input: subject name, e.g. 'past questions of Engineering Economics'"""
    question_store = load_question_index(question_index_path)
    chain, docs_str = load_question_info(question_store, query)
    return chain.invoke({
        "context": docs_str,
        "input": query
    })

@tool
def routine_fetcher(query: str) -> str:
    """Get the BCT exam routine/schedule.
    Input: 'exam routine' or a specific subject to find its exam date."""
    results = get_routine_results("give me the BCT exam routine")
    print("Routine results:", results)
    chain, formatted_result = routine_to_ai(results)
    return chain.invoke({
        "formatted_result": formatted_result
    })

tools = [
    syllabus_fetcher,
    explaner_fetcher,
    planner_fetcher,
    predictor_fetcher,
    question_fetcher,
    routine_fetcher,
]