import os
from datetime import datetime
from langchain_core.tools import tool
from tools.explaner_tool import load_explaner_info
from tools.planner_tool import load_planner_info
from tools.predictor_tool import question_predictor
from tools.question_tool import load_question_info, load_index as load_question_index
from tools.syllabus_tool import load_syllabus_info, load_index as load_syllabus_index
from tools.routine_tool import get_routine_results

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
syllabus_index_path = os.path.join(BASE_DIR, "faiss_index", "syllabus_index")
question_index_path = os.path.join(BASE_DIR, "faiss_index", "past_papers_index")
syllabus_store = load_syllabus_index(syllabus_index_path)
question_store = load_question_index(question_index_path)


@tool
def syllabus_fetcher(query: str) -> str:
    """
    Retrieve syllabus information for a TU BCT subject.

    Returns the retrieved syllabus text.
    """

    docs_str = load_syllabus_info(syllabus_store, query)
    return docs_str


@tool
def explaner_fetcher(query: str) -> str:
    """Explain a BCT concept or topic in detail.You may have to use the another tool result as input to this tool.
    Input: topic name, e.g. 'Explain demand and supply in Engineering Economics'"""
    chain = load_explaner_info()
    return chain.invoke({"input": query})


@tool
def planner_fetcher(query: str) -> str:
    """Create a day-by-day study plan for a BCT subject before an exam.
    Input MUST include subject name and exam date, e.g. 'Engineering Economics exam on 2026-07-19'
    """
    chain, syllabus_info = load_planner_info(query)
    return chain.invoke(
        {
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "syllabus_info": syllabus_info,
            "chat_history": memory,
            "input": query,
        }
    )


@tool
def predictor_fetcher(query: str) -> str:
    """Predict likely exam questions for a BCT subject based on past patterns.
    Input: subject name, e.g. 'predict questions for Engineering Economics'"""
    return question_predictor(query)


@tool
def question_fetcher(query: str) -> str:
    """
    Retrieve relevant TU BCT past exam questions.

    Returns the retrieved question text from the database.
    Does not explain, summarize, or modify the retrieved content.
    """

    docs_str = load_question_info(question_store, query)
    return docs_str


@tool
def routine_fetcher(query: str) -> str:
    """Get the BCT exam routine/schedule.
    Input: 'exam routine' or a specific subject to find its exam date."""
    results = get_routine_results("give me the BCT exam routine")
    return results


tools = [
    syllabus_fetcher,
    explaner_fetcher,
    planner_fetcher,
    predictor_fetcher,
    question_fetcher,
    routine_fetcher,
]
