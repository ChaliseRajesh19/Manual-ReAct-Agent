# llm_utils.py
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # safe to call multiple times — dotenv skips if already loaded


@lru_cache(maxsize=8)
def get_llm(
    model_name: str | None = None, temperature: float = 0.7, max_tokens: int = 4096
):
    """
    Returns a cached LLM instance for either Groq or Google Gemini.
    """
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

    if not model_name:
        raise ValueError("MODEL_NAME is not set. Add a model name to your .env file.")

    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in your .env file.")

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key
        )

    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in your .env file.")

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=1,
        api_key=api_key,
    )
