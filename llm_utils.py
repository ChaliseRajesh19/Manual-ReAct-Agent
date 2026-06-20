# llm_utils.py
from dotenv import load_dotenv
load_dotenv()  # safe to call multiple times — dotenv skips if already loaded

import os
from functools import lru_cache
from langchain_groq import ChatGroq



@lru_cache(maxsize=8)
def get_llm(model_name: str = None, temperature: float = 0.7, max_tokens: int = 4096):
    """
    Returns a cached ChatGroqAI LLM instance.
    max_tokens defaults to 4096 to avoid Groq AI context overflow.
    """
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

    if not model_name:
        raise ValueError(
            "MODEL_NAME is not set. Add MODEL_NAME=llama-3.1-8b-instant to your .env file."
        )

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=1,
    )