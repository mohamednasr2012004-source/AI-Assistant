from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config.config import (
    GROQ_API_KEY,
    MODEL_NAME,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REWRITE_MODEL,
)


def get_chat_model(temperature: float = 0) -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "AI Real Estate Assistant",
        },
    )


def get_rewrite_model(temperature: float = 0) -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=REWRITE_MODEL,
        temperature=temperature,
    )
