import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _secret(name: str, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


OPENROUTER_API_KEY = _secret("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = _secret(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
MODEL_NAME = _secret("MODEL_NAME", "inclusionai/ling-3.2-flash:free")

GROQ_API_KEY = _secret("GROQ_API_KEY")
REWRITE_MODEL = _secret("REWRITE_MODEL", "openai/gpt-oss-20b")
CHAT_HISTORY_LIMIT = 6

TAVILY_API_KEY = _secret("TAVILY_API_KEY") or _secret("TVLY_API_KEY")

DATA_PATH = str(BASE_DIR / "data" / "chatbot_corpus.jsonl")
PERSIST_DIR = str(BASE_DIR / "chroma_db")
EMBEDDING_MODEL = _secret(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
DENSE_TOP_K = 5
SPARSE_TOP_K = 5
SPARSE_CANDIDATES = 50
HYBRID_WEIGHTS = [0.6, 0.4]
