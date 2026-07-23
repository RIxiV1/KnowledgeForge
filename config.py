import os

from dotenv import load_dotenv

# Load values from a local .env file if present (see .env.example).
load_dotenv()


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
DATABASE_PATH = os.getenv("DATABASE_PATH", "chat_history.db")
TOP_K = _int("RETRIEVAL_TOP_K", 20)
MAX_CHUNK_SIZE = _int("CHUNK_SIZE", 1200)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
MAX_CONTEXT_LENGTH = _int("MAX_CONTEXT_LENGTH", 15000)
CONVERSATION_HISTORY_LIMIT = _int("CONVERSATION_HISTORY_LIMIT", 5)
# Minimum semantic relevance (0-1) for a query to be considered answerable.
# If even the best-matching chunk scores below this, the app says it couldn't
# find the answer instead of forcing one from irrelevant context. Set to 0 to
# disable the gate. Tune per embedding model if you see false "not found"s.
RELEVANCE_THRESHOLD = _float("RELEVANCE_THRESHOLD", 0.15)
