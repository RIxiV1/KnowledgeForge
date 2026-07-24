import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import EMBEDDING_MODEL, CHROMA_PATH

_embed_kwargs = {"model": EMBEDDING_MODEL}
_base = os.getenv("OLLAMA_HOST")  # e.g. http://ollama:11434 in Docker
if _base:
    _embed_kwargs["base_url"] = _base
embeddings = OllamaEmbeddings(**_embed_kwargs)

def get_vectorstore():
    # Use cosine distance so relevance scores are normalized and comparable
    # across queries (needed for the relevance threshold in rag_engine).
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )