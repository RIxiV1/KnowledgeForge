from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import *

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_vectorstore():
    # Use cosine distance so relevance scores are normalized and comparable
    # across queries (needed for the relevance threshold in rag_engine).
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )