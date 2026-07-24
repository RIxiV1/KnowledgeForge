"""Retrieval tests that require live Ollama embeddings.

Every test here is marked ``ollama`` and is skipped automatically when no local
Ollama server is reachable (e.g. in CI). They are fully runnable locally.
"""
import pytest

pytestmark = pytest.mark.ollama


def test_build_and_hybrid_search(tmp_path):
    """Index a tiny document into a fresh Chroma store and query it."""
    import os

    os.environ["CHROMA_PATH"] = str(tmp_path / "chroma_test")

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings

    from config import EMBEDDING_MODEL
    from rag_engine import add_documents, hybrid_search, reset_vectorstore

    vs = Chroma(
        persist_directory=str(tmp_path / "chroma_test"),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_metadata={"hnsw:space": "cosine"},
    )
    reset_vectorstore(vs)

    docs = [
        Document(
            page_content=(
                "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
                "It was completed in 1889 and is named after the engineer Gustave Eiffel."
            ),
            metadata={"source": "eiffel.txt", "file_type": "txt"},
        )
    ]
    add_documents(vs, docs)

    results = hybrid_search(vs, "Where is the Eiffel Tower located?", k=3, min_relevance=0)
    assert isinstance(results, list)
    assert results, "expected at least one retrieved chunk"
    assert any("Paris" in d.page_content for d in results)
