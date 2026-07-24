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


def test_multi_file_scope(tmp_path):
    """A list `source` restricts retrieval to exactly those files, covering each."""
    import os

    os.environ["CHROMA_PATH"] = str(tmp_path / "chroma_multi")

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings

    from config import EMBEDDING_MODEL
    from rag_engine import add_documents, hybrid_search, reset_vectorstore

    vs = Chroma(
        persist_directory=str(tmp_path / "chroma_multi"),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_metadata={"hnsw:space": "cosine"},
    )
    reset_vectorstore(vs)

    add_documents(vs, [Document(page_content="The mitochondria produces ATP energy.",
                                metadata={"source": "bio.pdf", "file_type": "pdf"})], file_hash="h1")
    add_documents(vs, [Document(page_content="React is a JavaScript UI library.",
                                metadata={"source": "web.txt", "file_type": "txt"})], file_hash="h2")
    add_documents(vs, [Document(page_content="Photosynthesis converts sunlight to energy.",
                                metadata={"source": "plants.pdf", "file_type": "pdf"})], file_hash="h3")

    docs = hybrid_search(vs, "what are these about", k=8, min_relevance=0,
                         source=["bio.pdf", "web.txt"])
    got = {d.metadata["source"] for d in docs}
    assert got <= {"bio.pdf", "web.txt"}, f"scope leaked: {got}"
    assert "plants.pdf" not in got
