# KnowledgeForge

A local-first Retrieval-Augmented Generation (RAG) chatbot for asking questions over your own documents. Upload files, and chat with a local LLM that answers using your content and cites the sources it used — all running on your machine via Ollama, with no API keys and no data leaving your computer.

Built with Ollama, ChromaDB, LangChain, and Streamlit.

## Features

- **Multi-format ingestion** — PDF, TXT, CSV, XLSX/XLS, DOCX, PPTX, and JSON. Uploads are de-duplicated by content hash (persisted, so it survives restarts) and each file can be removed individually — which also deletes its vectors.
- **Hybrid retrieval** — combines semantic (vector) search with BM25 keyword search using Reciprocal Rank Fusion, so both meaning and exact-term matches contribute to ranking. The BM25 index is cached and only rebuilt when the document set changes, and results are capped per file so no single document dominates.
- **Streaming, cited answers** — answers stream token-by-token with inline `[n]` citations, an evidence panel showing the exact source passages, and a relevance gate that says "I couldn't find this" instead of answering from noise. The context is fenced against prompt injection.
- **Socratic Study Mode** — instead of answering, it quizzes you from your own documents, grades your answer against the source, highlights the passage you missed, and tracks a session score.
- **Conversation memory** — recent turns are fed back as context within a session, and full history is persisted to SQLite.
- **Lightweight tabular analytics** — questions like "total revenue" or "top 5 by price" over an uploaded CSV/Excel file are answered with pandas via keyword-based intent detection (no LLM-generated code).

> **Note:** This is a personal/portfolio-scale project designed to run locally for a single user. It is not a hardened multi-tenant service — there is no authentication, no REST API, and throughput is bounded by your local Ollama instance.

## Tech Stack

- **UI:** Streamlit
- **LLM & embeddings:** Ollama (local inference, no API keys)
- **Vector store:** ChromaDB
- **Orchestration:** LangChain
- **Keyword retrieval:** rank-bm25
- **History:** SQLite
- **Language:** Python 3.11+

## Architecture

```
Streamlit UI (app.py)
    |
RAG engine (rag_engine.py)  ──  Analytics (analytics_engine.py)
    |                                   |
Retrieval: Chroma vector search + BM25 (RRF fusion)
    |
Persistence: ChromaDB (vectors) · SQLite (chat history) · uploads/ (files)
```

Modules:

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit interface, upload handling, chat loop |
| `rag_engine.py` | Chunking, hybrid search (RRF), prompt + streaming generation |
| `study_engine.py` | Socratic study: question generation, answer grading, passage highlight |
| `vector_store.py` | ChromaDB / embeddings setup |
| `document_loader.py` | Per-format document loading |
| `analytics_engine.py` | Keyword-based tabular analytics |
| `database.py` | SQLite chat history |
| `config.py` | Configuration (env-driven) |

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Enough RAM/CPU (or a GPU) to run your chosen Ollama model

## Setup

```bash
git clone https://github.com/RIxiV1/KnowledgeForge-AI-Powered-RAG-Platform.git
cd KnowledgeForge-AI-Powered-RAG-Platform

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Pull the models used by default:

```bash
ollama serve            # in one terminal
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

## Configuration

Configuration is read from environment variables (a local `.env` file is loaded automatically). Copy the example and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `llama3.1:8b` | Ollama model for generation |
| `EMBEDDING_MODEL` | `mxbai-embed-large:latest` | Ollama model for embeddings |
| `CHROMA_PATH` | `chroma_db` | Vector store directory |
| `DATABASE_PATH` | `chat_history.db` | SQLite history file |
| `CHUNK_SIZE` | `1200` | Chunk size (characters) |
| `CHUNK_OVERLAP` | `150` | Chunk overlap (characters) |
| `MAX_CONTEXT_LENGTH` | `9000` | Max characters of context sent to the LLM |
| `LLM_NUM_CTX` | `8192` | Model context window (tokens); keep ≥ `MAX_CONTEXT_LENGTH` in tokens |
| `MAX_UPLOAD_MB` | `50` | Maximum uploaded-file size, rejected before parsing |
| `RELEVANCE_THRESHOLD` | `0.15` | Min semantic relevance (0–1) to answer; below it the app says it couldn't find the answer. `0` disables the gate. |

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501, upload some documents from the top of the page, and start asking questions.

## How it works

1. **Ingestion** — files are parsed per format and split into overlapping chunks; chunks are embedded and stored in ChromaDB. Tabular files (CSV/Excel) are stored in row batches with metadata so they can be located for analytics.
2. **Retrieval** — for each question, the app runs vector search and BM25 keyword search, fuses the two rankings with Reciprocal Rank Fusion, and caps chunks per file so broad questions span multiple documents.
3. **Generation** — retrieved context (fenced against prompt injection) plus recent conversation history is streamed from the LLM with a grounding prompt; the answer shows inline citations and its source passages.
4. **Analytics path** — if a question looks like an aggregation ("count", "average", "top N"…), it is answered directly from the relevant dataframe with pandas instead of the LLM.

## Roadmap

Ideas for future work (not yet implemented):

- Configurable retrieval weights and reranking toggle
- A REST/FastAPI layer so the engine can be used headlessly
- Evaluation harness to measure retrieval quality on a labelled set
- Dockerfile / compose for one-command deployment
- Authentication and multi-user support

## Docker deployment

A `Dockerfile` and `docker-compose.yml` are provided to run the app and Ollama as two containers. The app image contains only the Streamlit app; the models run in a separate `ollama` service.

Bring the stack up:

```bash
docker compose up --build
```

This starts:

- `ollama` — the model server, on port `11434`, with its models persisted in the named `ollama` volume.
- `app` — the Streamlit UI, on port `8501` (open http://localhost:8501).

**Pull the models the first time.** The `ollama` container starts with no models. After the stack is up, pull the two models the app uses into the running container (they persist in the `ollama` volume):

```bash
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull mxbai-embed-large
```

**Base URL.** Both `ChatOllama` and `OllamaEmbeddings` read the `OLLAMA_HOST` environment variable, so inside the compose network the app reaches Ollama at `http://ollama:11434` (set for the `app` service in `docker-compose.yml`). For the local, non-Docker setup, leave `OLLAMA_HOST` unset and it defaults to `http://localhost:11434`.

**GPU acceleration.** By default Ollama runs on CPU. GPU acceleration is optional and requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host; uncomment the `deploy.resources.reservations.devices` block under the `ollama` service in `docker-compose.yml` to enable it.

## Credits

This is a fork of [KnowledgeForge by Arsath Mohamed](https://github.com/ArsathMohamed351/KnowledgeForge-AI-Powered-RAG-Platform). Original concept and implementation by the upstream author; this fork adds fixes and improvements to retrieval, configuration, and packaging.

## License

MIT License.
