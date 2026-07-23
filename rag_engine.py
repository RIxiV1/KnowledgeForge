import uuid
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from analytics_engine import is_analytic_question, analyze_dataframe
from config import *

llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=LLM_NUM_CTX)
splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# Cache the BM25 index so we don't rebuild it over the whole corpus on every query.
_bm25_cache = {"count": -1, "retriever": None}


def _invalidate_bm25_cache():
    _bm25_cache["count"] = -1
    _bm25_cache["retriever"] = None


def add_documents(vectorstore, docs, file_hash=None):
    """Add documents to vector store with batch processing.

    If file_hash is given, every chunk is tagged with it so the file's vectors
    can later be removed as a unit (see delete_documents).
    """
    chunks = splitter.split_documents(docs)

    if file_hash:
        for chunk in chunks:
            chunk.metadata["file_hash"] = file_hash

    print(f"Total Chunks Created: {len(chunks)}")
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [str(uuid.uuid4()) for _ in batch]

        try:
            vectorstore.add_documents(documents=batch, ids=ids)
            print(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")
        except Exception as e:
            print(f"Batch Failed: {e}")
    _invalidate_bm25_cache()
    print("Document Indexing Complete")


def delete_documents(vectorstore, file_hash):
    """Remove every chunk belonging to a file (by its content hash)."""
    try:
        vectorstore._collection.delete(where={"file_hash": file_hash})
        _invalidate_bm25_cache()
        return True
    except Exception as e:
        print(f"Delete failed: {e}")
        return False


def reset_vectorstore(vectorstore):
    """Remove all documents from the vector store."""
    try:
        data = vectorstore.get()
        ids = data.get("ids") or []
        if ids:
            vectorstore.delete(ids=ids)
        _invalidate_bm25_cache()
        return True
    except Exception as e:
        print(f"Reset failed: {e}")
        return False

def select_best_dataset(docs):
    """Select best dataset from retrieved documents"""
    dataset_score = {}
    for doc in docs:
        dataset_id = doc.metadata.get("dataset_id")
        if dataset_id:
            dataset_score[dataset_id] = dataset_score.get(dataset_id, 0) + 1
    if not dataset_score:
        return None
    return max(dataset_score, key=dataset_score.get)

def _get_bm25_retriever(vectorstore):
    """
    Build (or reuse a cached) BM25 retriever over the whole corpus.

    Rebuilding BM25 from every document on each query is O(corpus) and was the
    main latency cost of the old hybrid search. We cache the retriever and only
    rebuild when the number of indexed documents changes (e.g. a new upload).
    """
    all_docs = vectorstore.get()
    documents = all_docs.get("documents") or []
    metadatas = all_docs.get("metadatas") or []

    if not documents:
        return None

    if _bm25_cache["count"] == len(documents) and _bm25_cache["retriever"] is not None:
        return _bm25_cache["retriever"]

    docs_for_bm25 = [
        Document(page_content=content, metadata=metadatas[i] if i < len(metadatas) else {})
        for i, content in enumerate(documents)
    ]
    retriever = BM25Retriever.from_documents(docs_for_bm25)
    _bm25_cache["count"] = len(documents)
    _bm25_cache["retriever"] = retriever
    return retriever

def hybrid_search(vectorstore, question, k=5, min_relevance=None):
    """
    Hybrid retrieval: fuse semantic (vector) and keyword (BM25) rankings with
    Reciprocal Rank Fusion (RRF).

    RRF combines the two lists by rank position, so both signals contribute a
    real, comparable score -- unlike the previous version, where every BM25 hit
    got a flat 0.5 and results were deduped by object identity (which never
    matched between the two retrievers, so the "hybrid" merge never happened).

    A relevance gate (min_relevance) returns no results when even the best
    semantic match is too weak, so the caller can say "not found" instead of
    answering from irrelevant chunks. Pass min_relevance=0 to disable it.
    """
    if min_relevance is None:
        min_relevance = RELEVANCE_THRESHOLD
    try:
        try:
            scored = vectorstore.similarity_search_with_relevance_scores(question, k=k)
            semantic_results = [doc for doc, _ in scored]
            best_relevance = max((rel for _, rel in scored), default=0.0)
        except Exception:
            # Fallback if the collection has no relevance function configured.
            semantic_results = vectorstore.similarity_search(question, k=k)
            best_relevance = 1.0

        # Relevance gate: bail out early if nothing is semantically close enough.
        if min_relevance > 0 and best_relevance < min_relevance:
            return []

        bm25_results = []
        bm25_retriever = _get_bm25_retriever(vectorstore)
        if bm25_retriever is not None:
            bm25_retriever.k = k
            bm25_results = bm25_retriever.invoke(question)

        # Reciprocal Rank Fusion. Dedup by content so the same chunk retrieved by
        # both methods is merged and its scores add up.
        C = 60  # standard RRF damping constant
        scores = {}
        holder = {}
        for ranked_list in (semantic_results, bm25_results):
            for rank, doc in enumerate(ranked_list):
                key = doc.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (C + rank + 1)
                holder.setdefault(key, doc)

        if not scores:
            return semantic_results

        ordered = sorted(scores, key=scores.get, reverse=True)
        return [holder[key] for key in ordered[:k]]

    except Exception as e:
        print(f"Hybrid search error: {e}")
        return vectorstore.similarity_search(question, k=k)

def rerank_documents(docs, question, llm):
    """
    Rerank retrieved documents using LLM
    Keeps top documents relevant to the question
    """
    if len(docs) <= 3:
        return docs
    
    try:
        doc_summaries = "\n".join([f"{i+1}. {doc.page_content[:200]}..." for i, doc in enumerate(docs)])
        
        rerank_prompt = f"""
Given the question and document summaries, rank these documents by relevance (1=most relevant, {len(docs)}=least relevant).
Return ONLY the ranking as numbers separated by commas, like: 3,1,5,2,4
QUESTION: {question}
DOCUMENTS:
{doc_summaries}

RANKING (numbers only):
"""    
        response = llm.invoke(rerank_prompt)
        ranking_str = response.content.strip()
    
        try:
            ranking = [int(x.strip()) - 1 for x in ranking_str.split(',')]
            # Reorder documents based on LLM ranking
            reranked = [docs[i] for i in ranking if i < len(docs)]
            return reranked
        except:
            return docs
    except:
        return docs

def format_sources_with_context(retrieved_docs):
    """
    Format sources with page/section information
    """
    sources = []
    for doc in retrieved_docs:
        source_info = { "filename": doc.metadata.get("source", "Unknown"), "file_type": doc.metadata.get("file_type", "Unknown"), "file_path": doc.metadata.get("file_path", "Unknown"), }

        if doc.metadata.get("page"):
            source_info["page"] = doc.metadata["page"]
        if "batch_start" in doc.metadata:
            source_info["rows"] = f"{doc.metadata.get('batch_start', 0)}-{doc.metadata.get('batch_end', 0)}"
        sources.append(source_info)
    return sources

def build_conversation_context(conversation_history, max_history=5):
    """
    Build context from conversation history
    Includes last N exchanges to maintain context
    """
    context = "CONVERSATION HISTORY:\n"
    
    # Get last max_history exchanges
    recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
    
    for exchange in recent_history:
        context += f"\nUser: {exchange['question']}\nAssistant: {exchange['answer'][:500]}...\n"
    
    context += "\n---\n"
    return context

def ask_question(vectorstore, question, conversation_history=None):
    """
    Main question answering function with improved routing and context
    
    Args:
        vectorstore: Chroma vector store
        question: User question
        conversation_history: List of previous exchanges for context
    """
    try:
        if conversation_history is None:
            conversation_history = []
        if is_analytic_question(question):
            try:
                retrieved_docs = hybrid_search(vectorstore, question, k=30, min_relevance=0)
                selected_dataset = select_best_dataset(retrieved_docs)

                if selected_dataset:
                    result = analyze_dataframe(selected_dataset, question)
                    
                    if result:
                        return { "answer": result, "sources": format_sources_with_context(retrieved_docs), "is_analytics": True}
            except Exception as e:
                print(f"Analytics routing error: {e}")
            # No structured dataset produced an answer -> fall through to normal
            # document Q&A instead of dead-ending the query.

        retrieved_docs = hybrid_search(vectorstore, question, k=10)
        
        # Rerank documents using LLM for better relevance
        retrieved_docs = rerank_documents(retrieved_docs, question, llm)
        
        # top 5 after reranking
        retrieved_docs = retrieved_docs[:5]

        if not retrieved_docs:
            return { "answer": "No relevant information found in your documents. Try uploading more documents or rephrasing your question.", "sources": [], "is_analytics": False }
        # Number each passage with its source so the model can ground precisely.
        numbered = []
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source", "document")
            page = doc.metadata.get("page")
            tag = f"[{i}] {src}" + (f", p.{page}" if page else "")
            numbered.append(f"{tag}\n{doc.page_content}")
        context = "\n\n".join(numbered)[:MAX_CONTEXT_LENGTH]

        # Build conversation context for better continuity
        conversation_context = ""
        if conversation_history:
            conversation_context = build_conversation_context(conversation_history, max_history=3)
        
        prompt = f"""{conversation_context}You are KnowledgeForge, a precise document question-answering assistant.
Answer the user's question using ONLY the numbered context passages below.

RULES:
- Use only facts stated in the context. Never invent, guess, or rely on outside knowledge.
- Read ALL passages before answering, then combine the relevant details into one complete, well-structured answer. Group related facts; use short bullet points when it improves clarity.
- Stay faithful to the source wording; do not add opinions, commentary, or numbers that are not in the context.
- Answer directly. Do NOT preface the answer with meta-phrases like "Based on the context" or "Here is the answer" — just give the answer.
- If the answer is not in the context, reply exactly: "I couldn't find this in your documents."
- If only part of the question is supported, answer that part and state what is missing.
- When the question refers to earlier turns, use the conversation history only if the context supports those facts.

CONTEXT PASSAGES:
{context}

QUESTION: {question}

ANSWER:"""
        response = llm.invoke(prompt)
        sources = format_sources_with_context(retrieved_docs)
        return { "answer": response.content, "sources": sources, "is_analytics": False }
    except Exception as e:
        return { "answer": f"Error processing your question: {str(e)}", "sources": [], "is_analytics": False }