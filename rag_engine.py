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

def hybrid_search(vectorstore, question, k=5, min_relevance=None, source=None):
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
    flt = {"source": source} if source else None
    try:
        try:
            scored = vectorstore.similarity_search_with_relevance_scores(question, k=k, filter=flt)
            semantic_results = [doc for doc, _ in scored]
            best_relevance = max((rel for _, rel in scored), default=0.0)
        except Exception:
            # Fallback if the collection has no relevance function configured.
            semantic_results = vectorstore.similarity_search(question, k=k, filter=flt)
            best_relevance = 1.0

        # Relevance gate: bail out early if nothing is semantically close enough.
        if min_relevance > 0 and best_relevance < min_relevance:
            return []

        bm25_results = []
        bm25_retriever = _get_bm25_retriever(vectorstore)
        if bm25_retriever is not None:
            # When scoped to one file, pull more BM25 candidates then keep only
            # that file's, since BM25 ranks over the whole corpus.
            bm25_retriever.k = k * 4 if source else k
            bm25_results = bm25_retriever.invoke(question)
            if source:
                bm25_results = [d for d in bm25_results if d.metadata.get("source") == source][:k]

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
        if source:
            return [holder[key] for key in ordered[:k]]

        # Unscoped ("All documents"): cap chunks per file so one large document
        # can't monopolize the results. Broad questions then draw from several
        # files automatically -- no manual scoping needed. Reranking still floats
        # the most relevant chunks to the top for focused questions.
        cap = max(2, k // 3)
        per_src, primary, overflow = {}, [], []
        for key in ordered:
            src = holder[key].metadata.get("source")
            if per_src.get(src, 0) < cap:
                per_src[src] = per_src.get(src, 0) + 1
                primary.append(key)
            else:
                overflow.append(key)
        selected = (primary + overflow)[:k]
        return [holder[key] for key in selected]

    except Exception as e:
        print(f"Hybrid search error: {e}")
        return vectorstore.similarity_search(question, k=k, filter=flt)

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

def ask_question(vectorstore, question, conversation_history=None, scope=None):
    """
    Main question answering function with improved routing and context

    Args:
        vectorstore: Chroma vector store
        question: User question
        conversation_history: List of previous exchanges for context
        scope: Optional filename to restrict retrieval to a single document
    """
    try:
        if conversation_history is None:
            conversation_history = []
        if is_analytic_question(question):
            try:
                retrieved_docs = hybrid_search(vectorstore, question, k=30, min_relevance=0, source=scope)
                selected_dataset = select_best_dataset(retrieved_docs)

                if selected_dataset:
                    result = analyze_dataframe(selected_dataset, question)
                    
                    if result:
                        return { "answer": result, "sources": format_sources_with_context(retrieved_docs), "is_analytics": True}
            except Exception as e:
                print(f"Analytics routing error: {e}")
            # No structured dataset produced an answer -> fall through to normal
            # document Q&A instead of dead-ending the query.

        retrieved_docs = hybrid_search(vectorstore, question, k=10, source=scope)

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


def _passages(docs):
    """Turn retrieved docs into numbered passage records (for citations + evidence)."""
    out = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        out.append({
            "n": i,
            "filename": m.get("source", "Unknown"),
            "file_type": m.get("file_type", "doc"),
            "page": m.get("page"),
            "rows": (f"{m.get('batch_start', 0)}-{m.get('batch_end', 0)}" if "batch_start" in m else None),
            "text": doc.page_content,
        })
    return out


def prepare_answer(vectorstore, question, conversation_history=None, scope=None):
    """
    Retrieve context and return everything the UI needs, including a token STREAM.

    Returns a dict:
      is_analytics : bool
      sources      : list of numbered passage records (n, filename, page, text, ...)
      mode         : "text" or "stream"
      text         : final answer string           (when mode == "text")
      stream       : generator of str tokens        (when mode == "stream")
    """
    if conversation_history is None:
        conversation_history = []
    try:
        # Analytics path (short pandas results — no need to stream).
        if is_analytic_question(question):
            try:
                docs = hybrid_search(vectorstore, question, k=30, min_relevance=0, source=scope)
                dataset = select_best_dataset(docs)
                if dataset:
                    result = analyze_dataframe(dataset, question)
                    if result:
                        return {"is_analytics": True, "mode": "text", "text": result,
                                "sources": _passages(docs[:5])}
            except Exception as e:
                print(f"Analytics routing error: {e}")
            # fall through to normal document Q&A

        # No LLM reranker here: it adds a slow, fragile extra round-trip before
        # streaming can begin. RRF + per-file diversity already rank well.
        docs = hybrid_search(vectorstore, question, k=8, source=scope)[:5]
        if not docs:
            return {"is_analytics": False, "mode": "text",
                    "text": "I couldn't find this in your documents.", "sources": []}

        passages = _passages(docs)
        numbered = "\n\n".join(
            f"[{p['n']}] {p['filename']}" + (f", p.{p['page']}" if p['page'] else "") + f"\n{p['text']}"
            for p in passages
        )
        context = numbered[:MAX_CONTEXT_LENGTH]
        conv = build_conversation_context(conversation_history, max_history=3) if conversation_history else ""

        prompt = f"""{conv}You are KnowledgeForge, a precise document question-answering assistant.
Answer the user's question using ONLY the numbered context passages below.

RULES:
- Use only facts stated in the context. Never invent, guess, or rely on outside knowledge.
- Read ALL passages, then combine the relevant details into one complete, well-structured answer. Group related facts; use short bullet points when it helps.
- Cite sources inline: right after a fact, add the passage number(s) in square brackets, e.g. [1] or [2][3].
- Stay faithful to the source wording; do not add opinions or numbers not in the context.
- Start with the answer itself. Do NOT begin with any preamble such as "Here is the answer", "Based on the context", or "Here are the key points".
- If the answer is not in the context, reply exactly: "I couldn't find this in your documents."
- If only part of the question is supported, answer that part and state what is missing.

CONTEXT PASSAGES:
{context}

QUESTION: {question}

ANSWER:"""

        def _stream():
            for chunk in llm.stream(prompt):
                yield chunk.content

        return {"is_analytics": False, "mode": "stream", "stream": _stream(), "sources": passages}

    except Exception as e:
        return {"is_analytics": False, "mode": "text", "text": f"Error: {e}", "sources": []}