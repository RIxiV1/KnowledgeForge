import os
import shutil
import hashlib
import time
import streamlit as st
from document_loader import load_file
from vector_store import get_vectorstore
from rag_engine import add_documents, ask_question, delete_documents, reset_vectorstore
from database import (
    create_tables,
    save_chat,
    get_chat_history,
    clear_history,
    save_file,
    get_indexed_files,
    get_indexed_hashes,
    delete_file,
    clear_indexed_files,
)

create_tables()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config( page_title="KnowledgeForge", page_icon="◆", layout="wide", initial_sidebar_state="expanded" )

# Basic designing

st.markdown("""
<style>
:root{
  --ground:#0B1120; --surface:#161E2E; --surface-2:#1B2437;
  --border:#27324B; --text:#E6EAF3; --muted:#93A0B8;
  --accent:#6366F1; --accent-hover:#818CF8;
}
.stApp, [data-testid="stAppViewContainer"]{ background:var(--ground); }
h1,h2,h3,h4,h5,p,span,label,li{ color:var(--text); }
[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--border); }
[data-testid="stSidebar"] .stButton>button{
  background:var(--surface-2); color:var(--text); border:1px solid var(--border);
  border-radius:9px; font-weight:500; transition:border-color .15s ease, color .15s ease; }
[data-testid="stSidebar"] .stButton>button:hover{ border-color:var(--accent); color:#fff; }
.stButton>button{ border-radius:9px; }
.stChatMessage{ background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:14px 16px; margin-bottom:10px; }
[data-testid="stChatInput"]{ background:var(--surface); border:1px solid var(--border); border-radius:12px; }
[data-testid="stChatInput"]:focus-within{ border-color:var(--accent); }
[data-testid="stFileUploaderDropzone"]{
  background:var(--surface); border:1.5px dashed var(--border); border-radius:14px; }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--accent); }
[data-testid="stMetricValue"]{ color:var(--text); font-variant-numeric:tabular-nums; font-weight:600; }
[data-testid="stMetricLabel"]{ color:var(--muted); }
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:12px; background:var(--surface); }
.stAlert{ border-radius:10px; }
a{ color:var(--accent-hover); text-decoration:none; }
hr{ border-color:var(--border); margin:14px 0; }
#MainMenu, footer, header{ visibility:hidden; }
@media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }
</style>

""", unsafe_allow_html=True)

# ---- Brand mark: "Forged Facet" -------------------------------------------
# A cut gem — two indigo facets + a warm amber crown — reads as the ember of
# insight refined from raw documents. Static, geometric, legible at any size.
def _gem(size):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 32 32" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" style="display:block;flex:0 0 auto;">'
        '<path d="M16 2.5 L2.5 16 L16 29.5 Z" fill="#6366F1"/>'
        '<path d="M16 2.5 L29.5 16 L16 29.5 Z" fill="#4F46E5"/>'
        '<path d="M16 2.5 L22.5 9.5 L16 13 L9.5 9.5 Z" fill="#F59E0B"/>'
        '</svg>'
    )

_WORDMARK = 'Knowledge<span style="color:#818CF8;">Forge</span>'

HERO_HTML = f"""
<div style="display:flex;align-items:center;gap:13px;margin:2px 0 4px;">
  {_gem(34)}
  <div style="font-size:1.5rem;font-weight:700;letter-spacing:-.02em;color:#E6EAF3;">{_WORDMARK}</div>
</div>
<div style="color:#93A0B8;font-size:.92rem;margin:0 0 18px;max-width:62ch;">
  Ask questions across your documents — answers grounded in your own sources, with the files they came from.
</div>
"""

SIDEBAR_LOGO_HTML = f"""
<div style="display:flex;align-items:center;gap:9px;margin:2px 0 1px;">
  {_gem(26)}
  <div style="font-size:1.02rem;font-weight:700;letter-spacing:-.01em;color:#E6EAF3;">{_WORDMARK}</div>
</div>
<div style="color:#93A0B8;font-size:.76rem;margin:0 0 4px;">Private · local · source-grounded</div>
"""

EMPTY_STATE_HTML = f"""
<div style="text-align:center;padding:40px 0 12px;">
  <div style="opacity:.55;display:inline-block;">{_gem(40)}</div>
  <div style="color:#C7CEDB;margin-top:14px;font-size:.98rem;font-weight:500;">Your knowledge base is ready</div>
  <div style="color:#6B7688;font-size:.86rem;margin-top:3px;">Upload a document above, then ask a question about it.</div>
</div>
"""

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "uploaded_hashes" not in st.session_state:
    # Seed from the persistent registry so re-uploading an already-indexed file
    # across restarts doesn't create duplicate vectors.
    st.session_state.uploaded_hashes = get_indexed_hashes()

if "show_history" not in st.session_state:
    st.session_state.show_history = False

if "show_delete_warning" not in st.session_state:
    st.session_state.show_delete_warning = False

with st.sidebar:
    st.markdown(SIDEBAR_LOGO_HTML, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### Knowledge base")
    
    # Chat history info
    history_count = len(get_chat_history())
    st.metric("Conversations", history_count)

    if st.button("View history", use_container_width=True):
        st.session_state.show_history = not st.session_state.get("show_history", False)

    # Indexed documents, with per-file removal (deletes their vectors too).
    indexed_files = get_indexed_files()
    st.metric("Indexed Files", len(indexed_files))
    if indexed_files:
        with st.expander("Manage documents"):
            for fhash, fname in indexed_files:
                fcol, bcol = st.columns([4, 1])
                fcol.caption(fname)
                if bcol.button("Remove", key=f"del_{fhash}", use_container_width=True):
                    delete_documents(st.session_state.vectorstore, fhash)
                    delete_file(fhash)
                    st.session_state.uploaded_hashes.discard(fhash)
                    try:
                        os.remove(os.path.join(UPLOAD_DIR, fname))
                    except OSError:
                        pass
                    st.success(f"Removed {fname}")
                    time.sleep(0.5)
                    st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear history", use_container_width=True):
            clear_history()
            st.session_state.conversation_history = []
            st.success("History cleared")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("Delete all", use_container_width=True):
            st.session_state.show_delete_warning = True

    # Show warning if delete-all button was clicked
    if st.session_state.get("show_delete_warning", False):
        st.warning("This removes every document and its answers.")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Yes, Delete", use_container_width=True, key="confirm_delete"):
                try:
                    # Wipe files, their vectors, and the registry together so
                    # deleted documents can't still surface in answers.
                    reset_vectorstore(st.session_state.vectorstore)
                    clear_indexed_files()
                    if os.path.exists("uploads"):
                        shutil.rmtree("uploads")
                        os.makedirs("uploads", exist_ok=True)
                    st.session_state.uploaded_hashes = set()
                    st.session_state.show_delete_warning = False
                    st.success("All uploads deleted!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        with col2:
            if st.button("Cancel", use_container_width=True, key="cancel_delete"):
                st.session_state.show_delete_warning = False
                st.info("Cancelled")
                time.sleep(1)
                st.rerun()

st.markdown(HERO_HTML, unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload documents to your knowledge base",
    accept_multiple_files=True,
    type=["pdf", "txt", "csv", "xlsx", "xls", "docx", "pptx", "json"]
)

if uploaded_files:
    progress_placeholder = st.empty()
    
    for uploaded_file in uploaded_files:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        # Skip if already uploaded
        if file_hash in st.session_state.uploaded_hashes:
            st.info(f"{uploaded_file.name} already indexed")
            continue
        
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # file processing
        with st.spinner(f"Indexing {uploaded_file.name}…"):
            try:
                docs = load_file(file_path)
                add_documents(st.session_state.vectorstore, docs, file_hash=file_hash)
                st.session_state.uploaded_hashes.add(file_hash)
                save_file(file_hash, uploaded_file.name)
                st.success(f"Indexed {uploaded_file.name} — {len(docs)} section(s)")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
if st.session_state.get("show_history", False):
    with st.expander("Conversation history", expanded=True):
        history = get_chat_history()
        
        if history:
            for idx, (question, answer, created_at) in enumerate(history, 1):
                with st.container():
                    st.caption(f"**{idx}. {created_at}**")
                    st.write(f"**Q:** {question}")
                    # Handle None answer
                    if answer is None:
                        st.write("**A:** (No answer saved)")
                    # long answer truncation with expander
                    elif len(answer) > 300:
                        with st.expander("View full answer"):
                            st.write(answer)
                    else:
                        st.write(f"**A:** {answer}")
                    
                    st.divider()
        else:
            st.info("No conversations yet.")

# MAIN INTERFACE

# Calm empty state when there's no conversation yet
if not st.session_state.conversation_history:
    st.markdown(EMPTY_STATE_HTML, unsafe_allow_html=True)

# conversation history
for exchange in st.session_state.conversation_history:
    with st.chat_message("user"):
        st.write(exchange["question"])    
    with st.chat_message("assistant"):
        st.write(exchange["answer"])
        if exchange.get("sources"):
            with st.expander("Sources"):
                for source in exchange["sources"]:
                    if isinstance(source, dict):
                        source_text = f"{source.get('filename', 'Unknown')}"
                        if source.get('file_type'):
                            source_text += f" ({source['file_type'].upper()})"
                        if source.get('rows'):
                            source_text += f" - Rows: {source['rows']}"
                        st.info(source_text)
                    else:
                        st.info(f"{source}")

question = st.chat_input("Ask a question about your documents…")

if question:
    # input validation
    if len(question.strip()) < 3:
        st.warning("Please ask a longer question")
    else:
        # user message
        with st.chat_message("user"):
            st.write(question)
        
        # question Process
        start_time = time.time()
        # Defaults so the metrics row below never hits an undefined name if the
        # call raises before these are assigned.
        answer = "No response generated"
        sources = []
        is_analytics = False
        with st.spinner("Searching your documents…"):
            try:
                result = ask_question( st.session_state.vectorstore, question, conversation_history=st.session_state.conversation_history )
                answer = result.get("answer", "No response generated")
                sources = result.get("sources", [])
                is_analytics = result.get("is_analytics", False)

                save_chat(question, answer)

                st.session_state.conversation_history.append({ "question": question, "answer": answer, "sources": sources, "is_analytics": is_analytics })

                if len(st.session_state.conversation_history) > 10:
                    st.session_state.conversation_history = st.session_state.conversation_history[-10:]
            except Exception as e:
                answer = f"Error: {str(e)}"
                sources = []
        latency = round(time.time() - start_time, 2)
        with st.chat_message("assistant"):
            st.write(answer)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"{latency}s")
            with col2:
                if is_analytics:
                    st.caption("Analytics")
                else:
                    st.caption("Document Q&A")
            with col3:
                st.caption(f"{len(sources)} source(s)")
            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources, 1):
                        if isinstance(source, dict):
                            source_text = f"**{i}. {source.get('filename', 'Unknown')}**"
                            if source.get('file_type'):
                                source_text += f" ({source['file_type'].upper()})"
                            st.info(source_text)
                            if source.get('rows'):
                                st.caption(f"Rows: {source['rows']}")
                        else:
                            st.info(f"**{i}. {source}**")
