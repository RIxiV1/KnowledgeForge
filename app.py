import os
import shutil
import hashlib
import time
import html
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root{
  --bg:#0A0E1A; --bg-2:#0E1424;
  --surface:#121A2B; --surface-2:#182136; --elevated:#1B2540;
  --border:#242F49; --border-strong:#33436A;
  --text:#EAEEF9; --muted:#8B96B0; --faint:#5A6685;
  --accent:#6366F1; --accent-2:#818CF8; --violet:#8B5CF6; --ember:#F59E0B;
  --radius:16px; --radius-sm:11px;
  --shadow:0 12px 34px -16px rgba(0,0,0,.7);
  --font:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  --display:'Plus Jakarta Sans','Inter',sans-serif;
}
html, body, [class*="css"], .stApp, textarea, input, button{ font-family:var(--font); }
.stApp{
  background:
    radial-gradient(1100px 520px at 82% -12%, rgba(99,102,241,.11), transparent 60%),
    radial-gradient(880px 500px at -5% 0%, rgba(139,92,246,.07), transparent 55%),
    var(--bg);
}
.block-container{ max-width:920px; padding-top:2rem; }
h1,h2,h3,h4{ font-family:var(--display); letter-spacing:-.02em; color:var(--text); }
p,span,label,li{ color:var(--text); }

[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0F1524,#0B1019); border-right:1px solid var(--border); }

/* Buttons */
.stButton>button{
  border-radius:var(--radius-sm); font-weight:600; border:1px solid var(--border);
  background:var(--elevated); color:var(--text);
  transition:transform .12s ease, border-color .15s ease, background .15s ease, box-shadow .15s ease;
}
.stButton>button:hover{ border-color:var(--accent); background:#20294552; transform:translateY(-1px);
  box-shadow:0 8px 20px -12px rgba(99,102,241,.75); }
.stButton>button:active{ transform:translateY(0); }
.stButton>button:focus:not(:active){ box-shadow:0 0 0 3px rgba(99,102,241,.32); }

/* Chat bubbles */
.stChatMessage{
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius); padding:16px 18px; margin-bottom:12px; box-shadow:var(--shadow);
  animation:msgIn .34s cubic-bezier(.2,.7,.2,1) both;
}
@keyframes msgIn{ from{opacity:0; transform:translateY(9px);} to{opacity:1; transform:none;} }
.stChatMessage:has([data-testid*="AvatarUser"]),
.stChatMessage:has([data-testid*="avatar-user"]){
  background:linear-gradient(180deg,#17203a,#131a2f); border-color:#2c3a63; }

/* Chat input */
[data-testid="stChatInput"]{ background:var(--surface); border:1px solid var(--border);
  border-radius:14px; box-shadow:var(--shadow); }
[data-testid="stChatInput"]:focus-within{ border-color:var(--accent); box-shadow:0 0 0 3px rgba(99,102,241,.22); }
[data-testid="stChatInput"] textarea{ font-size:1rem; }

/* Uploader */
[data-testid="stFileUploaderDropzone"]{
  background:linear-gradient(180deg,var(--surface),var(--bg-2));
  border:1.5px dashed var(--border-strong); border-radius:var(--radius); transition:border-color .15s ease; }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--accent); }

/* Metrics */
[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius-sm); padding:12px 15px; }
[data-testid="stMetricValue"]{ font-family:var(--display); font-weight:800; font-variant-numeric:tabular-nums; }
[data-testid="stMetricLabel"]{ color:var(--muted); font-weight:500; }

/* Expander / alerts */
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--surface); }
.stAlert{ border-radius:var(--radius-sm); border:1px solid var(--border); }

/* Chips, links, scrollbar, selection */
.kf-chip{ display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:999px;
  background:#161f36; border:1px solid var(--border); color:var(--muted); font-size:.8rem; font-weight:500; }
.kf-chip b{ color:var(--text); font-weight:600; }
.kf-meta{ display:flex; flex-wrap:wrap; gap:7px; margin-top:12px; }
.kf-src-wrap{ display:flex; flex-direction:column; gap:7px; }
.kf-src{ display:flex; align-items:center; gap:10px; padding:9px 12px; border-radius:11px;
  background:#131b2e; border:1px solid var(--border); }
.kf-badge{ font-size:.66rem; font-weight:700; letter-spacing:.04em; color:var(--accent-2);
  background:rgba(99,102,241,.14); border:1px solid rgba(99,102,241,.32); padding:2px 7px; border-radius:6px; }
.kf-src-name{ color:var(--text); font-size:.88rem; font-weight:500; word-break:break-word; }
.kf-src-meta{ color:var(--muted); font-size:.8rem; margin-left:auto; white-space:nowrap;
  font-variant-numeric:tabular-nums; }
a{ color:var(--accent-2); text-decoration:none; }
a:hover{ text-decoration:underline; }
hr{ border-color:var(--border); margin:16px 0; }
::selection{ background:rgba(99,102,241,.35); }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:#26304a; border-radius:8px; border:2px solid transparent; background-clip:padding-box; }
::-webkit-scrollbar-thumb:hover{ background:#37456A; }
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

# Assistant avatar. Browsers/Streamlit block data:image/svg+xml avatars, so use
# a reliable brand-matched emoji (blue diamond) that always renders.
_GEM_AVATAR = "🔷"


def _sources_html(sources):
    """Render retrieved sources as compact cards, deduped by file + page."""
    if not sources:
        return ""
    seen, cards = set(), []
    for s in sources:
        if not isinstance(s, dict):
            key = str(s)
            if key in seen:
                continue
            seen.add(key)
            cards.append(f'<div class="kf-src"><span class="kf-src-name">{html.escape(key)}</span></div>')
            continue
        name = html.escape(s.get("filename", "Unknown"))
        ftype = html.escape((s.get("file_type") or "doc").upper())
        page, rows = s.get("page"), s.get("rows")
        key = (name, page, rows)
        if key in seen:
            continue
        seen.add(key)
        bits = []
        if page:
            bits.append(f"page {html.escape(str(page))}")
        if rows:
            bits.append(f"rows {html.escape(str(rows))}")
        meta = f'<span class="kf-src-meta">{" · ".join(bits)}</span>' if bits else ""
        cards.append(
            f'<div class="kf-src"><span class="kf-badge">{ftype}</span>'
            f'<span class="kf-src-name">{name}</span>{meta}</div>'
        )
    return '<div class="kf-src-wrap">' + "".join(cards) + "</div>"


def _meta_html(latency, is_analytics, n_sources):
    """A small row of pill chips: latency · mode · source count."""
    mode = "Analytics" if is_analytics else "Document Q&A"
    return (
        '<div class="kf-meta">'
        f'<span class="kf-chip"><b>{latency}s</b></span>'
        f'<span class="kf-chip">{mode}</span>'
        f'<span class="kf-chip"><b>{n_sources}</b> source(s)</span>'
        "</div>"
    )


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

# Empty state + one-tap starter questions when there's no conversation yet.
if not st.session_state.conversation_history:
    st.markdown(EMPTY_STATE_HTML, unsafe_allow_html=True)
    if get_indexed_files():
        st.markdown(
            '<div style="text-align:center;color:#6B7688;font-size:.78rem;'
            'letter-spacing:.06em;margin:2px 0 8px;">TRY ASKING</div>',
            unsafe_allow_html=True,
        )
        _examples = ["Summarize the key points", "What problem does this solve?", "What are the main features?"]
        for _col, _ex in zip(st.columns(len(_examples)), _examples):
            if _col.button(_ex, key=f"ex::{_ex}", use_container_width=True):
                st.session_state.pending_question = _ex
                st.rerun()

# Conversation transcript (rendered uniformly; new answers appear here after rerun).
for exchange in st.session_state.conversation_history:
    with st.chat_message("user"):
        st.write(exchange["question"])
    with st.chat_message("assistant", avatar=_GEM_AVATAR):
        st.write(exchange["answer"])
        st.markdown(
            _meta_html(
                exchange.get("latency", 0),
                exchange.get("is_analytics", False),
                len(exchange.get("sources") or []),
            ),
            unsafe_allow_html=True,
        )
        if exchange.get("sources"):
            with st.expander("Sources"):
                st.markdown(_sources_html(exchange["sources"]), unsafe_allow_html=True)

# Input: a typed question, or a starter chip that set pending_question.
typed = st.chat_input("Ask a question about your documents…")
question = typed or st.session_state.pop("pending_question", None)

if question:
    if len(question.strip()) < 3:
        st.warning("Please enter a longer question.")
    else:
        start_time = time.time()
        answer, sources, is_analytics = "No response generated", [], False
        with st.spinner("Searching your documents…"):
            try:
                result = ask_question(
                    st.session_state.vectorstore,
                    question,
                    conversation_history=st.session_state.conversation_history,
                )
                answer = result.get("answer", "No response generated")
                sources = result.get("sources", [])
                is_analytics = result.get("is_analytics", False)
                save_chat(question, answer)
            except Exception as e:
                answer = f"Error: {str(e)}"
                sources = []
        latency = round(time.time() - start_time, 2)
        st.session_state.conversation_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "is_analytics": is_analytics,
            "latency": latency,
        })
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]
        st.rerun()
