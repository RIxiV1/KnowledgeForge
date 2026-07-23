import os
import re
import shutil
import hashlib
import time
import html
import json
import traceback
import streamlit as st
from config import MAX_UPLOAD_MB
from document_loader import load_file
from vector_store import get_vectorstore
from rag_engine import add_documents, prepare_answer, delete_documents, reset_vectorstore
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
.kf-ev-wrap{ display:flex; flex-direction:column; gap:9px; }
.kf-ev{ border:1px solid var(--border); border-radius:11px; background:#111a2c; padding:10px 13px; }
.kf-ev-h{ color:var(--accent-2); font-size:.76rem; font-weight:700; letter-spacing:.02em; margin-bottom:5px; }
.kf-ev-b{ color:var(--muted); font-size:.83rem; line-height:1.55; white-space:pre-wrap; word-break:break-word; }
/* Live "thinking / searching" activity indicator */
.kf-think{ display:flex; align-items:center; gap:11px; padding:4px 2px; }
.kf-think .dots{ display:inline-flex; gap:5px; }
.kf-think .dots i{ width:7px; height:7px; border-radius:50%; background:var(--accent-2);
  display:inline-block; animation:kfpulse 1.1s ease-in-out infinite; }
.kf-think .dots i:nth-child(2){ animation-delay:.16s; }
.kf-think .dots i:nth-child(3){ animation-delay:.32s; }
@keyframes kfpulse{ 0%,100%{ transform:scale(.55); opacity:.35; } 50%{ transform:scale(1); opacity:1; } }
.kf-shimmer{ font-weight:600; color:#8b96b0;
  background:linear-gradient(90deg,#8b96b0 0%,#8b96b0 35%,#e6eaf9 50%,#818cf8 60%,#8b96b0 75%);
  background-size:220% 100%; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; animation:kfshimmer 1.7s linear infinite; }
@keyframes kfshimmer{ 0%{ background-position:120% 0; } 100%{ background-position:-120% 0; } }
/* Streaming typing cursor */
.kf-cursor{ display:inline-block; width:8px; height:1.05em; transform:translateY(2px);
  background:var(--accent-2); border-radius:1px; margin-left:1px; animation:kfblink .9s steps(2,start) infinite; }
@keyframes kfblink{ 0%,50%{ opacity:1; } 51%,100%{ opacity:0; } }
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

THINKING_HTML = """
<div class="kf-think">
  <span class="dots"><i></i><i></i><i></i></span>
  <span class="kf-shimmer">Searching your documents…</span>
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
        num = s.get("n")
        badge = f"[{num}] {ftype}" if num else ftype
        cards.append(
            f'<div class="kf-src"><span class="kf-badge">{badge}</span>'
            f'<span class="kf-src-name">{name}</span>{meta}</div>'
        )
    return '<div class="kf-src-wrap">' + "".join(cards) + "</div>"


def _evidence_html(sources):
    """Render the exact passages behind the answer (numbered to match [n] citations)."""
    if not sources:
        return ""
    rows = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        num = s.get("n")
        head = (f"[{num}] " if num else "") + html.escape(str(s.get("filename", "")))
        if s.get("page"):
            head += f" · p.{html.escape(str(s['page']))}"
        full = s.get("text") or ""
        body = html.escape(full[:700]) + (" …" if len(full) > 700 else "")
        rows.append(f'<div class="kf-ev"><div class="kf-ev-h">{head}</div><div class="kf-ev-b">{body}</div></div>')
    return '<div class="kf-ev-wrap">' + "".join(rows) + "</div>"


def _render_extras(latency, is_analytics, sources):
    """Metric chips + Sources + Evidence, shared by live and replayed messages."""
    st.markdown(_meta_html(latency, is_analytics, len(sources or [])), unsafe_allow_html=True)
    if sources:
        with st.expander("Sources"):
            st.markdown(_sources_html(sources), unsafe_allow_html=True)
        with st.expander("Evidence — the exact passages used"):
            st.markdown(_evidence_html(sources), unsafe_allow_html=True)


@st.dialog("What KnowledgeForge can do", width="large")
def show_help():
    st.markdown(
        """
KnowledgeForge answers questions about **your own documents** — privately, on your
machine, with citations you can verify.

**Ask & answer**
- Ask in plain English; answers are grounded **only** in your uploaded documents.
- Answers **stream live** with inline citations like `[1]` `[2]`, plus an **Evidence**
  panel showing the exact source text behind each one.
- If something isn't in your documents, it says *"I couldn't find this…"* — it won't make things up.

**Your documents**
- Upload **many files at once** — PDF, Word (DOCX), PowerPoint (PPTX), Excel (XLSX/XLS), CSV, TXT, JSON.
- Search **across everything**, or focus on **one file** using *Ask about* in the sidebar.
- **Remove** any single file (and its data), or **Delete all**, from the sidebar.

**Smart retrieval**
- **Hybrid search** (meaning + keywords) finds the right passages.
- Results are **balanced across files**, so one big document can't drown out the rest.
- Every answer shows its **sources with page numbers**.

**Data questions (CSV / Excel)**
- Ask things like *"how many rows"*, *"total revenue"*, *"average price"*, *"top 5 by amount"*.

**Memory**
- Remembers the current conversation, so you can ask **follow-up questions**.

**Private & local**
- Runs entirely on your machine via **Ollama**. No API keys, no cloud — your files never leave your computer.

---

**How to use — 3 steps**
1. **Upload** one or more documents at the top of the page.
2. *(Optional)* In the sidebar **Ask about**, pick a single file to focus on.
3. **Type your question** and press Enter.

**Good to know**
- The **Ollama** engine must be running.
- Each answer uses the ~5 most relevant passages — ask focused questions for the sharpest results.
- Scanned / image-only PDFs may contain no readable text (no OCR).
        """
    )
    if st.button("Got it", use_container_width=True):
        st.rerun()


_COMMANDS_HELP = """**Slash commands**

- `/help` — open the full help panel
- `/commands` — show this list
- `/files` — list your indexed documents
- `/scope <file>` — focus answers on one file  ·  `/scope all` to reset
- `/mode fast` · `/mode accurate` — switch response speed vs depth
- `/clear` — start a new conversation
"""


def _safe_name(name):
    """Strip any path components and unsafe characters from an uploaded filename."""
    name = os.path.basename(name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    return name[:120] or "file"


def _disk_name(file_hash, display_name):
    """Deterministic on-disk name: hash-prefixed + sanitized, so distinct files
    with the same name never collide or overwrite each other."""
    return f"{file_hash[:12]}_{_safe_name(display_name)}"


def _append_cmd(question, answer):
    """Add a slash-command result to the transcript (no LLM, no sources)."""
    st.session_state.conversation_history.append({
        "question": question, "answer": answer, "sources": [],
        "is_analytics": False, "latency": 0, "is_command": True,
    })
    st.session_state.conversation_history = st.session_state.conversation_history[-10:]


def _handle_command(raw):
    """Parse and run a /slash command. Reruns or opens a dialog; never returns to Q&A."""
    parts = raw.strip().split(maxsplit=1)
    cmd = parts[0].lstrip("/").lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    names = [n for _, n in get_indexed_files()]

    if cmd in ("help", "?"):
        show_help()
        return
    if cmd in ("commands", "cmds", "command"):
        _append_cmd(raw, _COMMANDS_HELP)
        st.rerun()
    if cmd in ("clear", "new", "reset"):
        st.session_state.conversation_history = []
        st.toast("Started a new conversation")
        st.rerun()
    if cmd in ("files", "docs", "ls"):
        body = ("**Indexed documents**\n\n" + "\n".join(f"- {n}" for n in names)) if names \
            else "No documents indexed yet — upload some at the top of the page."
        _append_cmd(raw, body)
        st.rerun()
    if cmd == "scope":
        if arg.lower() in ("all", "*", ""):
            st.session_state.pending_scope = "All documents"
            st.rerun()
        match = next((n for n in names if arg.lower() in n.lower()), None)
        if match:
            st.session_state.pending_scope = match
            st.rerun()
        _append_cmd(raw, f"No indexed file matches **{arg}**. Try `/files` to see the list.")
        st.rerun()
    if cmd == "mode":
        a = arg.lower()
        if a.startswith("f"):
            st.session_state.pending_mode = "Fast · qwen2.5:3b"
            st.rerun()
        if a.startswith("a"):
            st.session_state.pending_mode = "Accurate · llama3.1:8b"
            st.rerun()
        _append_cmd(raw, "Usage: `/mode fast` or `/mode accurate`.")
        st.rerun()

    _append_cmd(raw, f"Unknown command `/{cmd}`. Type `/commands` to see what's available.")
    st.rerun()


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


# Response-mode options (label -> Ollama model). "Accurate" first = default.
RESPONSE_MODES = {
    "Accurate · llama3.1:8b": "llama3.1:8b",
    "Fast · qwen2.5:3b": "qwen2.5:3b",
}

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

# Apply any pending slash-command widget overrides BEFORE those widgets render
# (Streamlit forbids setting a widget's state after it's been instantiated).
for _pk, _wk in (("pending_scope", "scope_select"), ("pending_mode", "mode_select")):
    if _pk in st.session_state:
        st.session_state[_wk] = st.session_state.pop(_pk)

with st.sidebar:
    st.markdown(SIDEBAR_LOGO_HTML, unsafe_allow_html=True)

    if st.button("How to use", use_container_width=True):
        show_help()

    st.selectbox(
        "Response mode",
        list(RESPONSE_MODES.keys()),
        key="mode_select",
        help="Fast = quicker answers (qwen2.5:3b, fits your GPU). Accurate = deeper model (llama3.1:8b).",
    )

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
        st.selectbox(
            "Ask about",
            ["All documents"] + [n for _, n in indexed_files],
            key="scope_select",
            help="Focus answers on one file, or search across everything.",
        )
        with st.expander("Manage documents"):
            for fhash, fname in indexed_files:
                fcol, bcol = st.columns([4, 1])
                fcol.caption(fname)
                if bcol.button("Remove", key=f"del_{fhash}", use_container_width=True):
                    delete_documents(st.session_state.vectorstore, fhash)
                    delete_file(fhash)
                    st.session_state.uploaded_hashes.discard(fhash)
                    try:
                        os.remove(os.path.join(UPLOAD_DIR, _disk_name(fhash, fname)))
                    except OSError:
                        pass
                    st.toast(f"Removed {fname}")
                    st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear history", use_container_width=True):
            clear_history()
            st.session_state.conversation_history = []
            st.toast("History cleared")
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
                    st.toast("All documents deleted")
                    st.rerun()
                except Exception:
                    traceback.print_exc()
                    st.error("Couldn't delete everything. Check the server log for details.")
        with col2:
            if st.button("Cancel", use_container_width=True, key="cancel_delete"):
                st.session_state.show_delete_warning = False
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
        data = uploaded_file.getvalue()

        # Reject oversized files before parsing (memory / decompression-bomb guard).
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"{uploaded_file.name} is larger than the {MAX_UPLOAD_MB} MB limit.")
            continue

        file_hash = hashlib.sha256(data).hexdigest()

        # Skip if already uploaded
        if file_hash in st.session_state.uploaded_hashes:
            st.info(f"{uploaded_file.name} already indexed")
            continue

        # Save under a sanitized, hash-prefixed name so a crafted or duplicate
        # filename can't escape the uploads folder or overwrite another file.
        file_path = os.path.join(UPLOAD_DIR, _disk_name(file_hash, uploaded_file.name))
        with open(file_path, "wb") as f:
            f.write(data)

        # file processing
        with st.spinner(f"Indexing {uploaded_file.name}…"):
            try:
                docs = load_file(file_path)
                add_documents(st.session_state.vectorstore, docs, file_hash=file_hash)
                st.session_state.uploaded_hashes.add(file_hash)
                save_file(file_hash, uploaded_file.name)
                st.success(f"Indexed {uploaded_file.name} — {len(docs)} section(s)")
            except Exception:
                traceback.print_exc()  # full detail to the server log, not the user
                st.error(f"Couldn't process {uploaded_file.name}. It may be corrupt, empty, or an unsupported layout.")
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
# st.chat_input pins to the bottom regardless of where it's called, so read it
# first and use its value to decide what else to render this run.
typed = st.chat_input("Ask a question about your documents…   ·   type / for commands")
incoming = typed or st.session_state.pop("pending_question", None)

# Empty state + one-tap starter questions — only when there's nothing to show
# and we're not about to answer a question.
if not st.session_state.conversation_history and not incoming:
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

# Replay the existing transcript (static).
for exchange in st.session_state.conversation_history:
    if exchange.get("question"):
        with st.chat_message("user"):
            st.write(exchange["question"])
    with st.chat_message("assistant", avatar=_GEM_AVATAR):
        st.markdown(exchange["answer"], unsafe_allow_html=True)
        if not exchange.get("is_command"):
            _render_extras(
                exchange.get("latency", 0),
                exchange.get("is_analytics", False),
                exchange.get("sources"),
            )

# Answer a new question with live token streaming.
if incoming:
    if incoming.strip().startswith("/"):
        _handle_command(incoming)  # reruns or opens a dialog; never falls through
    elif len(incoming.strip()) < 3:
        st.warning("Please enter a longer question.")
    else:
        with st.chat_message("user"):
            st.write(incoming)
        _scope_sel = st.session_state.get("scope_select", "All documents")
        scope = None if _scope_sel == "All documents" else _scope_sel
        _mode_sel = st.session_state.get("mode_select", "Accurate · llama3.1:8b")
        model = RESPONSE_MODES.get(_mode_sel)
        start_time = time.time()
        with st.chat_message("assistant", avatar=_GEM_AVATAR):
            # Live "searching" shimmer while we retrieve + wait for the first token.
            thinking = st.empty()
            thinking.markdown(THINKING_HTML, unsafe_allow_html=True)
            res = prepare_answer(
                st.session_state.vectorstore,
                incoming,
                conversation_history=st.session_state.conversation_history,
                scope=scope,
                model=model,
            )
            if res["mode"] == "stream":
                box = st.empty()
                buf = ""
                try:
                    for i, tok in enumerate(res["stream"]):
                        if i == 0:
                            thinking.empty()  # first token arrived — drop the shimmer
                        buf += tok
                        # Blinking cursor while streaming.
                        box.markdown(buf + ' <span class="kf-cursor"></span>', unsafe_allow_html=True)
                except Exception:
                    traceback.print_exc()
                    buf = buf or "Something went wrong while generating the answer. Is Ollama still running?"
                thinking.empty()
                answer = buf.strip() or "I couldn't generate a response."
                box.markdown(answer, unsafe_allow_html=True)  # final render, no cursor
            else:
                thinking.empty()
                answer = res["text"]
                st.markdown(answer)
            latency = round(time.time() - start_time, 2)
            _render_extras(latency, res["is_analytics"], res["sources"])
        try:
            save_chat(incoming, answer)
        except Exception:
            pass
        st.session_state.conversation_history.append({
            "question": incoming,
            "answer": answer,
            "sources": res["sources"],
            "is_analytics": res["is_analytics"],
            "latency": latency,
        })
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]


# --- Slash-command palette: live suggestions when you type "/" in the chat box.
# Injected via a same-origin iframe that reaches into the parent document to
# attach a filtered popover to Streamlit's chat textarea.
_PALETTE_JS = """
<script>
(function(){
  const pw = window.parent, doc = pw.document;
  pw.__KF_CMDS = __CMDS__;
  function ensurePop(){
    let p = doc.getElementById("kf-pop");
    if(!p){
      p = doc.createElement("div");
      p.id = "kf-pop";
      p.style.cssText = "position:fixed;z-index:1000000;display:none;max-height:280px;overflow:auto;background:#121A2B;border:1px solid #2E3852;border-radius:12px;padding:6px;box-shadow:0 18px 44px -14px rgba(0,0,0,.75);font-family:Inter,system-ui,sans-serif;";
      doc.body.appendChild(p);
    }
    return p;
  }
  function setVal(ta, v){
    const s = Object.getOwnPropertyDescriptor(pw.HTMLTextAreaElement.prototype, "value").set;
    s.call(ta, v);
    ta.dispatchEvent(new Event("input", {bubbles:true}));
    ta.focus();
  }
  function bind(){
    const ta = doc.querySelector('[data-testid="stChatInput"] textarea');
    if(!ta){ return setTimeout(bind, 300); }
    if(ta.dataset.kfPal === "1") return;
    ta.dataset.kfPal = "1";
    const pop = ensurePop();
    let items = [], active = 0;
    function hide(){ pop.style.display = "none"; }
    function draw(){
      pop.innerHTML = items.map(function(c, i){
        return '<div data-i="'+i+'" style="display:flex;gap:10px;align-items:baseline;padding:8px 10px;border-radius:8px;cursor:pointer;'+(i===active?'background:#1B2540;':'')+'">'
          + '<span style="color:#818CF8;font-weight:700;font-size:.85rem;white-space:nowrap;">'+c.c+'</span>'
          + '<span style="color:#8B96B0;font-size:.78rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'+c.d+'</span></div>';
      }).join("");
      pop.querySelectorAll("[data-i]").forEach(function(el){
        el.addEventListener("mouseenter", function(){ active = +el.dataset.i; draw(); });
        el.addEventListener("mousedown", function(e){ e.preventDefault(); setVal(ta, items[+el.dataset.i].c); hide(); });
      });
    }
    function refresh(){
      const v = ta.value;
      if(v.charAt(0) !== "/"){ hide(); return; }
      const q = v.toLowerCase();
      const CMDS = pw.__KF_CMDS || [];
      items = CMDS.filter(function(c){
        const lc = c.c.toLowerCase();
        if(lc.indexOf(q) === 0) return true;
        if(q.indexOf(lc) === 0) return true;
        if(q.indexOf("/scope ") === 0 && lc.indexOf(q.slice(7).trim()) >= 0) return true;
        return false;
      });
      if(!items.length){ hide(); return; }
      if(active >= items.length) active = 0;
      draw();
      const r = ta.getBoundingClientRect();
      pop.style.left = r.left + "px";
      pop.style.width = Math.max(r.width, 280) + "px";
      pop.style.bottom = (pw.innerHeight - r.top + 10) + "px";
      pop.style.display = "block";
    }
    ta.addEventListener("input", function(){ active = 0; refresh(); });
    ta.addEventListener("keydown", function(e){
      if(pop.style.display !== "block") return;
      if(e.key === "ArrowDown"){ e.preventDefault(); active = (active+1)%items.length; draw(); }
      else if(e.key === "ArrowUp"){ e.preventDefault(); active = (active-1+items.length)%items.length; draw(); }
      else if(e.key === "Tab"){ e.preventDefault(); setVal(ta, items[active].c); hide(); }
      else if(e.key === "Escape"){ hide(); }
    });
    ta.addEventListener("blur", function(){ setTimeout(hide, 150); });
    ta.addEventListener("focus", function(){ if(ta.value.charAt(0) === "/") refresh(); });
  }
  bind();
})();
</script>
"""

_pal_files = [n for _, n in get_indexed_files()]
_pal_cmds = [
    {"c": "/help", "d": "Open the help panel"},
    {"c": "/commands", "d": "List all commands"},
    {"c": "/files", "d": "List your documents"},
    {"c": "/clear", "d": "Start a new conversation"},
    {"c": "/mode fast", "d": "Fast responses (qwen2.5:3b)"},
    {"c": "/mode accurate", "d": "Deeper answers (llama3.1:8b)"},
    {"c": "/scope all", "d": "Search all documents"},
] + [{"c": "/scope " + n, "d": "Focus on this file"} for n in _pal_files]

st.iframe(_PALETTE_JS.replace("__CMDS__", json.dumps(_pal_cmds)), height=1)
