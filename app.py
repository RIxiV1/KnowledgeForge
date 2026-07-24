import os
import re
import shutil
import hashlib
import time
import html
import json
import logging
import traceback
import urllib.request
import streamlit as st
from config import MAX_UPLOAD_MB
from document_loader import load_file
from vector_store import get_vectorstore
from rag_engine import add_documents, prepare_answer, delete_documents, reset_vectorstore
from study_engine import pick_chunk, generate_question, grade_answer, highlight_pdf
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
    save_study_attempt,
    get_study_stats,
)

log = logging.getLogger(__name__)

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
/* Study-mode question card */
.kf-qcard{ background:linear-gradient(180deg,#17203a,#131a2e); border:1px solid #2c3a63;
  border-radius:16px; padding:20px 22px; font-size:1.18rem; font-weight:600; color:#EAEEF9;
  line-height:1.5; box-shadow:var(--shadow); margin-bottom:14px; }
a{ color:var(--accent-2); text-decoration:none; }
a:hover{ text-decoration:underline; }
hr{ border-color:var(--border); margin:16px 0; }
::selection{ background:rgba(99,102,241,.35); }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:#26304a; border-radius:8px; border:2px solid transparent; background-clip:padding-box; }
::-webkit-scrollbar-thumb:hover{ background:#37456A; }
/* Hide the Streamlit chrome we don't want (menu, footer, deploy toolbar,
   status widget) — but do NOT hide the whole `header`: the control that
   REOPENS a collapsed sidebar (stSidebarCollapsed) lives up there, and hiding
   the header made a collapsed sidebar impossible to bring back. */
footer,
#MainMenu,
[data-testid="stMainMenu"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"]{ visibility:hidden; }
header[data-testid="stHeader"]{ background:transparent; }
/* Belt-and-suspenders: always keep the sidebar expand/collapse controls shown
   (stExpandSidebarButton is the one that REOPENS a collapsed sidebar). */
[data-testid="stSidebarCollapsed"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"]{ visibility:visible !important; opacity:1 !important; z-index:1000001; }
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
        page, rows, section = s.get("page"), s.get("rows"), s.get("section")
        key = (name, page, rows, section)
        if key in seen:
            continue
        seen.add(key)
        bits = []
        if page:
            bits.append(f"page {html.escape(str(page))}")
        if section:
            bits.append(html.escape(str(section)))
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
        if s.get("section"):
            head += f" · {html.escape(str(s['section']))}"
        full = s.get("text") or ""
        body = html.escape(full[:700]) + (" …" if len(full) > 700 else "")
        rows.append(f'<div class="kf-ev"><div class="kf-ev-h">{head}</div><div class="kf-ev-b">{body}</div></div>')
    return '<div class="kf-ev-wrap">' + "".join(rows) + "</div>"


def _render_extras(latency, is_analytics, sources, confidence=None):
    """Metric chips + Sources + Evidence, shared by live and replayed messages."""
    st.markdown(_meta_html(latency, is_analytics, len(sources or []), confidence), unsafe_allow_html=True)
    if sources:
        with st.expander("Sources"):
            st.markdown(_sources_html(sources), unsafe_allow_html=True)
        with st.expander("Evidence — the exact passages used"):
            st.markdown(_evidence_html(sources), unsafe_allow_html=True)


STUDY_INTRO_HTML = """
<div style="text-align:center;padding:20px 0 8px;">
  <div style="font-size:1.15rem;font-weight:700;color:#EAEEF9;">Learn by explaining, not just reading</div>
  <div style="color:#93A0B8;font-size:.92rem;margin-top:6px;max-width:54ch;margin-inline:auto;line-height:1.55;">
    KnowledgeForge quizzes you from your own material, checks your answer against the source,
    and shows you the exact passage to revisit — active recall, powered by your documents.
  </div>
</div>
"""


def _verdict_html(verdict, feedback):
    styles = {
        "correct": ("#22c55e", "Correct"),
        "partial": ("#f59e0b", "Partly right"),
        "incorrect": ("#ef4444", "Not quite"),
    }
    color, label = styles.get(verdict, ("#818CF8", "Feedback"))
    return (
        '<div style="display:flex;gap:11px;align-items:flex-start;margin:8px 0 6px;">'
        f'<span style="background:{color}22;color:{color};border:1px solid {color}66;'
        'padding:4px 12px;border-radius:999px;font-weight:700;font-size:.82rem;white-space:nowrap;">'
        f'{label}</span>'
        f'<span style="color:var(--text);line-height:1.55;">{html.escape(feedback)}</span></div>'
    )


def _study_new_question(scope, model):
    """Pick a fresh chunk and generate a question from it, then rerun."""
    used = st.session_state.setdefault("study_used_ids", set())
    with st.spinner("Finding something to quiz you on…"):
        picked = pick_chunk(st.session_state.vectorstore, source=scope, exclude_ids=used)
        if not picked:
            st.toast("No content to study yet — upload a document first.")
            return
        cid, text, meta = picked
        try:
            question = generate_question(text, model)
        except Exception:
            traceback.print_exc()
            st.toast("Couldn't reach the model — is Ollama running?")
            return
        used.add(cid)
    st.session_state.study_chunk = {"id": cid, "text": text, "meta": meta}
    st.session_state.study_question = question
    st.session_state.study_phase = "question"
    st.session_state.pop("study_result", None)
    st.session_state.pop("study_png", None)
    st.rerun()


def render_study():
    """Socratic study mode: quiz the student from their own documents."""
    files = [n for _, n in get_indexed_files()]
    if not files:
        st.info("Upload a document above, then come back to Study mode to be quizzed on it.")
        return

    stats = st.session_state.setdefault(
        "study_stats", {"asked": 0, "correct": 0, "partial": 0, "incorrect": 0}
    )
    records = st.session_state.setdefault("study_records", [])
    phase = st.session_state.get("study_phase")

    if phase == "summary":
        _render_summary(stats, records)
        return

    study_doc = st.selectbox("Study from", ["All documents"] + files, key="study_doc")
    scope = None if study_doc == "All documents" else study_doc
    model = RESPONSE_MODES.get(st.session_state.get("mode_select", "Accurate · llama3.1:8b"))

    m1, m2, m3 = st.columns(3)
    m1.metric("Questions", stats["asked"])
    m2.metric("Correct", stats["correct"])
    acc = round(100 * stats["correct"] / stats["asked"]) if stats["asked"] else 0
    m3.metric("Accuracy", f"{acc}%")

    if stats["asked"]:
        if st.button("Finish & review session", use_container_width=True):
            st.session_state.study_phase = "summary"
            st.rerun()

    if not phase:
        st.markdown(STUDY_INTRO_HTML, unsafe_allow_html=True)
        if st.button("Start studying", type="primary", use_container_width=True):
            _study_new_question(scope, model)
        return

    st.markdown(
        f'<div class="kf-qcard">{html.escape(st.session_state.study_question)}</div>',
        unsafe_allow_html=True,
    )
    chunk = st.session_state.study_chunk

    if phase == "question":
        answer = st.text_area(
            "Your answer", key=f"ans_{chunk['id']}", height=130,
            placeholder="Explain it in your own words…",
        )
        c1, c2 = st.columns([3, 1])
        if c1.button("Submit answer", type="primary", use_container_width=True):
            if not answer.strip():
                st.toast("Write an answer first.")
            else:
                try:
                    with st.spinner("Checking your understanding…"):
                        res = grade_answer(st.session_state.study_question, answer, chunk["text"], model)
                except Exception:
                    traceback.print_exc()
                    st.toast("Couldn't reach the model — is Ollama running?")
                    st.stop()
                meta = chunk["meta"]
                png = None
                if meta.get("file_type") == "pdf" and meta.get("page") and meta.get("file_path"):
                    png = highlight_pdf(meta["file_path"], meta["page"], res.get("quote", ""))
                st.session_state.study_result = res
                st.session_state.study_png = png
                st.session_state.study_answer_shown = answer
                stats["asked"] += 1
                stats[res["verdict"]] = stats.get(res["verdict"], 0) + 1
                records.append({
                    "question": st.session_state.study_question,
                    "verdict": res["verdict"],
                    "missed": res.get("missed", ""),
                    "source": meta.get("source", ""),
                    "page": meta.get("page"),
                })
                # Persist so all-time progress survives restarts (best-effort:
                # a DB hiccup must never break the study flow).
                try:
                    save_study_attempt(
                        st.session_state.study_question, res["verdict"],
                        res.get("missed", ""), meta.get("source", ""), meta.get("page"),
                    )
                except Exception:
                    log.exception("Failed to persist study attempt")
                st.session_state.study_phase = "feedback"
                st.rerun()
        if c2.button("Skip", use_container_width=True):
            _study_new_question(scope, model)

    elif phase == "feedback":
        res = st.session_state.study_result
        st.caption("Your answer")
        st.markdown(f"> {html.escape(st.session_state.get('study_answer_shown', ''))}")
        st.markdown(_verdict_html(res["verdict"], res["feedback"]), unsafe_allow_html=True)
        if res.get("missed"):
            st.markdown(f"**Revisit:** {html.escape(res['missed'])}")

        meta = chunk["meta"]
        page = meta.get("page")
        st.markdown(f"**Source — {html.escape(meta.get('source', 'source'))}"
                    + (f" · page {page}" if page else "") + "**")
        png = st.session_state.get("study_png")
        if png is not None:
            st.image(png, use_container_width=True)
        else:
            body = res.get("quote") or chunk["text"][:600]
            st.markdown(
                f'<div class="kf-ev"><div class="kf-ev-b">{html.escape(body)}</div></div>',
                unsafe_allow_html=True,
            )

        if st.button("Next question", type="primary", use_container_width=True):
            _study_new_question(scope, model)


def _reset_study():
    for key in ("study_phase", "study_chunk", "study_question", "study_result",
                "study_png", "study_answer_shown", "study_records", "study_stats",
                "study_used_ids"):
        st.session_state.pop(key, None)


def _render_summary(stats, records):
    """The demo closer: session score, breakdown, and what to revisit."""
    asked = stats.get("asked", 0)
    acc = round(100 * stats.get("correct", 0) / asked) if asked else 0
    total = max(asked, 1)

    st.markdown(
        '<div style="text-align:center;padding:12px 0 2px;">'
        f'<div style="opacity:.9;display:inline-block;">{_gem(40)}</div>'
        "<div style=\"font-family:'Plus Jakarta Sans',sans-serif;font-size:1.4rem;font-weight:800;"
        'color:#EAEEF9;margin-top:8px;">Session complete</div></div>',
        unsafe_allow_html=True,
    )

    def seg(n, color):
        return f'<div style="width:{100 * n / total:.1f}%;background:{color};"></div>'

    bar = ('<div style="display:flex;height:12px;border-radius:999px;overflow:hidden;'
           'background:#1b2540;margin:12px 0 6px;">'
           + seg(stats.get("correct", 0), "#22c55e")
           + seg(stats.get("partial", 0), "#f59e0b")
           + seg(stats.get("incorrect", 0), "#ef4444") + "</div>")
    legend = (f'<span style="color:#22c55e;">&#9679; {stats.get("correct", 0)} correct</span>'
              f'&nbsp;&nbsp;<span style="color:#f59e0b;">&#9679; {stats.get("partial", 0)} partial</span>'
              f'&nbsp;&nbsp;<span style="color:#ef4444;">&#9679; {stats.get("incorrect", 0)} to review</span>')
    st.markdown(
        "<div style=\"text-align:center;\"><span style=\"font-family:'Plus Jakarta Sans',sans-serif;"
        f'font-size:3rem;font-weight:800;color:#EAEEF9;">{acc}%</span>'
        f'<div style="color:#93A0B8;">accuracy over {asked} question(s)</div></div>'
        + bar + f'<div style="text-align:center;font-size:.83rem;">{legend}</div>',
        unsafe_allow_html=True,
    )

    # All-time progress across every session (persisted in SQLite).
    try:
        alltime = get_study_stats()
    except Exception:
        log.exception("Failed to load all-time study stats")
        alltime = None
    if alltime and alltime["asked"] > asked:
        at_acc = round(100 * alltime["correct"] / alltime["asked"]) if alltime["asked"] else 0
        st.markdown(
            '<div style="text-align:center;color:#6B7688;font-size:.85rem;margin-top:10px;">'
            f'All-time: <b style="color:#93A0B8;">{at_acc}%</b> accuracy over '
            f'{alltime["asked"]} question(s) across all sessions</div>',
            unsafe_allow_html=True,
        )

    revisit = [r for r in records if r.get("verdict") in ("incorrect", "partial")]
    if revisit:
        st.markdown("#### Revisit these")
        for r in revisit:
            color = "#ef4444" if r["verdict"] == "incorrect" else "#f59e0b"
            src = html.escape(r.get("source", "")) + (f" &middot; p.{r['page']}" if r.get("page") else "")
            missed = (f'<div style="color:#93A0B8;font-size:.85rem;margin-top:5px;">'
                      f'{html.escape(r["missed"])}</div>') if r.get("missed") else ""
            st.markdown(
                f'<div style="border:1px solid var(--border);border-left:3px solid {color};'
                'border-radius:11px;padding:12px 14px;margin-bottom:9px;background:var(--surface);">'
                f'<div style="color:#EAEEF9;font-weight:600;line-height:1.45;">{html.escape(r["question"])}</div>'
                f'{missed}<div style="color:#6B7688;font-size:.78rem;margin-top:6px;">{src}</div></div>',
                unsafe_allow_html=True,
            )
    elif asked:
        st.success("You answered everything well — nothing to revisit.")

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("Study again", type="primary", use_container_width=True):
        _reset_study()
        st.rerun()
    if c2.button("Keep studying", use_container_width=True):
        st.session_state.study_phase = None
        st.rerun()


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


@st.dialog("Conversation history", width="large")
def show_history_dialog():
    history = get_chat_history()
    if not history:
        st.info("No conversations yet.")
        return
    for idx, (question, answer, created_at) in enumerate(history, 1):
        st.caption(f"{idx}.  {created_at}")
        st.markdown(f"**Q:** {question}")
        st.markdown(f"**A:** {answer}" if answer else "**A:** _(no answer saved)_")
        st.divider()


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


def _content_type_ok(ext, data):
    """Sniff magic bytes so a file's real type matches its extension (a .exe
    renamed to .pdf is rejected). Text formats are allowed through."""
    head = data[:8]
    if ext == ".pdf":
        return head.startswith(b"%PDF")
    if ext in (".docx", ".xlsx", ".pptx"):
        return head.startswith(b"PK\x03\x04")   # modern Office = zip
    return True   # txt / csv / json / legacy .xls: text or hard to sniff safely


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


def _confidence_chip(confidence):
    """A colored dot + label showing how relevant the best passage was (0-1 → High/Medium/Low).

    Hidden when confidence is None (analytics answers, or retrieval couldn't score).
    """
    if confidence is None:
        return ""
    pct = max(0, min(100, round(confidence * 100)))
    if confidence >= 0.5:
        label, color = "High", "#37D399"
    elif confidence >= 0.3:
        label, color = "Medium", "#F5C451"
    else:
        label, color = "Low", "#F08A7A"
    return (
        '<span class="kf-chip" title="How closely the top retrieved passage matches your question">'
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
        f'background:{color};margin-right:5px;vertical-align:middle;"></span>'
        f'{label} confidence · {pct}%</span>'
    )


def _meta_html(latency, is_analytics, n_sources, confidence=None):
    """A small row of pill chips: latency · mode · source count · confidence."""
    mode = "Analytics" if is_analytics else "Document Q&A"
    return (
        '<div class="kf-meta">'
        f'<span class="kf-chip"><b>{latency}s</b></span>'
        f'<span class="kf-chip">{mode}</span>'
        f'<span class="kf-chip"><b>{n_sources}</b> source(s)</span>'
        f'{_confidence_chip(confidence)}'
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
        show_history_dialog()

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

@st.cache_data(ttl=10, show_spinner=False)
def _ollama_up():
    base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


st.markdown(HERO_HTML, unsafe_allow_html=True)

if not _ollama_up():
    st.error(
        "Can't reach Ollama (the local AI engine). Start it with `ollama serve` (or "
        "launch the Ollama app), then refresh. Uploads and answers won't work until it's running."
    )

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

        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if not _content_type_ok(ext, data):
            st.error(f"{uploaded_file.name} doesn't look like a valid {ext or 'file'} — its contents don't match the extension.")
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
                # Show the original filename as the source (file_path keeps the
                # hashed on-disk path, which highlighting/analytics still use).
                for d in docs:
                    d.metadata["source"] = uploaded_file.name
                add_documents(st.session_state.vectorstore, docs, file_hash=file_hash)
                st.session_state.uploaded_hashes.add(file_hash)
                save_file(file_hash, uploaded_file.name)
                st.success(f"Indexed {uploaded_file.name} — {len(docs)} section(s)")
            except Exception:
                traceback.print_exc()  # full detail to the server log, not the user
                st.error(f"Couldn't process {uploaded_file.name}. It may be corrupt, empty, or an unsupported layout.")

# --- Mode switch: Study (Socratic) is the hero; Chat is the Q&A fallback. ------
st.markdown('<div style="margin-top:4px;"></div>', unsafe_allow_html=True)
_mode = st.radio(
    "Mode", ["🎓 Study", "💬 Chat"], horizontal=True,
    label_visibility="collapsed", key="app_mode",
)
if _mode == "🎓 Study":
    render_study()
    st.stop()

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
                exchange.get("confidence"),
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
            _render_extras(latency, res["is_analytics"], res["sources"], res.get("confidence"))
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
            "confidence": res.get("confidence"),
        })
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]


# --- Autocomplete popover for the chat box (Gboard-style). Three modes:
#   "/"       -> slash commands       (e.g. /scope report.pdf)
#   "." or "@"-> file suggestions      (".pdf" lists PDFs, ".rep" matches by name)
#   else      -> question suggestions  (starters + file prompts + recent Qs)
# Injected via a same-origin iframe that reaches into the parent document to
# attach a filtered popover to Streamlit's chat textarea.
_PALETTE_JS = r"""
<script>
(function(){
  const pw = window.parent, doc = pw.document;
  pw.__KF_CMDS = __CMDS__;
  pw.__KF_FILES = __FILES__;
  pw.__KF_SUGG = __SUGG__;
  function esc(s){ return String(s).replace(/[&<>"]/g, function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]; }); }
  function ensurePop(){
    let p = doc.getElementById("kf-pop");
    if(!p){
      p = doc.createElement("div");
      p.id = "kf-pop";
      p.style.cssText = "position:fixed;z-index:1000000;display:none;max-height:300px;overflow:auto;background:#121A2B;border:1px solid #2E3852;border-radius:12px;padding:6px;box-shadow:0 18px 44px -14px rgba(0,0,0,.75);font-family:Inter,system-ui,sans-serif;";
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
    // A file mention starts a word with "." or "@" — e.g. ".rep" or "@rep".
    const MENTION = /(^|\s)[.@]([^\s]*)$/;
    let items = [], active = 0;
    function hide(){ pop.style.display = "none"; }
    function draw(){
      pop.innerHTML = items.map(function(it, i){
        const label = it.kind === 'cmd'
          ? '<span style="color:#818CF8;font-weight:700;">' + esc(it.label) + '</span>'
          : esc(it.label);
        const primary = '<span style="font-size:.9rem;color:#EAEEF9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;">' + label + '</span>';
        const hint = it.hint ? '<span style="color:#5A6685;font-size:.72rem;white-space:nowrap;">' + esc(it.hint) + '</span>' : '';
        return '<div data-i="'+i+'" style="display:flex;gap:10px;align-items:center;padding:8px 10px;border-radius:8px;cursor:pointer;'+(i===active?'background:#1B2540;':'')+'">' + primary + hint + '</div>';
      }).join("");
      pop.querySelectorAll("[data-i]").forEach(function(el){
        el.addEventListener("mouseenter", function(){ active = +el.dataset.i; draw(); });
        el.addEventListener("mousedown", function(e){ e.preventDefault(); accept(+el.dataset.i); });
      });
    }
    function accept(i){
      const it = items[i]; if(!it) return;
      if(it.kind === 'file'){
        const v = ta.value, m = v.match(MENTION);
        const start = m ? (m.index + m[1].length) : v.length;
        setVal(ta, v.slice(0, start) + it.c + ' ');
      } else {
        setVal(ta, it.c);
      }
      hide();
    }
    function show(){
      if(!items.length){ hide(); return; }
      if(active >= items.length) active = 0;
      draw();
      const r = ta.getBoundingClientRect();
      pop.style.left = r.left + "px";
      pop.style.width = Math.max(r.width, 300) + "px";
      pop.style.bottom = (pw.innerHeight - r.top + 10) + "px";
      pop.style.display = "block";
    }
    function refresh(){
      const v = ta.value;
      // 1) slash commands
      if(v.charAt(0) === "/"){
        const q = v.toLowerCase(), CMDS = pw.__KF_CMDS || [];
        items = CMDS.filter(function(c){
          const lc = c.c.toLowerCase();
          if(lc.indexOf(q) === 0) return true;
          if(q.indexOf(lc) === 0) return true;
          if(q.indexOf("/scope ") === 0 && lc.indexOf(q.slice(7).trim()) >= 0) return true;
          return false;
        }).map(function(c){ return {c:c.c, label:c.c, hint:c.d, kind:'cmd'}; });
        return show();
      }
      // 2) @file mention (anywhere in the text)
      const mm = v.match(MENTION);
      if(mm){
        const q = mm[2].toLowerCase(), FILES = pw.__KF_FILES || [];
        items = FILES.filter(function(f){ return f.toLowerCase().indexOf(q) >= 0; })
          .slice(0, 8).map(function(f){ return {c:f, label:f, hint:'file', kind:'file'}; });
        return show();
      }
      // 3) question suggestions (Gboard-style)
      const t = v.trim();
      if(t.length >= 2){
        const q = t.toLowerCase(), SUGG = pw.__KF_SUGG || [];
        items = SUGG.filter(function(s){ const ls = s.toLowerCase(); return ls !== q && ls.indexOf(q) >= 0; })
          .slice(0, 6).map(function(s){ return {c:s, label:s, hint:'', kind:'sugg'}; });
        return show();
      }
      hide();
    }
    ta.addEventListener("input", function(){ active = 0; refresh(); });
    ta.addEventListener("keydown", function(e){
      if(pop.style.display !== "block") return;
      if(e.key === "ArrowDown"){ e.preventDefault(); active = (active+1)%items.length; draw(); }
      else if(e.key === "ArrowUp"){ e.preventDefault(); active = (active-1+items.length)%items.length; draw(); }
      else if(e.key === "Tab"){ e.preventDefault(); accept(active); }
      else if(e.key === "Escape"){ hide(); }
    });
    ta.addEventListener("blur", function(){ setTimeout(hide, 150); });
    ta.addEventListener("focus", function(){ refresh(); });
  }

  // --- Guaranteed sidebar toggle -------------------------------------------
  // Streamlit's own reopen control can end up hidden by custom chrome CSS, so
  // we inject our own always-present floating button that clicks whichever of
  // Streamlit's expand/collapse buttons currently exists.
  function ensureToggle(){
    let btn = doc.getElementById("kf-sb-toggle");
    if(!btn){
      btn = doc.createElement("button");
      btn.id = "kf-sb-toggle";
      btn.type = "button";
      btn.title = "Show/hide sidebar";
      btn.innerHTML = "&#9776;";  // ☰
      btn.style.cssText = "position:fixed;top:10px;left:10px;z-index:1000003;width:40px;height:40px;border-radius:10px;border:1px solid #2E3852;background:#121A2B;color:#EAEEF9;font-size:19px;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 6px 18px -8px rgba(0,0,0,.7);";
      btn.addEventListener("click", function(e){
        e.preventDefault();
        const expand = doc.querySelector('[data-testid="stExpandSidebarButton"] button, [data-testid="stExpandSidebarButton"]');
        const collapse = doc.querySelector('[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"]');
        const target = expand || collapse;
        if(target){ target.click(); }
      });
      doc.body.appendChild(btn);
    }
    // Only show our button while the sidebar is collapsed (otherwise the sidebar
    // has its own visible collapse button).
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    const collapsed = !sb || sb.getAttribute("aria-expanded") === "false" || sb.offsetWidth < 40;
    btn.style.display = collapsed ? "flex" : "none";
  }

  bind();
  ensureToggle();
  setInterval(ensureToggle, 400);  // survive Streamlit reruns + keep in sync
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

# Question suggestions: generic starters + one-per-file prompts + your recent
# questions (deduped, persisted in SQLite so they carry across sessions). The
# browser filters these live by what you've typed.
_starters = [
    "Summarize the key points",
    "What problem does this solve?",
    "What are the main features?",
    "Explain this in simple terms",
    "List the main takeaways",
    "What are the strengths and weaknesses?",
]
_file_prompts = []
for _n in _pal_files:
    _file_prompts.append("Summarize " + _n)
    _file_prompts.append("What are the key points in " + _n + "?")
_recent, _seen = [], set()
try:
    for _q, _a, _ts in get_chat_history(limit=40):
        _qs = (_q or "").strip()
        _k = _qs.lower()
        if _qs and not _qs.startswith("/") and _k not in _seen:
            _seen.add(_k)
            _recent.append(_qs)
except Exception:
    pass
_pal_sugg = _recent[:12] + _starters + _file_prompts

st.iframe(
    _PALETTE_JS
    .replace("__CMDS__", json.dumps(_pal_cmds))
    .replace("__FILES__", json.dumps(_pal_files))
    .replace("__SUGG__", json.dumps(_pal_sugg)),
    height=1,
)
