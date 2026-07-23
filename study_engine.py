"""Socratic study mode: quiz the student from their own documents.

Reuses the existing retrieval stack — this module only adds question generation,
answer grading against the source passage, and PDF passage highlighting.
"""
import json
import random

import fitz  # PyMuPDF

from rag_engine import get_llm


def pick_chunk(vectorstore, source=None, exclude_ids=None, min_len=180):
    """Pick a substantial chunk to quiz on, optionally scoped to one file.

    Returns (chunk_id, text, metadata) or None. Prefers unseen, meaty chunks;
    relaxes those constraints if nothing else is available.
    """
    exclude_ids = exclude_ids or set()
    data = vectorstore.get()
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    def candidates(strict):
        for cid, text, meta in zip(ids, docs, metas):
            meta = meta or {}
            if source and meta.get("source") != source:
                continue
            body = (text or "").strip()
            if not body:
                continue
            if strict and (cid in exclude_ids or len(body) < min_len):
                continue
            yield (cid, text, meta)

    pool = list(candidates(strict=True)) or list(candidates(strict=False))
    return random.choice(pool) if pool else None


_QGEN = """You are an expert tutor. From the PASSAGE below, write ONE clear, specific question that checks whether a student truly understood its key idea.

Rules:
- The question must be fully answerable using only this passage.
- Prefer conceptual questions (why / how / what does X mean) over trivia.
- Output ONLY the question text. No preamble, no answer, no quotes.

PASSAGE:
{chunk}

QUESTION:"""


def generate_question(chunk_text, model=None):
    """Generate a single comprehension question from a passage."""
    llm = get_llm(model)
    raw = llm.invoke(_QGEN.format(chunk=chunk_text[:2500])).content.strip()
    question = raw.split("\n")[0].strip().strip('"').strip()
    return question or "What is the main idea of this passage?"


_GRADE = """You are a Socratic tutor grading a student's answer. The PASSAGE is the ONLY source of truth.

PASSAGE:
{chunk}

QUESTION: {question}
STUDENT ANSWER: {answer}

Judge how well the student's answer matches the passage. Reply with ONLY a JSON object and nothing else:
{{"verdict": "correct" | "partial" | "incorrect",
  "feedback": "1-2 encouraging sentences addressed to the student (\\"You...\\")",
  "missed": "the key point they missed or got wrong; empty string if fully correct",
  "quote": "the exact sentence(s) copied verbatim from the passage that answer the question"}}"""


def grade_answer(question, answer, chunk_text, model=None):
    """Grade a student's answer against the passage. Returns a normalized dict."""
    llm = get_llm(model)
    raw = llm.invoke(
        _GRADE.format(chunk=chunk_text[:2500], question=question, answer=answer)
    ).content.strip()

    result = _parse_json(raw) or {"verdict": "partial", "feedback": raw[:400], "missed": "", "quote": ""}

    v = str(result.get("verdict", "")).lower()
    if "incorrect" in v or "wrong" in v:
        result["verdict"] = "incorrect"
    elif "partial" in v:
        result["verdict"] = "partial"
    elif "correct" in v:
        result["verdict"] = "correct"
    else:
        result["verdict"] = "partial"

    for key in ("feedback", "missed", "quote"):
        result[key] = str(result.get(key, "") or "")
    return result


def _parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def highlight_pdf(file_path, page_no, quote, dpi=130):
    """Render the source PDF page with the supporting quote highlighted -> PNG bytes.

    Best-effort: if the quote can't be located, returns the page without a
    highlight; if anything fails, returns None so the caller can fall back to text.
    """
    try:
        doc = fitz.open(file_path)
        page = doc[int(page_no) - 1]
        snippet = (quote or "").strip()[:90]
        if snippet:
            for rect in page.search_for(snippet):
                page.add_highlight_annot(rect)
        png = page.get_pixmap(dpi=dpi).tobytes("png")
        doc.close()
        return png
    except Exception:
        return None
