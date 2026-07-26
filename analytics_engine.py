import logging
import os
import re
from functools import lru_cache

import pandas as pd

log = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _load_df(path, _mtime, ext):
    """Cached dataframe load, keyed on path + mtime so file edits invalidate it."""
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def is_analytic_question(question):
    """
    Detect if a question is asking for data analytics/aggregation.

    Matches whole words, not substrings, so ordinary questions aren't misrouted
    to the spreadsheet path -- e.g. "summarize" must not trigger on the "sum"
    inside it, "meaning" must not trigger on "mean".
    """
    q = question.lower()
    single_word = {
        "count", "total", "sum", "average", "avg", "mean", "highest", "lowest",
        "most", "least", "top", "bottom", "maximum", "minimum", "spent",
        "revenue", "profit", "percentage", "compare", "trend", "distribution",
        "calculate", "rank", "sort", "rows", "records", "entries", "unique",
    }
    phrases = ["group by", "how many", "per year", "per month"]
    words = set(re.findall(r"[a-z]+", q))
    return bool(words & single_word) or any(p in q for p in phrases)


def _wb(word, q):
    """Whole-word (or exact-phrase) match of `word` in the lowercased question."""
    if " " in word:
        return word in q
    return re.search(r"\b" + re.escape(word) + r"\b", q) is not None


def _intent(q, *keys):
    return any(_wb(k, q) for k in keys)


def _pick_numeric_col(numeric_cols, q):
    """Pick which numeric column the question is about.

    1) a column whose name is named in the question (whole word);
    2) else a numeric column whose NAME contains a money-ish synonym;
    3) else the first numeric column.

    The old code tested the synonyms against the QUESTION, so any question
    containing "price"/"amount"/... returned numeric_cols[0] regardless of the
    real column asked for (e.g. "average price" -> mean of `id`).
    """
    for col in numeric_cols:
        if _wb(col.lower(), q):
            return col
    synonyms = ("price", "amount", "cost", "revenue", "sales", "value",
                "total", "qty", "quantity", "spend", "spent")
    for col in numeric_cols:
        if any(s in col.lower() for s in synonyms):
            return col
    return numeric_cols[0]


def analyze_dataframe(file_path, question):
    """
    Analyze a tabular file with deterministic pandas ops (no LLM code-gen).

    Returns a formatted answer string on success, or None when this path can't
    produce a real answer (non-tabular file, empty data, no matching column, or
    an error). Returning None lets the caller fall through to normal document
    Q&A instead of surfacing an internal message like "No numeric columns" to
    the user as if it were the answer.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".csv", ".xlsx", ".xls"):
            return None
        df = _load_df(file_path, os.path.getmtime(file_path), ext)
        if df.empty:
            return None
        q = question.lower()

        # COUNT — note "total" is intentionally NOT here: "total revenue" means
        # a sum (handled below), not a record count.
        if _intent(q, "how many", "count", "rows", "records", "entries"):
            if _wb("unique", q):
                for col in df.columns:
                    if _wb(str(col).lower(), q):
                        return f"Unique {col}: {df[col].nunique():,}"
                first = df.columns[0]
                return f"Unique values in {first}: {df[first].nunique():,}"
            return f"Total Records: {len(df):,}"

        # SUM / TOTAL
        if _intent(q, "sum", "total", "all together"):
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                return None
            col = _pick_numeric_col(numeric_cols, q)
            return f"Total {col}: {df[col].sum():,.2f}"

        # AVERAGE / MEAN
        if _intent(q, "average", "avg", "mean"):
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                return None
            col = _pick_numeric_col(numeric_cols, q)
            return f"Average {col}: {df[col].mean():,.2f}"

        # MAXIMUM
        if _intent(q, "maximum", "max", "highest", "largest"):
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                return None
            col = _pick_numeric_col(numeric_cols, q)
            return f"Maximum {col}: {df[col].max():,.2f}"

        # MINIMUM
        if _intent(q, "minimum", "min", "lowest", "smallest"):
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                return None
            col = _pick_numeric_col(numeric_cols, q)
            return f"Minimum {col}: {df[col].min():,.2f}"

        # TOP / BOTTOM N
        if _wb("top", q) or _wb("bottom", q):
            # Only take N right after top/bottom, so "top 5 from 2023" -> 5, not 2023.
            match = re.search(r"\b(?:top|bottom)\s+(\d+)", q)
            n = int(match.group(1)) if match else 5
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                return None
            col = _pick_numeric_col(numeric_cols, q)
            if _wb("top", q) or _wb("highest", q):
                return f"Top {n} {col}:\n{df.nlargest(n, col)[col].to_string()}"
            return f"Bottom {n} {col}:\n{df.nsmallest(n, col)[col].to_string()}"

        # GROUP BY
        if set(re.findall(r"[a-z]+", q)) & {"by", "per", "group"}:
            for col in df.columns:
                if _wb(str(col).lower(), q):
                    return f"Breakdown by {col}:\n{df[col].value_counts().to_string()}"
            return None

        # Fallback: overall numeric stats, else a column summary.
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            return f"Statistics:\n{df[numeric_cols].describe().to_string()}"
        return f"Dataset has {len(df):,} records with columns: {', '.join(map(str, df.columns.tolist()))}"

    except Exception:
        log.exception("Analytics failed for %s; falling back to document Q&A", file_path)
        return None
