"""Tests for analytics_engine (offline, no Ollama)."""
import analytics_engine


def test_is_analytic_question_true():
    assert analytics_engine.is_analytic_question("how many rows")
    assert analytics_engine.is_analytic_question("total revenue")
    assert analytics_engine.is_analytic_question("top 5 products")


def test_is_analytic_question_false():
    assert not analytics_engine.is_analytic_question("summarize this document")
    assert not analytics_engine.is_analytic_question("what is the meaning of X")


def _write_csv(tmp_path):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "name,amount,year\n"
        "Alice,100,2023\n"
        "Bob,200,2024\n"
        "Carol,300,2023\n",
        encoding="utf-8",
    )
    return str(csv_path)


def test_analyze_how_many_rows(tmp_path):
    path = _write_csv(tmp_path)
    out = analytics_engine.analyze_dataframe(path, "how many rows")
    assert "Total Records" in out


def test_analyze_top_n_no_error(tmp_path):
    path = _write_csv(tmp_path)
    out = analytics_engine.analyze_dataframe(path, "top 3 by amount")
    assert not out.lower().startswith("error")
    assert "Top 3" in out


def test_top_n_regex_fix(tmp_path):
    """'top 5 items from 2023' must pick N=5, not the year 2023."""
    path = _write_csv(tmp_path)
    out = analytics_engine.analyze_dataframe(path, "top 5 items from 2023")
    assert "Top 5" in out
    assert "Top 2023" not in out


def test_non_tabular_returns_none(tmp_path):
    """A .txt (or any non-CSV/Excel) must return None, not an error string.

    Returning None lets the caller fall through to document Q&A instead of
    surfacing 'Unsupported file format for analytics' to the user as the answer.
    """
    txt = tmp_path / "notes.txt"
    txt.write_text("Some prose with the word entries in it.", encoding="utf-8")
    assert analytics_engine.analyze_dataframe(str(txt), "how many entries") is None


def test_empty_dataframe_returns_none(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("col_a,col_b\n", encoding="utf-8")  # header only, no rows
    assert analytics_engine.analyze_dataframe(str(empty), "how many rows") is None


def test_missing_file_returns_none(tmp_path):
    """An unreadable path must be swallowed to None, not raise or leak an error."""
    assert analytics_engine.analyze_dataframe(str(tmp_path / "nope.csv"), "count") is None
