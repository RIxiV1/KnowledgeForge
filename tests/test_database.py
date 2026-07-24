"""Tests for the SQLite layer, focused on study-attempt persistence (offline)."""
import database


def _fresh_db(tmp_path, monkeypatch):
    """Point the database module at a throwaway file and create its tables.

    get_connection() reads DATABASE_PATH from module globals at call time, so
    patching the attribute is enough — no module reload, no cross-test leakage.
    """
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "test.db"))
    database.create_tables()
    return database


def test_study_stats_empty(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    assert db.get_study_stats() == {"asked": 0, "correct": 0, "partial": 0, "incorrect": 0}


def test_study_attempts_accumulate(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.save_study_attempt("What is ATP?", "correct", "", "bio.pdf", 3)
    db.save_study_attempt("Define osmosis", "partial", "tonicity", "bio.pdf", 5)
    db.save_study_attempt("What is mitosis?", "incorrect", "phases", "bio.pdf", None)

    stats = db.get_study_stats()
    assert stats == {"asked": 3, "correct": 1, "partial": 1, "incorrect": 1}


def test_study_attempt_allows_null_page(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    # A non-PDF source has no page — must persist without error.
    db.save_study_attempt("Explain recursion", "correct", "", "notes.txt", None)
    assert db.get_study_stats()["correct"] == 1
