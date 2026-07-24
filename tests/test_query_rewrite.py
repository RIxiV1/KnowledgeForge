"""Tests for follow-up query rewriting (offline parts only — no Ollama).

The LLM rewrite itself needs a running model, so here we only exercise the
cheap heuristic and the guard paths that must NOT call the model.
"""
import rag_engine


def test_followup_heuristic_flags_elliptical_and_pronoun_questions():
    assert rag_engine._looks_like_followup("and its price?")
    assert rag_engine._looks_like_followup("what about the second one?")
    assert rag_engine._looks_like_followup("why?")
    assert rag_engine._looks_like_followup("How does it compare to that?")


def test_followup_heuristic_ignores_standalone_questions():
    assert not rag_engine._looks_like_followup("Summarize the key points")
    assert not rag_engine._looks_like_followup(
        "What is the detailed cellular respiration process in eukaryotic organisms?"
    )


def test_standalone_passthrough_with_no_history_does_not_call_model(monkeypatch):
    # If the model were called, this would blow up — proving the guard short-circuits.
    monkeypatch.setattr(rag_engine, "get_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")))
    assert rag_engine._standalone_question("and its price?", []) == "and its price?"


def test_standalone_passthrough_when_not_a_followup(monkeypatch):
    monkeypatch.setattr(rag_engine, "get_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")))
    history = [{"question": "Tell me about mitochondria", "answer": "They make ATP."}]
    q = "What is the detailed cellular respiration process in eukaryotic organisms?"
    assert rag_engine._standalone_question(q, history) == q


def test_rewrite_rejects_junk_output(monkeypatch):
    # A model that returns a multi-line paragraph should be rejected -> original kept.
    class _Fake:
        def invoke(self, _prompt):
            class _R:
                content = "Here is the rewritten query:\nwhat is the structure?"
            return _R()

    monkeypatch.setattr(rag_engine, "get_llm", lambda *a, **k: _Fake())
    history = [{"question": "What does the mitochondria do?", "answer": "Makes ATP."}]
    assert rag_engine._standalone_question("what about its structure?", history) == "what about its structure?"


def test_rewrite_accepts_clean_output(monkeypatch):
    class _Fake:
        def invoke(self, _prompt):
            class _R:
                content = '"what is the structure of a mitochondrion?"'
            return _R()

    monkeypatch.setattr(rag_engine, "get_llm", lambda *a, **k: _Fake())
    history = [{"question": "What does the mitochondria do?", "answer": "Makes ATP."}]
    out = rag_engine._standalone_question("what about its structure?", history)
    assert out == "what is the structure of a mitochondrion?"
