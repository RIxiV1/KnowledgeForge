"""Tests for study_engine._study_worthy (offline, no Ollama)."""
import study_engine


def test_study_worthy_true_for_prose():
    prose = (
        "Photosynthesis is the process by which green plants convert sunlight into "
        "chemical energy. Chlorophyll in the leaves absorbs light, and the plant uses "
        "that energy to combine carbon dioxide and water into glucose. Oxygen is "
        "released as a byproduct, which is essential for most life on Earth."
    )
    assert study_engine._study_worthy(prose, {"file_type": "pdf"})


def test_study_worthy_false_for_number_blob():
    blob = '{"a":1,"b":2}' * 60
    assert not study_engine._study_worthy(blob, {"file_type": "txt"})


def test_study_worthy_false_for_short_fragment():
    assert not study_engine._study_worthy("Just a short note.", {"file_type": "txt"})


def test_study_worthy_false_for_non_study_type():
    prose = (
        "This paragraph has more than forty words of genuine prose so it would "
        "otherwise pass the length and sentence checks that the study heuristic "
        "applies, but because the metadata marks it as a csv file type it should "
        "still be rejected outright by the guard clause here."
    )
    assert not study_engine._study_worthy(prose, {"file_type": "csv"})
