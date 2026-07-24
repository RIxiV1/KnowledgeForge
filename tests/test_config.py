"""Tests for config defaults (offline, no Ollama)."""
import config


def test_max_chunk_size_is_int():
    assert isinstance(config.MAX_CHUNK_SIZE, int)
    assert config.MAX_CHUNK_SIZE > 0


def test_relevance_threshold_is_float_in_range():
    assert isinstance(config.RELEVANCE_THRESHOLD, float)
    assert 0.0 <= config.RELEVANCE_THRESHOLD <= 1.0


def test_max_upload_mb_positive_int():
    assert isinstance(config.MAX_UPLOAD_MB, int)
    assert config.MAX_UPLOAD_MB > 0


def test_model_names_present():
    assert isinstance(config.LLM_MODEL, str) and config.LLM_MODEL
    assert isinstance(config.EMBEDDING_MODEL, str) and config.EMBEDDING_MODEL
