"""Shared pytest fixtures and Ollama-availability gating.

Tests that need a live Ollama server (embeddings or the LLM) are marked with
``@pytest.mark.ollama``. When Ollama is unreachable (e.g. in CI) those tests are
skipped automatically so the rest of the suite still passes offline.
"""
import urllib.request

import pytest


def _ollama_up(url="http://localhost:11434/api/tags", timeout=1):
    """Return True if a local Ollama server answers, False otherwise."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - any failure means Ollama is unreachable
        return False


OLLAMA_AVAILABLE = _ollama_up()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ollama: test requires a running local Ollama server (skipped when unreachable).",
    )


@pytest.fixture(autouse=True)
def _skip_when_ollama_down(request):
    """Skip any test marked ``ollama`` if no local Ollama server is reachable."""
    if request.node.get_closest_marker("ollama") and not OLLAMA_AVAILABLE:
        pytest.skip("Ollama server not reachable at http://localhost:11434 (offline).")
