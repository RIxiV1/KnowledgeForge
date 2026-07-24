"""Smoke tests that app.py renders with no exceptions via Streamlit AppTest.

AppTest runs the app offline; it does not hit Ollama at import/render time.
"""
import os

from streamlit.testing.v1 import AppTest

# app.py lives at the repo root, one level above this tests/ directory.
APP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


def test_app_renders_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0


def test_app_chat_mode_no_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["app_mode"] = "💬 Chat"
    at.run()
    assert len(at.exception) == 0
