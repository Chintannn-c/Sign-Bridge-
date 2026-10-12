"""
Unit and integration tests for SignBridge Flask app and LLM endpoints.
Covers Section A & B fixes:
1. clean_llm_text preserves apostrophes, underscores, strips <think> (including unterminated), converts quotes.
2. refine endpoint falls back to fallback_refine when both providers fail or return empty.
3. simplify keeps underscores, filters to vocabulary tokens, and returns unknown words in 'fingerspell'.
4. limit_sentences preserves multi-sentence statements without dropping needed context.
5. has() helper matches whole words only (no substring false positives like 'tea' in 'teacher' or 'hi' in 'this').
6. LLM routes reject non-string, missing, empty, or over-length inputs with 400.
7. DB failure in log_conversation does not cause 500 errors on LLM endpoints.
8. /api/translate/word rejects invalid frame lengths and dimensions with 400.
9. /api/model/reload rejects GET requests with 405 Method Not Allowed.
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure backend root is on sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import (
    app,
    clean_llm_text,
    limit_sentences,
    has,
    fallback_refine,
    fallback_simplify,
    get_smart_fallback_response,
    get_word_vocabulary,
    safe_log
)


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as test_client:
        yield test_client


# ─── 1. clean_llm_text tests ────────────────────────────────────────────────

def test_clean_llm_text_preserves_apostrophe_and_underscore():
    raw = "What's the status of THANK_YOU gesture?"
    cleaned = clean_llm_text(raw)
    assert "What's" in cleaned
    assert "THANK_YOU" in cleaned


def test_clean_llm_text_strips_think_tags_and_unterminated():
    # Terminated <think>
    text_with_think = "<think>Analyzing user gestures and posture...</think>What's your name?"
    assert clean_llm_text(text_with_think) == "What's your name?"

    # Unterminated <think>
    unterminated = "<think>Model crashed mid reasoning without closing tag"
    assert clean_llm_text(unterminated) == ""


def test_clean_llm_text_converts_curly_quotes_and_strips_labels():
    curly = "\u2018Final Answer: What\u2019s up?\u2019"
    cleaned = clean_llm_text(curly)
    assert cleaned == "What's up?"


# ─── 2. refine fallback tests ───────────────────────────────────────────────

def test_fallback_refine_logic():
    # Question word at end -> ends with ?
    assert fallback_refine("NAME YOU WHAT") == "Name you what?"
    # Question word at end with underscore
    assert fallback_refine("YOU_WHERE") == "You where?"
    # Regular statement -> ends with .
    assert fallback_refine("WATER PLEASE") == "Water please."
    assert fallback_refine("HELLO_WORLD") == "Hello world."


def test_refine_endpoint_uses_fallback_refine(client, monkeypatch):
    from services.groq_manager import groq_manager
    from services.gemini_manager import gemini_manager

    # Force both providers to fail
    monkeypatch.setattr(groq_manager, "is_available", lambda: False)
    monkeypatch.setattr(gemini_manager, "is_available", lambda: False)

    res = client.post('/api/llm/refine', json={'text': 'NAME YOU WHAT'})
    assert res.status_code == 200
    data = res.get_json()
    assert data['raw_text'] == 'NAME YOU WHAT'
    # Must use fallback_refine, NOT a chat answer like "I'm SignBridge!"
    assert data['refined_sentence'] == 'Name you what?'
    assert data['fallback_used'] is True
    assert data['llm_provider'] == 'local_fallback'


def test_refine_endpoint_handles_empty_model_response(client, monkeypatch):
    import app as app_module

    # Mock generate_llm_response returning empty string
    monkeypatch.setattr(
        app_module,
        "generate_llm_response",
        lambda *args, **kwargs: {"text": "", "provider": "Groq (mock)", "fallback_used": False}
    )

    res = client.post('/api/llm/refine', json={'text': 'WATER PLEASE'})
    assert res.status_code == 200
    data = res.get_json()
    assert data['refined_sentence'] == 'Water please.'
    assert data['fallback_used'] is True
    assert data['llm_provider'] == 'local_fallback'


# ─── 3. simplify preserves underscores and returns fingerspell ──────────────

def test_simplify_keeps_underscores_and_filters_vocab(client, monkeypatch):
    import app as app_module

    vocab = ["WATER", "PLEASE", "THANK_YOU", "DOCTOR"]
    monkeypatch.setattr(app_module, "get_word_vocabulary", lambda: vocab)

    # Mock model output containing vocab tokens and an unknown word with underscores
    monkeypatch.setattr(
        app_module,
        "generate_llm_response",
        lambda *args, **kwargs: {
            "text": "WATER BOTTLE THANK_YOU",
            "provider": "Groq (mock)",
            "fallback_used": False
        }
    )

    res = client.post('/api/llm/simplify', json={'text': 'Can I get a water bottle, thank you?'})
    assert res.status_code == 200
    data = res.get_json()
    assert data['robot_keywords'] == "WATER THANK_YOU"
    # Unknown word 'BOTTLE' should be broken into A-Z letters in fingerspell
    assert "fingerspell" in data
    assert data['fingerspell'] == ['B', 'O', 'T', 'T', 'L', 'E']


def test_fallback_simplify_function():
    vocab = ["WATER", "FOOD", "THANK_YOU"]
    assert fallback_simplify("NEED WATER AND FOOD", vocab) == "WATER FOOD"
    assert fallback_simplify("THANK_YOU VERY MUCH", vocab) == "THANK_YOU"


# ─── 4. limit_sentences preserves multi-sentence statements ─────────────────

def test_limit_sentences_preserves_meaning():
    text = "Where is the doctor? I need help."
    result = limit_sentences(text, max_sentences=2, max_words=25)
    assert result == "Where is the doctor? I need help."


def test_limit_sentences_caps_at_word_and_sentence_limits():
    long_text = "Sentence one is here. Sentence two is here. Sentence three should be cut off."
    capped = limit_sentences(long_text, max_sentences=2, max_words=25)
    assert "Sentence one is here." in capped
    assert "Sentence two is here." in capped
    assert "Sentence three" not in capped


# ─── 5. has() helper matches whole words only ───────────────────────────────

def test_has_word_boundary_matching():
    # Negative checks: substrings must not trigger matches
    assert not has("teacher", "tea")
    assert not has("this is signbridge", "hi")
    assert not has("great lunch", "eat")
    assert not has("behind the car", "hi")

    # Positive checks: whole words must match regardless of punctuation
    assert has("cup of tea please", "tea")
    assert has("hi, how are you?", "hi")
    assert has("i want to eat food", "eat")
    assert has("Where is the washroom?", "washroom")


# ─── 6. LLM routes input validation (400 on invalid input) ──────────────────

@pytest.mark.parametrize("route", ['/api/llm/refine', '/api/llm/simplify', '/api/llm/answer'])
def test_llm_routes_reject_invalid_inputs(client, route):
    # Non-string input
    r1 = client.post(route, json={'text': 12345})
    assert r1.status_code == 400

    r2 = client.post(route, json={'text': ['hello']})
    assert r2.status_code == 400

    # Missing text field
    r3 = client.post(route, json={'wrong_key': 'hello'})
    assert r3.status_code == 400

    # Empty string
    r4 = client.post(route, json={'text': '   '})
    assert r4.status_code == 400

    # Over-length (> 300 chars)
    r5 = client.post(route, json={'text': 'A' * 301})
    assert r5.status_code == 400


# ─── 7. DB logging failure is non-fatal ──────────────────────────────────────

def test_db_failure_does_not_crash_llm_endpoints(client, monkeypatch):
    import database.schema as schema_mod

    def broken_log(*args, **kwargs):
        raise RuntimeError("SQLite database file is corrupted or locked")

    monkeypatch.setattr(schema_mod, "log_conversation", broken_log)

    # Calling refine should succeed despite DB logging error
    res_refine = client.post('/api/llm/refine', json={'text': 'HELLO WORLD'})
    assert res_refine.status_code == 200

    # Calling answer should succeed despite DB logging error
    res_answer = client.post('/api/llm/answer', json={'text': 'HELLO'})
    assert res_answer.status_code == 200


# ─── 8. /api/translate/word payload shape validation ────────────────────────

def test_translate_word_validates_frame_shapes(client):
    # Missing frames
    r_empty = client.post('/api/translate/word', json={})
    assert r_empty.status_code == 400

    # Less than 15 frames
    r_short = client.post('/api/translate/word', json={'frames': [[0.0] * 126] * 10})
    assert r_short.status_code == 400

    # More than 60 frames
    r_long = client.post('/api/translate/word', json={'frames': [[0.0] * 126] * 65})
    assert r_long.status_code == 400

    # Frame with wrong number of landmarks (e.g. 100 instead of 126)
    r_dim = client.post('/api/translate/word', json={'frames': [[0.0] * 100] * 30})
    assert r_dim.status_code == 400

    # Non-numeric coordinate
    r_non_num = client.post('/api/translate/word', json={'frames': [['zero'] * 126] * 30})
    assert r_non_num.status_code == 400

    # Valid shape all-zeros should trigger rejection without 400 or 500
    r_zeros = client.post('/api/translate/word', json={'frames': [[0.0] * 126] * 30})
    # If word_recognizer is available, returns rejected: True with status 200, or 503 if unavailable
    assert r_zeros.status_code in (200, 503)
    if r_zeros.status_code == 200:
        assert r_zeros.get_json()['rejected'] is True


# ─── 9. /api/model/reload rejects GET ───────────────────────────────────────

def test_model_reload_rejects_get(client):
    res = client.get('/api/model/reload')
    assert res.status_code == 405
