"""session save / load / search (uses patched SESSIONS_DIR from conftest)."""

from __future__ import annotations

from nano_claw_code.session import (
    generate_session_title,
    load_session,
    rename_session,
    save_session,
    search_sessions,
)


def test_save_load_roundtrip():
    messages = [
        {"role": "user", "content": "Fix the bug in parser.py"},
        {"role": "assistant", "content": "OK"},
    ]
    path = save_session(messages, filename="t1.json", model="m1")
    data = load_session("t1.json")
    assert data["model"] == "m1"
    assert len(data["messages"]) == 2
    assert "parser" in data.get("title", "").lower() or "fix" in data.get("title", "").lower()


def test_generate_title():
    t = generate_session_title([{"role": "user", "content": "Hello world test"}])
    assert "Hello" in t or "world" in t


def test_search_sessions():
    save_session([{"role": "user", "content": "x"}], filename="unique_alpha_beta.json")
    hits = search_sessions("unique_alpha")
    assert any("unique_alpha" in h.get("filename", "") for h in hits)


def test_rename_session():
    save_session([{"role": "user", "content": "x"}], filename="rn.json", title="old")
    assert rename_session("rn.json", "newtitle")
    data = load_session("rn.json")
    assert data["title"] == "newtitle"
