"""frontmatter.parse_markdown_frontmatter, lists, meta_* helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from nano_claw_code.frontmatter import (
    expand_memory_includes,
    meta_bool,
    meta_int,
    parse_comma_list,
    parse_markdown_frontmatter,
)


def test_parse_frontmatter_basic():
    text = "---\nfoo: bar\nnum: 42\n---\nBody here"
    meta, body = parse_markdown_frontmatter(text)
    assert meta["foo"] == "bar"
    assert meta["num"] == "42"
    assert body == "Body here"


def test_parse_frontmatter_quoted():
    text = '---\nk: "hello world"\n---\nx'
    meta, body = parse_markdown_frontmatter(text)
    assert meta["k"] == "hello world"


def test_parse_no_frontmatter():
    meta, body = parse_markdown_frontmatter("plain")
    assert meta == {}
    assert body == "plain"


def test_parse_comma_list():
    assert parse_comma_list("Read, Write, Bash") == ["Read", "Write", "Bash"]
    assert parse_comma_list("") == []


def test_meta_bool():
    assert meta_bool({"a": "true"}, "a") is True
    assert meta_bool({"a": "false"}, "a") is False
    assert meta_bool({}, "a", default=True) is True


def test_meta_int():
    assert meta_int({"n": "7"}, "n") == 7
    assert meta_int({}, "n") is None


def test_expand_memory_skips_code_fence(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("```\n@evil\n```\n\n@inc.md\n", encoding="utf-8")
    inc = tmp_path / "inc.md"
    inc.write_text("included", encoding="utf-8")
    out = expand_memory_includes(f.read_text(), f)
    # @evil inside ``` must not be treated as an @include directive
    assert "[include missing or unreadable: evil]" not in out
    assert "included" in out


def test_expand_memory_circular(tmp_path: Path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("@b.md", encoding="utf-8")
    b.write_text("@a.md", encoding="utf-8")
    out = expand_memory_includes(a.read_text(), a)
    assert "circular" in out.lower()
