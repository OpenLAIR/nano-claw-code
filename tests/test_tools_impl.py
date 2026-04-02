"""tools_impl: Read, Write, Edit, Bash, Glob, Grep, NotebookEdit, TodoWrite, dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nano_claw_code.tools_impl import (
    TOOL_DISPATCH,
    anthropic_tool_defs,
    dispatch_tool,
    get_bash_cwd,
    get_todos,
    reset_bash_cwd,
    tool_todowrite,
    _html_to_text,
)


def test_anthropic_tool_defs_names():
    names = {t["name"] for t in anthropic_tool_defs()}
    assert names == {
        "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        "WebFetch", "WebSearch", "NotebookEdit", "TodoWrite", "Agent", "Skill",
    }


def test_dispatch_unknown_tool(tmp_path: Path):
    out = dispatch_tool(tmp_path, "Nope", {})
    assert "unknown tool" in out.lower()


def test_read_write_edit(tmp_path: Path):
    f = tmp_path / "a.txt"
    dispatch_tool(tmp_path, "Write", {"file_path": str(f), "content": "hello\nworld\n"})
    text = dispatch_tool(tmp_path, "Read", {"file_path": str(f)})
    assert "hello" in text
    assert "world" in text

    dispatch_tool(
        tmp_path,
        "Edit",
        {"file_path": str(f), "old_string": "world", "new_string": "moon"},
    )
    raw = f.read_text()
    assert "moon" in raw
    assert "world" not in raw


def test_read_list_dir(tmp_path: Path):
    (tmp_path / "x.py").write_text("1")
    out = dispatch_tool(tmp_path, "Read", {"file_path": str(tmp_path)})
    assert "x.py" in out or "[file]" in out


def test_bash_persists_cwd(tmp_path: Path):
    reset_bash_cwd()
    sub = tmp_path / "sub"
    sub.mkdir()
    dispatch_tool(tmp_path, "Bash", {"command": f"cd {sub}"})
    assert get_bash_cwd() == sub.resolve()
    out = dispatch_tool(tmp_path, "Bash", {"command": "pwd"})
    assert str(sub.resolve()) in out or sub.name in out


def test_bash_sandbox_blocks(tmp_path: Path):
    reset_bash_cwd()
    out = dispatch_tool(tmp_path, "Bash", {"command": "rm -rf /", "sandbox": True})
    assert "blocked" in out.lower()


def test_glob(tmp_path: Path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.js").write_text("y")
    out = dispatch_tool(tmp_path, "Glob", {"pattern": "*.py"})
    assert "a.py" in out


def test_grep(tmp_path: Path):
    (tmp_path / "t.py").write_text("alpha beta\ngamma\n")
    out = dispatch_tool(tmp_path, "Grep", {"pattern": "beta", "path": str(tmp_path)})
    assert "beta" in out


def test_notebook_new_cell(tmp_path: Path):
    nb = tmp_path / "n.ipynb"
    dispatch_tool(
        tmp_path,
        "NotebookEdit",
        {
            "target_notebook": str(nb),
            "cell_idx": 0,
            "is_new_cell": True,
            "new_string": "print(1)",
            "cell_language": "python",
        },
    )
    data = json.loads(nb.read_text())
    assert len(data["cells"]) == 1
    assert "print(1)" in "".join(data["cells"][0]["source"])


def test_todo_merge_and_clear(tmp_path: Path):
    tool_todowrite(tmp_path, {"todos": [{"id": "1", "content": "a", "status": "pending"}]})
    tool_todowrite(
        tmp_path,
        {"todos": [{"id": "1", "content": "a", "status": "completed"}], "merge": True},
    )
    assert get_todos() == []


def test_html_to_text_strips_script_and_style():
    html = "<html><body><p>Hi</p><script>exfil</script></body></html>"
    t = _html_to_text(html)
    assert "Hi" in t
    assert "exfil" not in t
    assert "<script" not in t.lower()
    assert "</script>" not in t.lower()

    styled = "<style>body{color:red}</style><p>visible</p>"
    t2 = _html_to_text(styled)
    assert "visible" in t2
    assert "color:red" not in t2
    assert "<style" not in t2.lower()


def test_tool_dispatch_covers_all_defs():
    for d in anthropic_tool_defs():
        assert d["name"] in TOOL_DISPATCH
