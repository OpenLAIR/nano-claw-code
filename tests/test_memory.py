"""memory.load_memory_context layered rules."""

from __future__ import annotations

from pathlib import Path

from nano_claw_code.memory import load_memory_context


def test_memory_layers_order(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text("root-claude", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "CLAUDE.md").write_text("nested-claude", encoding="utf-8")
    (sub / ".claude" / "rules").mkdir(parents=True)
    (sub / ".claude" / "rules" / "r.md").write_text("---\n---\nrule-nested", encoding="utf-8")

    ctx = load_memory_context(str(sub))
    assert "root-claude" in ctx
    assert "nested-claude" in ctx
    assert "rule-nested" in ctx
    # nearer cwd should appear after root in file order (both present)
    assert ctx.index("nested-claude") > ctx.index("root-claude")


def test_claude_local(tmp_path: Path):
    (tmp_path / "CLAUDE.local.md").write_text("local only", encoding="utf-8")
    ctx = load_memory_context(str(tmp_path))
    assert "local only" in ctx
