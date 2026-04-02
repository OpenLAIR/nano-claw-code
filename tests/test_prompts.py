"""build_system_prompt sections."""

from __future__ import annotations

from pathlib import Path

from nano_claw_code.prompts import build_subagent_system_prompt, build_system_prompt
from nano_claw_code.agents import resolve_agent


def test_build_system_prompt_has_tools_and_cwd(tmp_path: Path):
    p = tmp_path / "proj"
    p.mkdir()
    (p / "CLAUDE.md").write_text("Use ruff.", encoding="utf-8")
    s = build_system_prompt(cwd=str(p), bare=False)
    assert "Read" in s
    assert "Nano Claw Code" in s
    assert "ruff" in s.lower() or "Use ruff" in s


def test_build_subagent_omit_memory(tmp_path: Path):
    ag = resolve_agent("Explore", str(tmp_path))
    s = build_subagent_system_prompt(str(tmp_path), ag)
    assert "Explore" in s
    assert "read-only" in s.lower() or "READ-ONLY" in s
