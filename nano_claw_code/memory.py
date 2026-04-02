"""Layered CLAUDE.md, rules, and local memory — aligned with Claude Code claudemd.ts.

Loads (in order, later chunks closer to CWD so they appear last in the prompt):
  1. User: ~/.claude/CLAUDE.md, ~/.claude/rules/*.md
  2. For each directory from filesystem root → cwd:
     - CLAUDE.md
     - .claude/CLAUDE.md
     - .claude/rules/*.md (sorted)
     - CLAUDE.local.md

Supports @include in memory files (line-anchored @path, outside ``` blocks).
"""

from __future__ import annotations

from pathlib import Path

from nano_claw_code.frontmatter import expand_memory_includes

MAX_FILE_CHARS = 12_000
MAX_SECTION_CHARS = 56_000


def _dirs_root_to_cwd(start: Path) -> list[Path]:
    dirs: list[Path] = []
    p = start.resolve()
    for _ in range(64):
        dirs.append(p)
        parent = p.parent
        if parent == p:
            break
        p = parent
    return list(reversed(dirs))


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    half = limit // 2
    return s[:half] + f"\n\n[... truncated {len(s) - limit} chars ...]\n\n" + s[-half:]


def _read_memory_file(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    raw = _truncate(raw, MAX_FILE_CHARS)
    raw = expand_memory_includes(raw, path)
    return _truncate(raw, MAX_FILE_CHARS)


def _collect_rules(rules_dir: Path) -> list[Path]:
    if not rules_dir.is_dir():
        return []
    return sorted(
        p for p in rules_dir.iterdir()
        if p.suffix.lower() == ".md" and p.is_file()
    )


def load_memory_context(cwd: str) -> str:
    """Return markdown section for system prompt (empty if nothing found)."""
    start = Path(cwd).resolve()
    chunks: list[str] = []

    # User-level (global)
    user_claude = Path.home() / ".claude" / "CLAUDE.md"
    if user_claude.is_file():
        text = _read_memory_file(user_claude)
        if text.strip():
            chunks.append(f"[User CLAUDE.md: {user_claude}]\n{text}")

    user_rules = _collect_rules(Path.home() / ".claude" / "rules")
    for rp in user_rules:
        text = _read_memory_file(rp)
        if text.strip():
            chunks.append(f"[User rule: {rp.name}]\n{text}")

    for d in _dirs_root_to_cwd(start):
        claude = d / "CLAUDE.md"
        if claude.is_file():
            text = _read_memory_file(claude)
            if text.strip():
                chunks.append(f"[Project CLAUDE.md: {claude}]\n{text}")

        dot_claude = d / ".claude" / "CLAUDE.md"
        if dot_claude.is_file():
            text = _read_memory_file(dot_claude)
            if text.strip():
                chunks.append(f"[Project .claude/CLAUDE.md: {dot_claude}]\n{text}")

        for rp in _collect_rules(d / ".claude" / "rules"):
            text = _read_memory_file(rp)
            if text.strip():
                chunks.append(f"[Project rule {d.name}/.claude/rules/{rp.name}]\n{text}")

        local_md = d / "CLAUDE.local.md"
        if local_md.is_file():
            text = _read_memory_file(local_md)
            if text.strip():
                chunks.append(f"[Local CLAUDE.local.md: {local_md}]\n{text}")

    if not chunks:
        return ""

    combined = "\n\n".join(chunks)
    if len(combined) > MAX_SECTION_CHARS:
        combined = _truncate(combined, MAX_SECTION_CHARS) + "\n[memory section truncated]\n"

    return "\n\n# Memory / rules (CLAUDE.md hierarchy)\n" + combined + "\n"
