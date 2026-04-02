"""Shared YAML-like frontmatter parsing for SKILL.md and agent markdown."""

from __future__ import annotations

import re
from pathlib import Path
from typing import FrozenSet

# Lines that are only @path — include file text (Claude Code memory @include).
_INCLUDE = re.compile(r"^@([^\s#]+)\s*$")


def parse_markdown_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse `---` delimited frontmatter. Keys are lowercased; values stripped.

    Supports simple `key: value` lines. Values can be quoted with " or '.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip()
    body = text[end + 3:].lstrip("\n")
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        k = key.strip().lower().replace(" ", "-")
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        meta[k] = val
    return meta, body


def parse_comma_list(value: str) -> list[str]:
    if not value or not value.strip():
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def meta_bool(meta: dict[str, str], *keys: str, default: bool = False) -> bool:
    for k in keys:
        if k in meta:
            return meta[k].strip().lower() in ("true", "1", "yes", "on")
    return default


def meta_int(meta: dict[str, str], *keys: str) -> int | None:
    for k in keys:
        if k not in meta:
            continue
        try:
            return int(meta[k].strip())
        except ValueError:
            return None
    return None


def expand_memory_includes(
    text: str,
    base_file: Path,
    *,
    _chain: FrozenSet[Path] | None = None,
    _depth: int = 0,
    max_depth: int = 8,
    max_chars: int = 100_000,
) -> str:
    """Expand @path lines at line start (outside ``` fences). Circular-safe."""
    if _chain is None:
        _chain = frozenset()
    if _depth > max_depth:
        return text[:max_chars] + "\n[include depth limit]\n"

    try:
        here = base_file.resolve()
    except OSError:
        here = base_file
    if here in _chain:
        return "[circular @include]\n"
    chain = _chain | {here}

    parts = re.split(r"(```[\s\S]*?```)", text)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
            continue
        lines_out: list[str] = []
        for line in part.splitlines(keepends=True):
            stripped = line.strip()
            m = _INCLUDE.match(stripped) if stripped else None
            if not m:
                lines_out.append(line)
                continue
            spec = m.group(1).strip()
            inc_path = _resolve_include_path(spec, base_file.parent)
            if not inc_path or not inc_path.is_file():
                lines_out.append(f"[include missing or unreadable: {spec}]\n")
                continue
            try:
                raw = inc_path.read_text(encoding="utf-8", errors="replace")
                if len(raw) > max_chars:
                    raw = raw[:max_chars] + "\n[truncated]\n"
                expanded = expand_memory_includes(
                    raw,
                    inc_path,
                    _chain=chain,
                    _depth=_depth + 1,
                    max_depth=max_depth,
                    max_chars=max_chars,
                )
                lines_out.append(
                    f"\n--- included from {inc_path} ---\n{expanded}\n--- end include ---\n"
                )
            except OSError:
                lines_out.append(f"[include error: {spec}]\n")
        out.append("".join(lines_out))
    return "".join(out)


def _resolve_include_path(spec: str, base_dir: Path) -> Path | None:
    s = spec.strip()
    if not s:
        return None
    if s.startswith("~/"):
        p = Path.home() / s[2:]
    elif s.startswith("/"):
        p = Path(s)
    else:
        p = (base_dir / s).resolve()
    return p if p.exists() else None
