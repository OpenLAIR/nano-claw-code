"""Claude Code-compatible tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch,
NotebookEdit, TodoWrite, Agent.

Converted from nano-claude-code TypeScript tools/ directory. 11 tools total.
Features: per-tool truncation budgets, persistent Bash cwd, sandbox mode, Edit diff preview.
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ── Per-tool truncation budgets (characters) ──────────────────────────────
# Different tools produce different amounts of output; fine-grained limits
# prevent any single tool result from dominating the context window.

TOOL_CHAR_LIMITS: dict[str, int] = {
    "Read":         100_000,
    "Write":         2_000,
    "Edit":         10_000,
    "Bash":         60_000,
    "Glob":         30_000,
    "Grep":         40_000,
    "WebFetch":     50_000,
    "WebSearch":    20_000,
    "NotebookEdit":  5_000,
    "TodoWrite":     5_000,
    "Agent":        50_000,
    "Skill":        50_000,
}
DEFAULT_CHAR_LIMIT = 100_000

BASH_TIMEOUT_SEC = 600

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}


def _truncate(s: str, limit: int = DEFAULT_CHAR_LIMIT) -> tuple[str, bool]:
    if len(s) <= limit:
        return s, False
    half = limit // 2
    return (
        s[:half]
        + f"\n\n[... truncated {len(s) - limit} chars ...]\n\n"
        + s[-half:]
    ), True


def _truncate_tool(tool_name: str, s: str) -> str:
    limit = TOOL_CHAR_LIMITS.get(tool_name, DEFAULT_CHAR_LIMIT)
    t, was_truncated = _truncate(s, limit)
    if was_truncated:
        t += f"\n[output truncated to {limit} chars for {tool_name}]"
    return t


def _abs_path(cwd: Path, file_path: str) -> Path:
    p = Path(file_path).expanduser()
    if not p.is_absolute():
        p = (cwd / p).resolve()
    return p


# ── Read ──────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
IMAGE_MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def tool_read(cwd: Path, inp: dict) -> str | list:
    """Read a file. Returns str for text files, or a list of content blocks for images."""
    file_path = inp.get("file_path") or inp.get("path") or ""
    offset = inp.get("offset")
    limit = inp.get("limit")
    path = _abs_path(cwd, file_path)
    if not path.exists():
        return f"Error: file not found: {path}"
    if path.is_dir():
        entries = sorted(path.iterdir())
        listing = "\n".join(
            f"{'[dir]  ' if e.is_dir() else '[file] '}{e.name}" for e in entries[:200]
        )
        return f"Directory listing of {path}:\n{listing}"

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return _read_image(path, suffix)

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading file: {e}"
    lines = raw.splitlines()
    start = 0
    end = len(lines)
    if offset is not None:
        start = max(0, int(offset) - 1)
    if limit is not None:
        end = min(end, start + int(limit))
    chunk = lines[start:end]
    out_lines = [f"{start + i + 1:6}|{line}" for i, line in enumerate(chunk)]
    text = "\n".join(out_lines)
    if len(lines) > end or start > 0:
        text += f"\n\n[Showing lines {start + 1}-{end} of {len(lines)} total]"
    return _truncate_tool("Read", text)


def _read_image(path: Path, suffix: str) -> str | list:
    import base64
    media_type = IMAGE_MEDIA_TYPES.get(suffix, "image/png")
    try:
        data = path.read_bytes()
    except OSError as e:
        return f"Error reading image: {e}"
    if len(data) > MAX_IMAGE_BYTES:
        return f"Error: image too large ({len(data) / 1024 / 1024:.1f} MB, max {MAX_IMAGE_BYTES / 1024 / 1024:.0f} MB)"
    b64 = base64.b64encode(data).decode("ascii")
    return [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
        {"type": "text", "text": f"[Image: {path.name} ({len(data)} bytes, {media_type})]"},
    ]


# ── Write ─────────────────────────────────────────────────────────────────


def tool_write(cwd: Path, inp: dict) -> str:
    file_path = inp.get("file_path") or inp.get("path") or ""
    content = inp.get("content")
    if content is None:
        return "Error: missing content"
    path = _abs_path(cwd, file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
    except OSError as e:
        return f"Error writing file: {e}"
    lc = str(content).count("\n") + 1
    return f"Wrote {path} ({len(str(content))} chars, {lc} lines)"


# ── Edit (with diff preview) ─────────────────────────────────────────────


def tool_edit(cwd: Path, inp: dict) -> str:
    file_path = inp.get("file_path") or inp.get("path") or ""
    old = inp.get("old_string")
    new = inp.get("new_string")
    replace_all = bool(inp.get("replace_all", False))
    if old is None or new is None:
        return "Error: old_string and new_string are required"
    if old == new:
        return "Error: old_string and new_string must differ"
    path = _abs_path(cwd, file_path)
    if not path.exists():
        if old == "":
            return tool_write(cwd, {"file_path": file_path, "content": new})
        return f"Error: file not found: {path}"
    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading file: {e}"
    if old not in original:
        return (
            "Error: old_string not found in file (must match exactly, including whitespace). "
            "Read the file again and copy the exact span to replace."
        )
    if not replace_all and original.count(old) > 1:
        return (
            f"Error: old_string appears {original.count(old)} times. "
            "Include more context in old_string to make it unique, or set replace_all to true."
        )
    updated = original.replace(old, new) if replace_all else original.replace(old, new, 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as e:
        return f"Error writing file: {e}"

    diff = _generate_diff(original, updated, str(path))
    mode = "all occurrences" if replace_all else "one occurrence"
    result = f"Edited {path} ({mode})"
    if diff:
        result += f"\n\n{diff}"
    return _truncate_tool("Edit", result)


def _generate_diff(original: str, updated: str, filename: str) -> str:
    """Generate a unified diff between original and updated content."""
    orig_lines = original.splitlines(keepends=True)
    new_lines = updated.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        orig_lines, new_lines,
        fromfile=f"a/{Path(filename).name}",
        tofile=f"b/{Path(filename).name}",
        n=3,
    ))
    if not diff_lines:
        return ""
    diff_text = "".join(diff_lines)
    if len(diff_text) > 3000:
        diff_text = diff_text[:3000] + "\n[diff truncated]"
    return diff_text


# ── Bash (persistent cwd + sandbox mode) ─────────────────────────────────

_bash_cwd: Path | None = None

SANDBOX_BLOCKED_PATTERNS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev",
    "> /dev/sda", "chmod -R 777 /", ":(){ :|:& };:",
]

SANDBOX_BLOCKED_PIPE_SINKS = ["bash", "sh", "zsh", "eval"]


def _is_sandbox_blocked(cmd: str) -> bool:
    """Check if a command is blocked in sandbox mode."""
    c = cmd.strip().lower()
    if any(pat.lower() in c for pat in SANDBOX_BLOCKED_PATTERNS):
        return True
    if "|" in c:
        parts = [p.strip() for p in c.split("|")]
        for part in parts[1:]:
            first_word = part.split()[0] if part.split() else ""
            if first_word in SANDBOX_BLOCKED_PIPE_SINKS:
                if any(parts[0].startswith(dl) for dl in ("curl", "wget")):
                    return True
    return False


def _extract_cd_path(cmd: str) -> str | None:
    """Extract target directory from a cd command, if present."""
    stripped = cmd.strip()
    if stripped == "cd" or stripped == "cd ~":
        return str(Path.home())
    if stripped.startswith("cd "):
        parts = stripped[3:].strip()
        if parts.startswith("&&") or parts.startswith(";"):
            return None
        target = parts.split("&&")[0].split(";")[0].strip()
        target = target.strip("'\"")
        return target
    for sep in ["&&", ";"]:
        if f"cd " in stripped:
            for segment in stripped.split(sep):
                segment = segment.strip()
                if segment.startswith("cd "):
                    target = segment[3:].strip().split("&&")[0].split(";")[0].strip()
                    return target.strip("'\"")
    return None


def tool_bash(cwd: Path, inp: dict) -> str:
    global _bash_cwd
    command = inp.get("command")
    timeout = int(inp.get("timeout", BASH_TIMEOUT_SEC))
    sandbox = bool(inp.get("sandbox", False))
    if not command or not isinstance(command, str):
        return "Error: command is required"

    if sandbox and _is_sandbox_blocked(command):
        return f"Error: command blocked by sandbox mode: {command[:80]}"

    effective_cwd = _bash_cwd or cwd

    if not effective_cwd.exists():
        effective_cwd = cwd
        _bash_cwd = None

    cd_target = _extract_cd_path(command)

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(effective_cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error executing command: {e}"

    if cd_target and proc.returncode == 0:
        new_cwd = Path(cd_target).expanduser()
        if not new_cwd.is_absolute():
            new_cwd = (effective_cwd / new_cwd).resolve()
        if new_cwd.is_dir():
            _bash_cwd = new_cwd

    out = []
    if proc.stdout:
        out.append(proc.stdout)
    if proc.stderr:
        out.append(f"stderr:\n{proc.stderr}")
    out.append(f"\n[exit code: {proc.returncode}]")
    if _bash_cwd and _bash_cwd != cwd:
        out.append(f"[cwd: {_bash_cwd}]")
    text = "\n".join(out).strip()
    return _truncate_tool("Bash", text)


def get_bash_cwd() -> Path | None:
    """Get the persistent Bash working directory (for CLI display)."""
    return _bash_cwd


def reset_bash_cwd() -> None:
    """Reset the persistent Bash working directory."""
    global _bash_cwd
    _bash_cwd = None


def reset_transient_tool_state() -> None:
    """Reset process-wide Bash cwd and in-memory todos (tests / fresh REPL)."""
    global _bash_cwd, _todo_store
    _bash_cwd = None
    _todo_store.clear()


# ── Glob ──────────────────────────────────────────────────────────────────


def tool_glob(cwd: Path, inp: dict) -> str:
    pattern = inp.get("pattern") or inp.get("glob_pattern") or ""
    base = inp.get("path") or inp.get("target_directory")
    root = _abs_path(cwd, base) if base else cwd
    if not root.exists():
        return f"Error: path does not exist: {root}"
    if not root.is_dir():
        return f"Error: path is not a directory: {root}"

    if not pattern.startswith("**/") and "/" not in pattern:
        expanded_pattern = f"**/{pattern}"
    else:
        expanded_pattern = pattern

    matches: list[Path] = []
    try:
        for p in root.glob(expanded_pattern):
            if p.is_file() and not any(x in SKIP_DIRS for x in p.parts):
                matches.append(p.resolve())
            if len(matches) >= 200:
                break
    except (OSError, ValueError):
        pass

    if not matches:
        try:
            for p in root.glob(pattern):
                if p.is_file() and not any(x in SKIP_DIRS for x in p.parts):
                    matches.append(p.resolve())
                if len(matches) >= 200:
                    break
        except (OSError, ValueError):
            pass

    if not matches:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(x in SKIP_DIRS for x in p.parts):
                continue
            if fnmatch.fnmatch(p.name, pattern):
                matches.append(p.resolve())
            if len(matches) >= 200:
                break

    matches.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
    paths = [str(p) for p in matches[:200]]
    msg = "\n".join(paths)
    if len(matches) > 200:
        msg += f"\n\n[truncated: showing 200 of {len(matches)}+ matches]"
    return _truncate_tool("Glob", msg or "(no files matched)")


# ── Grep ──────────────────────────────────────────────────────────────────


def tool_grep(cwd: Path, inp: dict) -> str:
    pattern = inp.get("pattern")
    if not pattern:
        return "Error: pattern is required"
    path_arg = inp.get("path")
    glob_pat = inp.get("glob")
    head_limit = inp.get("head_limit")
    case_insensitive = inp.get("case_insensitive", inp.get("-i", False))
    context_lines = inp.get("context", inp.get("-C", 0))
    output_mode = inp.get("output_mode", "content")
    if head_limit is None:
        head_limit = 250
    head_limit = int(head_limit)
    root = _abs_path(cwd, path_arg) if path_arg else cwd

    rg = shutil.which("rg")
    if rg and root.exists():
        cmd = [rg, "--no-heading", "--color", "never"]
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        else:
            cmd.append("--line-number")
        if case_insensitive:
            cmd.append("-i")
        if context_lines and output_mode == "content":
            cmd.extend(["-C", str(int(context_lines))])
        if glob_pat:
            cmd.extend(["--glob", glob_pat])
        if head_limit > 0 and output_mode == "content":
            cmd.extend(["--max-count", str(head_limit)])
        cmd.extend([pattern, str(root)])
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), capture_output=True, text=True, timeout=120
            )
            text = (proc.stdout or "") + (proc.stderr or "")
            return _truncate_tool("Grep", text.strip() or "(no matches)")
        except (subprocess.TimeoutExpired, OSError) as e:
            return f"ripgrep failed ({e}), falling back to Python scan"

    try:
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: invalid regex: {e}"
    lines_out: list[str] = []
    count = 0

    def scan_file(fp: Path) -> None:
        nonlocal count
        if head_limit > 0 and count >= head_limit:
            return
        if glob_pat and not fnmatch.fnmatch(fp.name, glob_pat):
            return
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for i, line in enumerate(content.splitlines(), 1):
            if head_limit > 0 and count >= head_limit:
                return
            if regex.search(line):
                if output_mode == "files_with_matches":
                    lines_out.append(str(fp))
                    count += 1
                    return
                elif output_mode == "count":
                    count += 1
                else:
                    lines_out.append(f"{fp}:{i}:{line}")
                    count += 1

    if root.is_file():
        scan_file(root)
    elif root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if head_limit > 0 and count >= head_limit:
                    break
                scan_file(Path(dirpath) / fn)
    else:
        return f"Error: path not found: {root}"

    if output_mode == "count":
        return str(count)
    text = "\n".join(lines_out) if lines_out else "(no matches)"
    return _truncate_tool("Grep", text)


# ── WebFetch ──────────────────────────────────────────────────────────────


def tool_webfetch(cwd: Path, inp: dict) -> str:
    url = inp.get("url", "")
    if not url:
        return "Error: url is required"
    try:
        import httpx
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    try:
        r = httpx.get(
            url,
            headers={"User-Agent": "NanoClawCode/0.2"},
            timeout=30,
            follow_redirects=True,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"

    ct = r.headers.get("content-type", "")
    if "html" in ct:
        text = _html_to_text(r.text)
    else:
        text = r.text
    return _truncate_tool("WebFetch", text)


def _html_to_text(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<header[^>]*>.*?</header>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?div[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?li[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    for tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        text = re.sub(
            rf"<{tag}[^>]*>(.*?)</{tag}>",
            lambda m, t=tag: f"\n{'#' * int(t[1:])} {m.group(1)}\n",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    text = re.sub(r"<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", r"[\2](\1)", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ── WebSearch ─────────────────────────────────────────────────────────────


def tool_websearch(cwd: Path, inp: dict) -> str:
    query = inp.get("query") or inp.get("search_term") or ""
    if not query:
        return "Error: query is required"
    try:
        import httpx
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"

    try:
        r = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; NanoClawCode/0.2)"},
            timeout=30,
            follow_redirects=True,
        )
        titles = re.findall(
            r'class="result__title"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r.text,
            re.DOTALL,
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</[^>]+>',
            r.text,
            re.DOTALL,
        )
        results = []
        for i, (link, title) in enumerate(titles[:10]):
            t = re.sub(r"<[^>]+>", "", title).strip()
            s = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            results.append(f"**{t}**\n{link}\n{s}")
        return _truncate_tool("WebSearch", "\n\n".join(results) if results else "No results found")
    except Exception as e:
        return f"Error searching: {e}"


# ── NotebookEdit ──────────────────────────────────────────────────────────


def tool_notebook_edit(cwd: Path, inp: dict) -> str:
    notebook_path = inp.get("target_notebook") or inp.get("notebook") or ""
    cell_idx = inp.get("cell_idx")
    is_new = inp.get("is_new_cell", False)
    old_string = inp.get("old_string", "")
    new_string = inp.get("new_string", "")
    cell_language = inp.get("cell_language", "python")

    if cell_idx is None:
        return "Error: cell_idx is required"
    cell_idx = int(cell_idx)

    path = _abs_path(cwd, notebook_path)

    cell_type_map = {
        "python": "code", "markdown": "markdown", "raw": "raw",
        "r": "code", "sql": "code", "shell": "code",
        "javascript": "code", "typescript": "code",
    }
    nb_cell_type = cell_type_map.get(cell_language, "code")

    if is_new and not path.exists():
        nb = {
            "cells": [],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    else:
        if not path.exists():
            return f"Error: notebook not found: {path}"
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            return f"Error reading notebook: {e}"

    cells = nb.get("cells", [])

    if is_new:
        new_cell: dict[str, Any] = {
            "cell_type": nb_cell_type,
            "metadata": {},
            "source": new_string.splitlines(keepends=True),
        }
        if nb_cell_type == "code":
            new_cell["execution_count"] = None
            new_cell["outputs"] = []
        idx = min(cell_idx, len(cells))
        cells.insert(idx, new_cell)
        nb["cells"] = cells
    else:
        if cell_idx < 0 or cell_idx >= len(cells):
            return f"Error: cell_idx {cell_idx} out of range (notebook has {len(cells)} cells)"
        cell = cells[cell_idx]
        source = "".join(cell.get("source", []))
        if old_string and old_string not in source:
            return f"Error: old_string not found in cell {cell_idx}."
        if old_string:
            new_source = source.replace(old_string, new_string, 1)
        else:
            new_source = new_string
        cell["source"] = new_source.splitlines(keepends=True)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        return f"Error writing notebook: {e}"
    action = "Created new cell" if is_new else "Edited cell"
    return f"{action} {cell_idx} in {path}"


# ── TodoWrite ─────────────────────────────────────────────────────────────

_todo_store: dict[str, list[dict]] = {}


def tool_todowrite(cwd: Path, inp: dict) -> str:
    todos = inp.get("todos", [])
    merge = inp.get("merge", False)
    session_key = "default"

    if not isinstance(todos, list) or len(todos) == 0:
        return "Error: todos array is required with at least one item"

    old_todos = list(_todo_store.get(session_key, []))

    if merge and old_todos:
        by_id = {t["id"]: t for t in old_todos}
        for t in todos:
            tid = t.get("id", "")
            if tid in by_id:
                by_id[tid].update({k: v for k, v in t.items() if v is not None})
            else:
                by_id[tid] = t
        new_todos = list(by_id.values())
    else:
        new_todos = todos

    all_done = all(t.get("status") == "completed" for t in new_todos)
    _todo_store[session_key] = [] if all_done else new_todos

    lines = []
    for t in new_todos:
        status = t.get("status", "pending")
        marker = {"pending": "○", "in_progress": "●", "completed": "✓", "cancelled": "✗"}.get(status, "?")
        lines.append(f"  {marker} [{status}] {t.get('content', '')}")

    return (
        f"Todos updated ({len(new_todos)} items):\n"
        + "\n".join(lines)
        + "\nProceed with current tasks."
    )


def get_todos() -> list[dict]:
    return list(_todo_store.get("default", []))


# ── Agent (sub-agent) ────────────────────────────────────────────────────

def tool_agent(cwd: Path, inp: dict) -> str:
    """Sub-agent loop with optional subagent_type (built-in or .claude/agents/*.md)."""
    prompt = inp.get("prompt", "")
    if not prompt:
        return "Error: prompt is required"

    description = inp.get("description", "Sub-agent task")
    subagent_type = inp.get("subagent_type") or inp.get("subagent-type")

    from nano_claw_code.agents import filter_tools_for_agent, resolve_agent
    from nano_claw_code.config import resolve_api_env
    from nano_claw_code.prompts import build_subagent_system_prompt, resolve_model
    import anthropic as anth

    ag = resolve_agent(subagent_type, str(cwd))

    api_env = resolve_api_env()
    if not api_env.get("api_key"):
        return "Error: no API key available for sub-agent"

    api_env.pop("provider", None)
    client = anth.Anthropic(**api_env)
    model = resolve_model(ag.model if ag.model else None)
    system = build_subagent_system_prompt(str(cwd), ag)
    all_defs = [t for t in anthropic_tool_defs() if t["name"] != "Agent"]
    sub_tools = filter_tools_for_agent(all_defs, ag)
    if not sub_tools:
        sub_tools = all_defs

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    max_turns = ag.max_turns if ag.max_turns is not None else 10
    all_text: list[str] = []

    for _turn in range(max_turns):
        try:
            resp = client.messages.create(
                model=model, max_tokens=8192, system=system,
                tools=sub_tools, messages=messages,
            )
        except Exception as e:
            return f"Sub-agent error: {e}"

        blocks = list(resp.content)
        text_parts = [b.text for b in blocks if b.type == "text"]
        tool_uses = [b for b in blocks if b.type == "tool_use"]

        if text_parts:
            all_text.extend(text_parts)

        if resp.stop_reason == "end_turn" and not tool_uses:
            break
        if not tool_uses:
            break

        result_blocks: list[dict[str, Any]] = []
        for tu in tool_uses:
            raw_in = tu.input if isinstance(tu.input, dict) else {}
            out = dispatch_tool(cwd, tu.name, raw_in)
            is_err = isinstance(out, str) and out.startswith("Error:")
            result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": out if isinstance(out, str) else str(out),
                "is_error": is_err,
            })

        messages.append({
            "role": "assistant",
            "content": [c.model_dump(mode="json") for c in resp.content],
        })
        messages.append({"role": "user", "content": result_blocks})

    result_text = "\n".join(all_text) if all_text else "(sub-agent produced no text)"
    return _truncate_tool(
        "Agent",
        f"[Sub-agent: {ag.agent_type} — {description}]\n{result_text}",
    )


# ── Skill ─────────────────────────────────────────────────────────────────

def tool_skill(cwd: Path, inp: dict) -> str:
    """Execute a discovered skill by name, with optional arguments."""
    from nano_claw_code.skills import (
        get_skills, expand_skill_prompt, execute_skill_forked,
    )
    skill_name = (inp.get("skill") or "").strip().lstrip("/").lower()
    args = inp.get("args", "") or ""
    if not skill_name:
        return "Error: skill name is required"

    skills = get_skills(str(cwd))
    skill = skills.get(skill_name)
    if skill is None:
        available = ", ".join(sorted(skills.keys())[:20])
        return f"Error: unknown skill '{skill_name}'. Available: {available or '(none)'}"

    if skill.get("disable_model_invocation"):
        return f"Error: skill '{skill_name}' is not available for model invocation"

    if skill.get("context") == "fork":
        result = execute_skill_forked(skill, args, cwd)
        return _truncate_tool("Skill", f"[Skill: {skill_name} (forked)]\n{result}")

    prompt = expand_skill_prompt(skill, args)
    return _truncate_tool("Skill", f"[Skill: {skill_name}]\n{prompt}")


# ── Dispatcher ────────────────────────────────────────────────────────────

TOOL_DISPATCH: dict[str, Callable] = {
    "Read": tool_read,
    "Write": tool_write,
    "Edit": tool_edit,
    "Bash": tool_bash,
    "Glob": tool_glob,
    "Grep": tool_grep,
    "WebFetch": tool_webfetch,
    "WebSearch": tool_websearch,
    "NotebookEdit": tool_notebook_edit,
    "TodoWrite": tool_todowrite,
    "Agent": tool_agent,
    "Skill": tool_skill,
}


def dispatch_tool(cwd: Path, name: str, tool_input: dict) -> str | list:
    """Dispatch a tool call. Returns str for text results, or a list of content
    blocks for rich results (e.g. images via Read tool)."""
    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"
    try:
        return handler(cwd, tool_input)
    except Exception as e:
        return f"Error in {name}: {type(e).__name__}: {e}"


# ── Tool Definitions (Anthropic API format) ───────────────────────────────


def anthropic_tool_defs() -> list[dict]:
    return [
        {
            "name": "Read",
            "description": (
                "Read a file from the filesystem. Returns numbered lines. "
                "Supports images (png/jpg/gif/webp) as base64. "
                "If path is a directory, returns a listing. Use offset/limit for large files."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute or relative path to the file"},
                    "offset": {"type": "integer", "description": "1-based line number to start reading from"},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "Write",
            "description": "Create or overwrite a file with the given content. Creates parent directories automatically.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute or relative path"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["file_path", "content"],
            },
        },
        {
            "name": "Edit",
            "description": (
                "Replace old_string with new_string in a file. old_string must match exactly "
                "(including whitespace) and be unique unless replace_all is true. "
                "Returns a unified diff preview of the change."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string", "description": "Exact text to find and replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "default": False, "description": "Replace all occurrences"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
        {
            "name": "Bash",
            "description": (
                "Run a shell command. Working directory persists across calls (cd is tracked). "
                "Use sandbox=true for restricted mode that blocks dangerous commands."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": f"Seconds before timeout (default {BASH_TIMEOUT_SEC})"},
                    "description": {"type": "string", "description": "Short description of the command"},
                    "sandbox": {"type": "boolean", "default": False, "description": "Enable sandbox mode (blocks dangerous commands)"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "Glob",
            "description": "Find files matching a glob pattern. Auto-prepends **/ for simple patterns.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py' or '**/*.ts'"},
                    "path": {"type": "string", "description": "Base directory (default: project root)"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "Grep",
            "description": "Search file contents with regex (uses ripgrep when available).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search"},
                    "glob": {"type": "string", "description": "File filter glob, e.g. '*.py'"},
                    "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
                    "case_insensitive": {"type": "boolean"},
                    "head_limit": {"type": "integer", "description": "Max matching lines (default 250)"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "WebFetch",
            "description": "Fetch content from a URL and return it as readable text. HTML is auto-converted.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "prompt": {"type": "string", "description": "Hint for what to extract"},
                },
                "required": ["url"],
            },
        },
        {
            "name": "WebSearch",
            "description": "Search the web for information. Returns top results with titles, URLs, and snippets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "NotebookEdit",
            "description": "Edit a Jupyter notebook cell. Can create new cells or edit existing ones.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_notebook": {"type": "string", "description": "Path to the .ipynb file"},
                    "cell_idx": {"type": "integer", "description": "0-based cell index"},
                    "is_new_cell": {"type": "boolean", "default": False},
                    "cell_language": {"type": "string", "enum": ["python", "markdown", "raw", "r", "sql", "shell"]},
                    "old_string": {"type": "string", "description": "Text to replace in existing cell"},
                    "new_string": {"type": "string", "description": "Replacement text or new cell content"},
                },
                "required": ["target_notebook", "cell_idx", "new_string"],
            },
        },
        {
            "name": "TodoWrite",
            "description": (
                "Create or update a task list to track progress on complex tasks. "
                "Each todo has id, content, and status (pending/in_progress/completed/cancelled). "
                "Use merge=true to update existing todos by id."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "Array of todo items",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Unique identifier"},
                                "content": {"type": "string", "description": "Task description"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "cancelled"],
                                },
                            },
                            "required": ["id", "content", "status"],
                        },
                    },
                    "merge": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, merge with existing todos by id",
                    },
                },
                "required": ["todos"],
            },
        },
        {
            "name": "Agent",
            "description": (
                "Launch a sub-agent to handle a complex, multi-step task autonomously. "
                "The sub-agent gets its own conversation and cannot use the Agent tool. "
                "Optional **subagent_type**: general-purpose (default), Explore, Plan, or a "
                "custom name from .claude/agents/*.md."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The task for the sub-agent"},
                    "description": {"type": "string", "description": "Short description (3-5 words)"},
                    "subagent_type": {
                        "type": "string",
                        "description": (
                            "Agent profile: general-purpose, Explore, Plan, or custom id "
                            "(see system prompt sub-agent list)"
                        ),
                    },
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "Skill",
            "description": (
                "Execute a skill (custom prompt loaded from .claude/skills/ directories). "
                "Skills extend the agent with user-defined capabilities. "
                "IMPORTANT: Only use this for skills listed in the skill listing — do not guess."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": 'The skill name, e.g. "commit", "review-pr"'},
                    "args": {"type": "string", "description": "Optional arguments for the skill"},
                },
                "required": ["skill"],
            },
        },
    ]
