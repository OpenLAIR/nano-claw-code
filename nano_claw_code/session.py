"""Session save/load/resume with title support.

Ported from nano-claude-code TypeScript (utils/sessionStorage.ts, sessionTitle.ts).
Features: auto-title generation, manual naming, search by title.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from nano_claw_code.config import SESSIONS_DIR, ensure_dirs

_LATEST_LINK = SESSIONS_DIR / "_latest.json"


def _derive_title_from_message(text: str) -> str:
    """Derive a short title from the first user message (heuristic, no LLM)."""
    text = text.strip()
    first_line = text.split("\n")[0].strip()
    first_line = re.sub(r"^[#\-*>]+\s*", "", first_line)
    first_line = re.sub(r"\s+", " ", first_line).strip()
    if len(first_line) > 60:
        first_line = first_line[:57] + "..."
    return first_line or "(untitled)"


def _extract_first_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract text from the first non-empty user message."""
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        return text
    return ""


def generate_session_title(messages: list[dict[str, Any]]) -> str:
    """Generate a session title from conversation content."""
    text = _extract_first_user_text(messages)
    if not text:
        return "(untitled session)"
    return _derive_title_from_message(text)


def save_session(
    messages: list[dict[str, Any]],
    *,
    filename: str | None = None,
    turn_count: int = 0,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    model: str = "",
    title: str | None = None,
) -> Path:
    ensure_dirs()
    if not filename:
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = Path(filename) if "/" in filename else SESSIONS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_messages = _serialize_messages(messages)

    if title is None and messages:
        title = generate_session_title(messages)

    data = {
        "title": title or "",
        "messages": serializable_messages,
        "turn_count": turn_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "model": model,
        "saved_at": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, default=str))

    _update_latest_pointer(path)

    return path


def auto_save_session(
    messages: list[dict[str, Any]],
    *,
    turn_count: int = 0,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    model: str = "",
    title: str | None = None,
) -> Path | None:
    """Auto-save current session after each turn. Uses a stable filename per REPL session."""
    if not messages:
        return None
    return save_session(
        messages,
        filename="autosave_latest.json",
        turn_count=turn_count,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        model=model,
        title=title,
    )


def load_session(filename: str) -> dict[str, Any]:
    path = Path(filename) if "/" in filename else SESSIONS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")
    return json.loads(path.read_text())


def load_latest_session() -> dict[str, Any] | None:
    """Load the most recent session. Returns None if no session exists."""
    if _LATEST_LINK.exists():
        try:
            pointer = json.loads(_LATEST_LINK.read_text())
            target = Path(pointer.get("path", ""))
            if target.exists():
                return json.loads(target.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    sessions = list_sessions()
    if sessions:
        try:
            return json.loads(sessions[0].read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return None


def list_sessions() -> list[Path]:
    """List all session files sorted by modification time (newest first)."""
    ensure_dirs()
    return sorted(
        (p for p in SESSIONS_DIR.glob("*.json") if p.name != "_latest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def get_session_info(path: Path) -> dict[str, Any]:
    """Read lightweight metadata from a session file without loading all messages."""
    try:
        data = json.loads(path.read_text())
        return {
            "filename": path.name,
            "path": str(path),
            "title": data.get("title", ""),
            "turn_count": data.get("turn_count", 0),
            "model": data.get("model", ""),
            "saved_at": data.get("saved_at", ""),
            "message_count": len(data.get("messages", [])),
        }
    except (json.JSONDecodeError, OSError):
        return {"filename": path.name, "path": str(path), "title": "(corrupt)"}


def list_sessions_with_info() -> list[dict[str, Any]]:
    """List all sessions with their metadata, newest first."""
    sessions = list_sessions()
    return [get_session_info(s) for s in sessions]


def search_sessions(query: str) -> list[dict[str, Any]]:
    """Search sessions by title (case-insensitive substring match)."""
    query_lower = query.lower().strip()
    if not query_lower:
        return list_sessions_with_info()
    results = []
    for info in list_sessions_with_info():
        title = info.get("title", "").lower()
        filename = info.get("filename", "").lower()
        if query_lower in title or query_lower in filename:
            results.append(info)
    return results


def rename_session(filename: str, new_title: str) -> bool:
    """Rename a session by updating its title field."""
    path = Path(filename) if "/" in filename else SESSIONS_DIR / filename
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        data["title"] = new_title
        path.write_text(json.dumps(data, indent=2, default=str))
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serializable = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            blocks = []
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "image" and b.get("source", {}).get("type") == "base64":
                        blocks.append({
                            "type": "text",
                            "text": f"[image: {b.get('source', {}).get('media_type', 'unknown')}]",
                        })
                    else:
                        blocks.append(b)
                elif hasattr(b, "model_dump"):
                    blocks.append(b.model_dump(mode="json"))
                else:
                    blocks.append({"type": "text", "text": str(b)})
            serializable.append({**m, "content": blocks})
        else:
            serializable.append(m)
    return serializable


def _update_latest_pointer(path: Path) -> None:
    try:
        _LATEST_LINK.write_text(json.dumps({"path": str(path.resolve())}))
    except OSError:
        pass
