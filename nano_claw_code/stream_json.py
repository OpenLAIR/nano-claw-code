"""NDJSON stream-json output compatible with swebench_harness/run_swebench_claude_code.py.

Each line on stdout is a JSON object. Event types:
  - assistant: model response (content blocks: text, thinking, tool_use)
  - user: tool results fed back to the model
  - result: final outcome (success, error_during_execution, error_max_turns)
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_uuid() -> str:
    return str(uuid.uuid4())


def write_event(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


# ── Assistant message ─────────────────────────────────────────────────────


def emit_assistant(
    *,
    message: dict[str, Any],
    request_id: str | None = None,
) -> str:
    uid = new_uuid()
    evt: dict[str, Any] = {
        "type": "assistant",
        "uuid": uid,
        "timestamp": _now_iso(),
        "message": message,
    }
    if request_id:
        evt["requestId"] = request_id
    write_event(evt)
    return uid


# ── Streaming delta (optional — for real-time partial output) ─────────────


def emit_stream_delta(
    *,
    delta_type: str,
    content: str,
    tool_name: str | None = None,
) -> None:
    evt: dict[str, Any] = {
        "type": "stream_delta",
        "uuid": new_uuid(),
        "timestamp": _now_iso(),
        "delta_type": delta_type,
        "content": content,
    }
    if tool_name:
        evt["tool_name"] = tool_name
    write_event(evt)


# ── User tool results ────────────────────────────────────────────────────


def emit_user_tool_results(content_blocks: list[dict[str, Any]]) -> str:
    uid = new_uuid()
    write_event(
        {
            "type": "user",
            "uuid": uid,
            "timestamp": _now_iso(),
            "message": {"role": "user", "content": content_blocks},
        }
    )
    return uid


# ── Final result ──────────────────────────────────────────────────────────


def emit_result(
    *,
    subtype: str,
    is_error: bool,
    num_turns: int,
    duration_ms: int,
    duration_api_ms: int = 0,
    result_text: str = "",
    total_cost_usd: float = 0.0,
    usage: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> None:
    usage = usage or {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    write_event(
        {
            "type": "result",
            "subtype": subtype,
            "duration_ms": duration_ms,
            "duration_api_ms": duration_api_ms,
            "is_error": is_error,
            "num_turns": num_turns,
            "stop_reason": None,
            "session_id": new_uuid(),
            "total_cost_usd": total_cost_usd,
            "usage": usage,
            "modelUsage": {},
            "permission_denials": [],
            "uuid": new_uuid(),
            **({"result": result_text} if result_text else {}),
            **({"errors": errors or []} if errors else {}),
        }
    )


# ── Helpers ───────────────────────────────────────────────────────────────


def api_message_to_stream_message(msg: Any) -> dict[str, Any]:
    """Convert anthropic.types.Message to Claude Code-shaped message dict."""
    blocks: list[dict[str, Any]] = []
    for block in msg.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            blocks.append({"type": "text", "text": block.text})
        elif btype == "thinking":
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                }
            )
        elif btype == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input if isinstance(block.input, dict) else {},
                }
            )
        else:
            blocks.append({"type": str(btype), "raw": str(block)})
    return {
        "id": msg.id,
        "model": msg.model,
        "role": "assistant",
        "stop_reason": msg.stop_reason,
        "usage": msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else dict(msg.usage),
        "content": blocks,
    }


def streaming_message_to_stream_message(
    *,
    message_id: str,
    model: str,
    stop_reason: str | None,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    """Build a Claude Code-shaped message dict from accumulated streaming data."""
    return {
        "id": message_id,
        "model": model,
        "role": "assistant",
        "stop_reason": stop_reason,
        "usage": usage,
        "content": content_blocks,
    }
