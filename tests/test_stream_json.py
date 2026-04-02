"""stream_json event helpers (stdout captured)."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

from nano_claw_code.stream_json import (
    emit_assistant,
    emit_result,
    emit_user_tool_results,
    new_uuid,
)


def test_new_uuid_format():
    u = new_uuid()
    assert len(u) == 36


def test_emit_assistant_json_line():
    buf = StringIO()
    with patch("nano_claw_code.stream_json.sys.stdout", buf):
        emit_assistant(
            message={"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        )
    line = buf.getvalue().strip()
    obj = json.loads(line)
    assert obj["type"] == "assistant"
    assert obj["message"]["role"] == "assistant"


def test_emit_result_structure():
    buf = StringIO()
    with patch("nano_claw_code.stream_json.sys.stdout", buf):
        emit_result(
            subtype="success",
            is_error=False,
            num_turns=1,
            duration_ms=10,
            result_text="done",
        )
    obj = json.loads(buf.getvalue().strip())
    assert obj["type"] == "result"
    assert obj["subtype"] == "success"
    assert obj["result"] == "done"


def test_emit_user_tool_results():
    buf = StringIO()
    with patch("nano_claw_code.stream_json.sys.stdout", buf):
        uid = emit_user_tool_results(
            [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]
        )
    assert len(uid) == 36
    obj = json.loads(buf.getvalue().strip())
    assert obj["type"] == "user"
