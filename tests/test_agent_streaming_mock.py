"""run_streaming with mocked Anthropic stream (no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from nano_claw_code.agent import AgentState, PermissionRequest, ToolEnd, ToolStart, TurnDone, run_streaming


class _FakeStream:
    def __init__(self, final):
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self._final


def _msg_tool_then_text(tool_name="Read", tool_input=None):
    tool_input = tool_input or {"file_path": "x.txt"}
    tu = SimpleNamespace(
        type="tool_use",
        id="toolu_01",
        name=tool_name,
        input=tool_input,
    )
    usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 1, "output_tokens": 0})
    return SimpleNamespace(
        content=[tu],
        stop_reason="tool_use",
        usage=usage,
    )


def _msg_end_text(text="done"):
    tb = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 2, "output_tokens": 1})
    return SimpleNamespace(
        content=[tb],
        stop_reason="end_turn",
        usage=usage,
    )


def test_run_streaming_end_turn(tmp_path: Path):
    final = _msg_end_text("hello")
    stream = _FakeStream(final)
    client = MagicMock()
    client.messages.stream.return_value = stream

    state = AgentState()
    events = list(
        run_streaming(
            "hi",
            state,
            client=client,
            model="claude-test",
            system_prompt="sys",
            tools=[],
            cwd=tmp_path,
            permission_mode="accept-all",
            enable_cache=False,
        )
    )
    assert any(isinstance(e, TurnDone) for e in events)
    # Streaming yields TextChunk only from deltas; final text lives on the message.
    assert len(state.messages) >= 2
    assert any("hello" in str(m) for m in state.messages)


def test_run_streaming_tool_use_read(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("inside", encoding="utf-8")
    s1 = _FakeStream(_msg_tool_then_text("Read", {"file_path": str(f)}))
    s2 = _FakeStream(_msg_end_text("done"))
    client = MagicMock()
    client.messages.stream.side_effect = [s1, s2]

    state = AgentState()
    events = list(
        run_streaming(
            "read file",
            state,
            client=client,
            model="claude-test",
            system_prompt="sys",
            tools=[{"name": "Read", "input_schema": {"type": "object"}}],
            cwd=tmp_path,
            permission_mode="accept-all",
            enable_cache=False,
        )
    )
    assert any(isinstance(e, ToolStart) for e in events)
    assert any(isinstance(e, ToolEnd) and "inside" in e.result for e in events)
    assert state.messages[-2]["role"] == "user"
    assert state.messages[-1]["role"] == "assistant"


def test_run_streaming_permission_denied(tmp_path: Path):
    s1 = _FakeStream(_msg_tool_then_text("Write", {"file_path": "nope.txt", "content": "x"}))
    s2 = _FakeStream(_msg_end_text("after deny"))
    client = MagicMock()
    client.messages.stream.side_effect = [s1, s2]

    state = AgentState()
    gen = run_streaming(
        "write",
        state,
        client=client,
        model="claude-test",
        system_prompt="sys",
        tools=[{"name": "Write", "input_schema": {"type": "object"}}],
        cwd=tmp_path,
        permission_mode="manual",
        enable_cache=False,
    )
    ev = next(gen)
    while not isinstance(ev, PermissionRequest):
        ev = next(gen)
    assert isinstance(ev, PermissionRequest)
    ev.granted = False
    rest = list(gen)
    assert any(isinstance(e, ToolEnd) and not e.permitted for e in rest)
