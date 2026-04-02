"""OpenAI Chat Completions backend for Azure AI / Kimi / other OpenAI-compatible endpoints.

Env: OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_MODEL
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from nano_claw_code.permissions import describe_permission, needs_permission
from nano_claw_code.prompts import build_system_prompt
from nano_claw_code.stream_json import (
    api_message_to_stream_message,
    emit_assistant,
    emit_result,
    emit_user_tool_results,
)
from nano_claw_code.tools_impl import anthropic_tool_defs, dispatch_tool

from nano_claw_code.agent import (
    AgentState,
    CompactionNotice,
    MAX_CONTINUATIONS,
    PermissionRequest,
    TextChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
    _needs_compaction,
)

# ── Retry (mirror agent.py) ─────────────────────────────────────────────

MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 60.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

CONTEXT_WINDOW_TOKENS = 200_000
COMPACTION_THRESHOLD = 0.75
COMPACTION_KEEP_RECENT = 6

_COMPACTION_PROMPT = (
    "The conversation is getting long. Please provide a brief summary of what has been "
    "accomplished so far and what the current task is. Be concise — focus on files changed, "
    "key decisions made, and the current goal. This will replace older messages."
)


def _should_retry_openai(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", None)
        return code in _RETRYABLE_STATUS_CODES if code is not None else False
    return False


def _retry_delay(attempt: int, exc: Exception) -> float:
    delay = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
    return min(delay, RETRY_MAX_DELAY)


def anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        name = t.get("name", "")
        desc = t.get("description", "")
        schema = t.get("input_schema") or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (desc or "")[:4096],
                    "parameters": schema,
                },
            }
        )
    return out


def _messages_to_openai_chat(
    system_prompt: str, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build OpenAI `messages` from Nano Claw dict-shaped history."""
    oai: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            if isinstance(content, str):
                oai.append({"role": "user", "content": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        tid = block.get("tool_use_id", "")
                        body = block.get("content", "")
                        if not isinstance(body, str):
                            body = json.dumps(body, ensure_ascii=False)
                        oai.append({"role": "tool", "tool_call_id": tid, "content": body})
        elif role == "assistant":
            if not isinstance(content, list):
                continue
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                            },
                        }
                    )
            msg: dict[str, Any] = {"role": "assistant"}
            joined = "\n".join(text_parts).strip()
            msg["content"] = joined if joined else None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            oai.append(msg)
    return oai


def _choice_to_assistant_blocks(choice: Any) -> list[dict[str, Any]]:
    msg = choice.message
    blocks: list[dict[str, Any]] = []
    c = getattr(msg, "content", None)
    if c:
        blocks.append({"type": "text", "text": c})
    for tc in getattr(msg, "tool_calls", None) or []:
        fn = tc.function
        raw_args = getattr(fn, "arguments", None) or "{}"
        try:
            inp = json.loads(raw_args) if isinstance(raw_args, str) else {}
        except json.JSONDecodeError:
            inp = {"_raw_arguments": raw_args}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": fn.name,
                "input": inp if isinstance(inp, dict) else {},
            }
        )
    return blocks


def _blocks_to_fake_message(
    resp_id: str,
    model: str,
    finish_reason: str | None,
    blocks: list[dict[str, Any]],
    usage: dict[str, int],
) -> Any:
    """Minimal object graph for api_message_to_stream_message."""
    parts: list[Any] = []
    for b in blocks:
        if b["type"] == "text":
            parts.append(SimpleNamespace(type="text", text=b.get("text", "")))
        elif b["type"] == "tool_use":
            parts.append(
                SimpleNamespace(
                    type="tool_use",
                    id=b.get("id", ""),
                    name=b.get("name", ""),
                    input=b.get("input", {}),
                )
            )
    stop = "tool_use" if finish_reason == "tool_calls" else "end_turn"
    u = SimpleNamespace(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )

    class _Msg:
        pass

    m = _Msg()
    m.id = resp_id
    m.model = model
    m.stop_reason = stop
    m.content = parts
    m.usage = u
    return m


def _usage_from_openai(u: Any) -> dict[str, int]:
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0}
    pt = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None) or 0
    ct = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None) or 0
    return {"input_tokens": int(pt), "output_tokens": int(ct)}


def _accumulate_usage(total: dict[str, int], part: dict[str, Any]) -> None:
    for k in total:
        total[k] += int(part.get(k, 0))


def _chat_create(client: OpenAI, **kwargs: Any) -> Any:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except (APIError, APIConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRIES and _should_retry_openai(e):
                time.sleep(_retry_delay(attempt, e))
                continue
            raise
    raise last_exc  # type: ignore[misc]


def compact_messages_openai(
    messages: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    system_prompt: str,
) -> list[dict[str, Any]]:
    if len(messages) <= COMPACTION_KEEP_RECENT + 2:
        return messages
    recent = messages[-COMPACTION_KEEP_RECENT:]
    old = messages[:-COMPACTION_KEEP_RECENT]
    old_text_parts: list[str] = []
    for m in old:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            old_text_parts.append(f"[{role}] {content[:500]}")
        elif isinstance(content, list):
            bits = []
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        bits.append(b.get("text", "")[:300])
                    elif b.get("type") == "tool_use":
                        bits.append(f"[tool: {b.get('name', '?')}]")
            old_text_parts.append(f"[{role}] {' '.join(bits)[:500]}")
    conversation_so_far = "\n".join(old_text_parts[-20:])
    try:
        r = _chat_create(
            client,
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "You summarize conversations concisely."},
                {
                    "role": "user",
                    "content": (
                        f"Summarize this conversation history in 2-3 paragraphs.\n\n{conversation_so_far}"
                    ),
                },
            ],
        )
        summary = (r.choices[0].message.content or "Previous conversation context.").strip()
    except Exception:
        summary = f"[Compacted {len(old)} earlier messages]"
    compacted = [
        {
            "role": "user",
            "content": f"[Context from earlier in our conversation]\n{summary}",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Understood. I have the context from our earlier conversation.",
                }
            ],
        },
    ]
    return compacted + recent


def run_streaming_openai(
    user_message: str,
    state: AgentState,
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    tools: list[dict],
    max_tokens: int = 16_384,
    cwd: Path,
    thinking: bool = False,
    thinking_budget: int = 10_000,
    permission_mode: str = "accept-all",
    enable_cache: bool = True,
) -> Generator:
    del thinking, thinking_budget, enable_cache  # not supported on OpenAI-compat path

    if _needs_compaction(state.messages, state.last_input_tokens):
        old_count = len(state.messages)
        state.messages = compact_messages_openai(state.messages, client, model, system_prompt)
        yield CompactionNotice(old_count, len(state.messages))

    state.messages.append({"role": "user", "content": user_message})
    oai_tools = anthropic_tools_to_openai(tools)
    continuations = 0

    while True:
        state.turn_count += 1
        api_messages = _messages_to_openai_chat(system_prompt, state.messages)

        tool_uses: list[dict[str, Any]] = []
        in_tokens = out_tokens = 0
        final_blocks: list[dict[str, Any]] = []
        finish_reason: str | None = None
        resp_id = ""

        try:
            resp = _chat_create(
                client,
                model=model,
                max_tokens=max_tokens,
                messages=api_messages,
                tools=oai_tools if oai_tools else None,
                tool_choice="auto" if oai_tools else None,
            )
            choice = resp.choices[0]
            finish_reason = choice.finish_reason
            resp_id = getattr(resp, "id", "") or "openai"
            final_blocks = _choice_to_assistant_blocks(choice)
            u = _usage_from_openai(resp.usage)
            in_tokens = u["input_tokens"]
            out_tokens = u["output_tokens"]
            state.total_input_tokens += in_tokens
            state.total_output_tokens += out_tokens
            state.last_input_tokens = in_tokens

            for b in final_blocks:
                if b["type"] == "text":
                    yield TextChunk(b.get("text", ""))
                elif b["type"] == "tool_use":
                    tool_uses.append(b)

            state.messages.append({"role": "assistant", "content": final_blocks})
        except KeyboardInterrupt:
            yield TextChunk("\n[interrupted by user]\n")
            break
        except (APIError, APIConnectionError) as e:
            yield TextChunk(f"\n[API Error: {e}]\n")
            break

        yield TurnDone(in_tokens, out_tokens, 0, 0)

        if finish_reason == "length" and not tool_uses:
            continuations += 1
            if continuations <= MAX_CONTINUATIONS:
                state.messages.append(
                    {
                        "role": "user",
                        "content": "Please continue from where you left off. Do not repeat what you already said.",
                    }
                )
                continue
            yield TextChunk("\n[Reached max continuations limit]\n")
            break

        continuations = 0

        if finish_reason != "tool_calls" or not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            raw_in = tu.get("input", {}) if isinstance(tu.get("input"), dict) else {}
            name = tu.get("name", "")

            yield ToolStart(name, raw_in)

            if needs_permission(name, raw_in, permission_mode):
                desc = describe_permission(name, raw_in)
                req = PermissionRequest(description=desc)
                yield req
                if not req.granted:
                    result = "Denied: user rejected this operation"
                    yield ToolEnd(name, result, permitted=False)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.get("id", ""),
                            "content": result,
                            "is_error": True,
                        }
                    )
                    continue

            try:
                result = dispatch_tool(cwd, name, raw_in)
            except KeyboardInterrupt:
                result = "Error: operation interrupted by user"
            is_err = isinstance(result, str) and result.startswith("Error:")
            display = result if isinstance(result, str) else f"[{name}: rich content]"
            yield ToolEnd(name, display, permitted=True)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.get("id", ""),
                    "content": result,
                    "is_error": is_err,
                }
            )

        state.messages.append({"role": "user", "content": tool_results})


def run_agent_loop_openai(
    *,
    cwd: Path,
    user_prompt: str,
    model: str | None,
    max_turns: int,
    bare: bool,
    verbose: bool,
    streaming: bool = False,
    thinking: bool = False,
    thinking_budget: int = 10_000,
) -> int:
    from nano_claw_code.prompts import resolve_model as _resolve_model

    del verbose, streaming, thinking, thinking_budget
    cwd = cwd.resolve()
    model_id = _resolve_model(model)
    system = build_system_prompt(cwd=str(cwd), bare=bare)
    tools = anthropic_tool_defs()
    oai_tools = anthropic_tools_to_openai(tools)

    from nano_claw_code.config import resolve_api_env

    api_env = resolve_api_env()
    if not api_env.get("api_key") or api_env.get("provider") != "openai_compat":
        emit_result(
            subtype="error_during_execution",
            is_error=True,
            num_turns=0,
            duration_ms=0,
            errors=["OpenAI-compat provider not configured (OPENAI_COMPAT_*)"],
        )
        return 1

    client = OpenAI(api_key=api_env["api_key"], base_url=api_env["base_url"])
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    total_usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    t0 = time.time()
    total_api_ms = 0
    turns = 0
    continuations = 0

    for turn in range(max_turns):
        turns = turn + 1
        api_ms_start = time.time()
        api_messages = _messages_to_openai_chat(system, messages)

        try:
            resp = _chat_create(
                client,
                model=model_id,
                max_tokens=16_384,
                messages=api_messages,
                tools=oai_tools,
                tool_choice="auto",
            )
        except (APIError, APIConnectionError) as e:
            emit_result(
                subtype="error_during_execution",
                is_error=True,
                num_turns=turns,
                duration_ms=int((time.time() - t0) * 1000),
                errors=[str(e)],
                usage=total_usage,
            )
            return 1

        api_ms = int((time.time() - api_ms_start) * 1000)
        total_api_ms += api_ms

        choice = resp.choices[0]
        blocks = _choice_to_assistant_blocks(choice)
        u = _usage_from_openai(resp.usage)
        _accumulate_usage(
            total_usage,
            {
                "input_tokens": u["input_tokens"],
                "output_tokens": u["output_tokens"],
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )

        fake = _blocks_to_fake_message(
            getattr(resp, "id", "") or "oai",
            model_id,
            choice.finish_reason,
            blocks,
            u,
        )
        inner = api_message_to_stream_message(fake)
        emit_assistant(message=inner, request_id=getattr(resp, "id", None))

        text_parts = [b["text"] for b in blocks if b["type"] == "text"]
        tool_uses = [b for b in blocks if b["type"] == "tool_use"]
        combined_text = "\n".join(text_parts)

        if choice.finish_reason == "length" and not tool_uses:
            continuations += 1
            if continuations <= MAX_CONTINUATIONS:
                messages.append({"role": "assistant", "content": blocks})
                messages.append(
                    {
                        "role": "user",
                        "content": "Please continue from where you left off. Do not repeat what you already said.",
                    }
                )
                continue

        continuations = 0

        if choice.finish_reason != "tool_calls" and not tool_uses:
            emit_result(
                subtype="success",
                is_error=False,
                num_turns=turns,
                duration_ms=int((time.time() - t0) * 1000),
                duration_api_ms=total_api_ms,
                result_text=combined_text.strip() or "(no text)",
                usage=total_usage,
            )
            return 0

        if not tool_uses:
            emit_result(
                subtype="success",
                is_error=False,
                num_turns=turns,
                duration_ms=int((time.time() - t0) * 1000),
                duration_api_ms=total_api_ms,
                result_text=combined_text.strip() or "(no tool calls; stopping)",
                usage=total_usage,
            )
            return 0

        result_blocks: list[dict[str, Any]] = []
        for tu in tool_uses:
            raw_in = tu.get("input", {}) if isinstance(tu, dict) else {}
            name = tu.get("name", "")
            out = dispatch_tool(cwd, name, raw_in)
            is_err = isinstance(out, str) and out.startswith("Error:")
            result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.get("id", ""),
                    "content": out,
                    "is_error": is_err,
                }
            )

        emit_user_tool_results(result_blocks)
        messages.append({"role": "assistant", "content": blocks})
        messages.append({"role": "user", "content": result_blocks})

    emit_result(
        subtype="error_max_turns",
        is_error=True,
        num_turns=max_turns,
        duration_ms=int((time.time() - t0) * 1000),
        duration_api_ms=total_api_ms,
        errors=[f"Reached max turns ({max_turns})"],
        usage=total_usage,
    )
    return 1
