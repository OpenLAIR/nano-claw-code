"""Core agent loop: Anthropic Messages API with streaming, tool dispatch, permissions, cost.

Ported from nano-claude-code TypeScript (query.ts, services/api/claude.ts, agent.py).
Supports streaming and non-streaming modes, extended thinking, permission gates,
API retry with exponential backoff, max_tokens auto-continuation, Ctrl+C graceful
interrupt, prompt caching, context compaction, and NDJSON stream-json output.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import anthropic

from nano_claw_code.config import calc_cost, resolve_api_env
from nano_claw_code.permissions import describe_permission, needs_permission
from nano_claw_code.prompts import build_system_prompt, resolve_model
from nano_claw_code.stream_json import (
    api_message_to_stream_message,
    emit_assistant,
    emit_result,
    emit_stream_delta,
    emit_user_tool_results,
)
from nano_claw_code.tools_impl import anthropic_tool_defs, dispatch_tool


# ── Retry configuration ──────────────────────────────────────────────────

MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 60.0

_RETRYABLE_STATUS_CODES = {429, 529, 500, 502, 503, 504}


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.InternalServerError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return getattr(exc, "status_code", 0) in _RETRYABLE_STATUS_CODES
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    return False


def _get_retry_after(exc: Exception) -> float | None:
    if isinstance(exc, anthropic.APIStatusError):
        headers = getattr(exc, "response", None)
        if headers and hasattr(headers, "headers"):
            ra = headers.headers.get("retry-after")
            if ra:
                try:
                    return float(ra)
                except ValueError:
                    pass
    return None


def _retry_delay(attempt: int, exc: Exception) -> float:
    server_delay = _get_retry_after(exc)
    if server_delay is not None:
        return min(server_delay, RETRY_MAX_DELAY)
    delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
    return min(delay, RETRY_MAX_DELAY)


# ── Context window / compaction ───────────────────────────────────────────

CONTEXT_WINDOW_TOKENS = 200_000
COMPACTION_THRESHOLD = 0.75
COMPACTION_KEEP_RECENT = 6

_COMPACTION_PROMPT = (
    "The conversation is getting long. Please provide a brief summary of what has been "
    "accomplished so far and what the current task is. Be concise — focus on files changed, "
    "key decisions made, and the current goal. This will replace older messages."
)


def _estimate_message_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token for text, fixed cost per tool block."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        total += len(block.get("text", "")) // 4
                    elif btype == "tool_use":
                        total += len(json.dumps(block.get("input", {}))) // 4 + 50
                    elif btype == "tool_result":
                        c = block.get("content", "")
                        total += (len(c) if isinstance(c, str) else len(str(c))) // 4
                    elif btype == "image":
                        total += 1000
                    else:
                        total += 100
                elif hasattr(block, "model_dump"):
                    total += 200
                else:
                    total += len(str(block)) // 4
    return total


def _needs_compaction(messages: list[dict], last_input_tokens: int) -> bool:
    if last_input_tokens > 0:
        return last_input_tokens > int(CONTEXT_WINDOW_TOKENS * COMPACTION_THRESHOLD)
    return _estimate_message_tokens(messages) > int(CONTEXT_WINDOW_TOKENS * COMPACTION_THRESHOLD)


def compact_messages(
    messages: list[dict],
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
) -> list[dict]:
    """Summarize old messages to reduce context size, keeping recent ones intact."""
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
            summary_parts = []
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        summary_parts.append(b.get("text", "")[:300])
                    elif b.get("type") == "tool_use":
                        summary_parts.append(f"[tool: {b.get('name', '?')}]")
                    elif b.get("type") == "tool_result":
                        c = b.get("content", "")
                        preview = c[:200] if isinstance(c, str) else "[rich content]"
                        summary_parts.append(f"[result: {preview}]")
            old_text_parts.append(f"[{role}] {' '.join(summary_parts)[:500]}")

    conversation_so_far = "\n".join(old_text_parts[-20:])

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system="You are a helpful assistant that summarizes conversations concisely.",
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this conversation history in 2-3 paragraphs. Focus on: "
                    f"files modified, key decisions, current task state.\n\n{conversation_so_far}"
                ),
            }],
        )
        summary = resp.content[0].text if resp.content else "Previous conversation context."
    except Exception:
        summary = f"[Compacted {len(old)} earlier messages]"

    compacted = [
        {"role": "user", "content": f"[Context from earlier in our conversation]\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from our earlier conversation. Let me continue from where we left off."},
    ]
    return compacted + recent


# ── Prompt caching ────────────────────────────────────────────────────────

def _build_cached_system(system_prompt: str) -> list[dict]:
    """Wrap system prompt with cache_control for Anthropic prompt caching.

    Returns the system as a list of content blocks with cache breakpoints.
    """
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _add_cache_breakpoints(messages: list[dict]) -> list[dict]:
    """Add cache_control breakpoints to the last user message for prompt caching.

    Anthropic allows up to 4 cache breakpoints. We place one on the system prompt
    (handled separately) and one on the most recent user turn.
    """
    if not messages:
        return messages

    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        msg = result[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            result[i] = {
                **msg,
                "content": [{
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }],
            }
        elif isinstance(content, list) and content:
            new_content = list(content)
            last_block = new_content[-1]
            if isinstance(last_block, dict) and "cache_control" not in last_block:
                new_content[-1] = {**last_block, "cache_control": {"type": "ephemeral"}}
            result[i] = {**msg, "content": new_content}
        break
    return result


# ── Usage helpers ─────────────────────────────────────────────────────────


def _usage_dict(u: Any) -> dict[str, Any]:
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    if hasattr(u, "model_dump"):
        d = u.model_dump()
    else:
        d = dict(u) if isinstance(u, dict) else {}
    return {
        "input_tokens": int(d.get("input_tokens", 0)),
        "output_tokens": int(d.get("output_tokens", 0)),
        "cache_creation_input_tokens": int(d.get("cache_creation_input_tokens", 0)),
        "cache_read_input_tokens": int(d.get("cache_read_input_tokens", 0)),
    }


def _accumulate_usage(total: dict[str, int], part: dict[str, Any]) -> None:
    for k in total:
        total[k] += int(part.get(k, 0))


# ── Event types for REPL rendering ───────────────────────────────────────


@dataclass
class TextChunk:
    text: str

@dataclass
class ThinkingChunk:
    text: str

@dataclass
class ToolStart:
    name: str
    inputs: dict

@dataclass
class ToolEnd:
    name: str
    result: str
    permitted: bool = True

@dataclass
class TurnDone:
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

@dataclass
class PermissionRequest:
    """Yielded when permission is needed. Caller sets .granted before resuming."""
    description: str
    granted: bool = False

@dataclass
class CompactionNotice:
    """Yielded when context compaction occurs."""
    old_count: int
    new_count: int


@dataclass
class AgentState:
    messages: list = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turn_count: int = 0
    last_input_tokens: int = 0


# ── API call with retry ──────────────────────────────────────────────────


def _api_call_streaming(client: anthropic.Anthropic, kwargs: dict, *, on_retry=None):
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.messages.stream(**kwargs)
        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRIES and _should_retry(e):
                delay = _retry_delay(attempt, e)
                if on_retry:
                    on_retry(attempt + 1, MAX_RETRIES, delay, e)
                time.sleep(delay)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _api_call_create(client: anthropic.Anthropic, kwargs: dict, *, on_retry=None):
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)
        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRIES and _should_retry(e):
                delay = _retry_delay(attempt, e)
                if on_retry:
                    on_retry(attempt + 1, MAX_RETRIES, delay, e)
                time.sleep(delay)
                continue
            raise
    raise last_exc  # type: ignore[misc]


# ── Streaming agent loop (for interactive REPL) ──────────────────────────

MAX_CONTINUATIONS = 3


def run_streaming(
    user_message: str,
    state: AgentState,
    *,
    client: anthropic.Anthropic,
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
    """Generator-based streaming agent loop.

    Features: retry, auto-continuation, Ctrl+C interrupt, prompt caching,
    automatic context compaction.
    """
    # ── Context compaction check (before adding new message) ──
    if _needs_compaction(state.messages, state.last_input_tokens):
        old_count = len(state.messages)
        state.messages = compact_messages(state.messages, client, model, system_prompt)
        yield CompactionNotice(old_count, len(state.messages))

    state.messages.append({"role": "user", "content": user_message})
    continuations = 0

    cached_system = _build_cached_system(system_prompt) if enable_cache else system_prompt

    while True:
        state.turn_count += 1

        msgs_for_api = _add_cache_breakpoints(state.messages) if enable_cache else state.messages

        kwargs: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens, "system": cached_system,
            "messages": msgs_for_api, "tools": tools,
        }
        if thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        tool_uses: list[Any] = []
        in_tokens = out_tokens = 0
        cache_creation = cache_read = 0
        final = None

        try:
            stream_ctx = _api_call_streaming(
                client, kwargs,
                on_retry=lambda a, m, d, e: None,
            )
            with stream_ctx as stream:
                for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "content_block_delta":
                        delta = event.delta
                        dtype = getattr(delta, "type", None)
                        if dtype == "text_delta":
                            yield TextChunk(delta.text)
                        elif dtype == "thinking_delta":
                            yield ThinkingChunk(delta.thinking)

                final = stream.get_final_message()
                usage = _usage_dict(getattr(final, "usage", None))
                in_tokens = usage["input_tokens"]
                out_tokens = usage["output_tokens"]
                cache_creation = usage["cache_creation_input_tokens"]
                cache_read = usage["cache_read_input_tokens"]
                state.total_input_tokens += in_tokens
                state.total_output_tokens += out_tokens
                state.last_input_tokens = in_tokens

                for block in final.content:
                    if block.type == "tool_use":
                        tool_uses.append(block)

                state.messages.append({"role": "assistant", "content": final.content})
        except KeyboardInterrupt:
            yield TextChunk("\n[interrupted by user]\n")
            break
        except anthropic.APIError as e:
            yield TextChunk(f"\n[API Error: {e}]\n")
            break

        yield TurnDone(in_tokens, out_tokens, cache_creation, cache_read)

        if final and final.stop_reason == "max_tokens" and not tool_uses:
            continuations += 1
            if continuations <= MAX_CONTINUATIONS:
                yield TextChunk("")
                state.messages.append({
                    "role": "user",
                    "content": "Please continue from where you left off. Do not repeat what you already said.",
                })
                continue
            else:
                yield TextChunk("\n[Reached max continuations limit]\n")
                break

        continuations = 0

        if final is None or final.stop_reason != "tool_use" or not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            raw_in = tu.input if isinstance(tu.input, dict) else {}

            yield ToolStart(tu.name, raw_in)

            if needs_permission(tu.name, raw_in, permission_mode):
                desc = describe_permission(tu.name, raw_in)
                req = PermissionRequest(description=desc)
                yield req
                if not req.granted:
                    result = "Denied: user rejected this operation"
                    yield ToolEnd(tu.name, result, permitted=False)
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tu.id,
                        "content": result, "is_error": True,
                    })
                    continue

            try:
                result = dispatch_tool(cwd, tu.name, raw_in)
            except KeyboardInterrupt:
                result = "Error: operation interrupted by user"
            is_err = isinstance(result, str) and result.startswith("Error:")
            display = result if isinstance(result, str) else f"[{tu.name}: rich content ({len(result)} blocks)]"
            yield ToolEnd(tu.name, display, permitted=True)
            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": result, "is_error": is_err,
            })

        state.messages.append({"role": "user", "content": tool_results})


# ── Non-streaming agent loop (for harness / --print mode) ────────────────


def run_agent_loop(
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
    """Run the agent loop with stream-json output for the SWE-bench harness."""
    cwd = cwd.resolve()
    model_id = resolve_model(model)
    system = build_system_prompt(cwd=str(cwd), bare=bare)
    tools = anthropic_tool_defs()

    api_env = resolve_api_env()
    if not api_env.get("api_key"):
        emit_result(subtype="error_during_execution", is_error=True, num_turns=0,
                    duration_ms=0, errors=[
                        "No API key found. Set OPENAI_COMPAT_* or ANTHROPIC_API_KEY or OPENROUTER_API_KEY in .env or env",
                    ])
        return 1

    if api_env.get("provider") == "openai_compat":
        from nano_claw_code.openai_compat import run_agent_loop_openai

        return run_agent_loop_openai(
            cwd=cwd,
            user_prompt=user_prompt,
            model=model,
            max_turns=max_turns,
            bare=bare,
            verbose=verbose,
            streaming=streaming,
            thinking=thinking,
            thinking_budget=thinking_budget,
        )

    api_env.pop("provider", None)
    client = anthropic.Anthropic(**api_env)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    total_usage: dict[str, int] = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    }
    t0 = time.time()
    total_api_ms = 0
    turns = 0
    continuations = 0

    cached_system = _build_cached_system(system)

    for turn in range(max_turns):
        turns = turn + 1
        api_ms_start = time.time()

        msgs_for_api = _add_cache_breakpoints(messages)

        create_kwargs: dict[str, Any] = {
            "model": model_id, "max_tokens": 16_384, "system": cached_system,
            "tools": tools, "messages": msgs_for_api,
        }
        if thinking:
            create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        def on_retry(attempt, max_r, delay, exc):
            print(f"[Retry {attempt}/{max_r} in {delay:.1f}s: {exc}]", file=sys.stderr)

        if streaming:
            resp = _stream_one_turn(client, create_kwargs, verbose=verbose, on_retry=on_retry)
            if resp is None:
                emit_result(subtype="error_during_execution", is_error=True, num_turns=turns,
                            duration_ms=int((time.time() - t0) * 1000),
                            errors=["Streaming API call failed"], usage=total_usage)
                return 1
        else:
            try:
                resp = _api_call_create(client, create_kwargs, on_retry=on_retry)
            except (anthropic.APIError, anthropic.APIConnectionError) as e:
                emit_result(subtype="error_during_execution", is_error=True, num_turns=turns,
                            duration_ms=int((time.time() - t0) * 1000),
                            errors=[str(e)], usage=total_usage)
                return 1

        api_ms = int((time.time() - api_ms_start) * 1000)
        total_api_ms += api_ms

        u = _usage_dict(getattr(resp, "usage", None))
        _accumulate_usage(total_usage, u)

        inner = api_message_to_stream_message(resp)
        emit_assistant(message=inner, request_id=getattr(resp, "id", None))

        blocks = list(resp.content)
        text_parts = [b.text for b in blocks if b.type == "text"]
        tool_uses = [b for b in blocks if b.type == "tool_use"]
        combined_text = "\n".join(text_parts)

        if resp.stop_reason == "max_tokens" and not tool_uses:
            continuations += 1
            if continuations <= MAX_CONTINUATIONS:
                messages.append({"role": "assistant",
                                 "content": [c.model_dump(mode="json") for c in resp.content]})
                messages.append({"role": "user",
                                 "content": "Please continue from where you left off. Do not repeat what you already said."})
                continue

        continuations = 0

        if resp.stop_reason == "end_turn" and not tool_uses:
            emit_result(subtype="success", is_error=False, num_turns=turns,
                        duration_ms=int((time.time() - t0) * 1000),
                        duration_api_ms=total_api_ms,
                        result_text=combined_text.strip() or "(no text)", usage=total_usage)
            return 0

        if not tool_uses:
            emit_result(subtype="success", is_error=False, num_turns=turns,
                        duration_ms=int((time.time() - t0) * 1000),
                        duration_api_ms=total_api_ms,
                        result_text=combined_text.strip() or "(no tool calls; stopping)",
                        usage=total_usage)
            return 0

        result_blocks: list[dict[str, Any]] = []
        for tu in tool_uses:
            raw_in = tu.input if isinstance(tu.input, dict) else {}
            out = dispatch_tool(cwd, tu.name, raw_in)
            is_err = isinstance(out, str) and out.startswith("Error:")
            result_blocks.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": out, "is_error": is_err,
            })

        emit_user_tool_results(result_blocks)
        messages.append({"role": "assistant",
                         "content": [c.model_dump(mode="json") for c in resp.content]})
        messages.append({"role": "user", "content": result_blocks})

    emit_result(subtype="error_max_turns", is_error=True, num_turns=max_turns,
                duration_ms=int((time.time() - t0) * 1000), duration_api_ms=total_api_ms,
                errors=[f"Reached max turns ({max_turns})"], usage=total_usage)
    return 1


def _stream_one_turn(client: anthropic.Anthropic, kwargs: dict[str, Any],
                     *, verbose: bool = False, on_retry=None) -> Any | None:
    try:
        stream_ctx = _api_call_streaming(client, kwargs, on_retry=on_retry)
        with stream_ctx as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta" and verbose:
                        emit_stream_delta(delta_type="text", content=delta.text)
                    elif dtype == "thinking_delta" and verbose:
                        emit_stream_delta(delta_type="thinking", content=delta.thinking)
            return stream.get_final_message()
    except (anthropic.APIError, anthropic.APIConnectionError) as e:
        print(f"[stream error: {e}]", file=sys.stderr)
        return None
