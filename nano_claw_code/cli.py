"""CLI: Full interactive REPL + non-interactive print mode with all slash commands.

Ported from nano-claude-code TypeScript (main.tsx, screens/REPL.tsx, cli/print.ts).
Features: streaming, slash commands, session save/load, cost tracking, permissions,
rich markdown rendering, extended thinking, and stream-json harness output.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import readline
import sys
import time
from pathlib import Path
from typing import Any

from nano_claw_code.agent import (
    AgentState,
    CompactionNotice,
    PermissionRequest,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
    run_agent_loop,
    run_streaming,
)
from nano_claw_code.config import (
    HISTORY_FILE,
    MODELS,
    PERMISSION_MODES,
    calc_cost,
    ensure_dirs,
    load_config,
    load_dotenv,
    resolve_api_env,
    save_config,
)
from nano_claw_code.permissions import describe_permission, needs_permission
from nano_claw_code.prompts import build_system_prompt, resolve_model
from nano_claw_code.session import (
    auto_save_session,
    list_sessions,
    list_sessions_with_info,
    load_latest_session,
    load_session,
    rename_session,
    save_session,
    search_sessions,
)
from nano_claw_code.agents import clear_agent_cache, get_agents, tools_summary
from nano_claw_code.skills import (
    clear_skill_cache,
    format_skill_listing,
    get_skills,
    get_user_invocable_skills,
)
from nano_claw_code.tools_impl import anthropic_tool_defs, get_todos


# ── Optional rich rendering ───────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.markdown import Markdown

    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None


# ── ANSI helpers ──────────────────────────────────────────────────────────

_C = {
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "red": "\033[31m", "blue": "\033[34m", "magenta": "\033[35m",
    "bold": "\033[1m", "dim": "\033[2m", "reset": "\033[0m",
}


def _clr(text: str, *keys: str) -> str:
    return "".join(_C.get(k, "") for k in keys) + str(text) + _C["reset"]


def _info(msg: str) -> None:
    print(_clr(msg, "cyan"))


def _ok(msg: str) -> None:
    print(_clr(msg, "green"))


def _warn(msg: str) -> None:
    print(_clr(f"Warning: {msg}", "yellow"))


def _err(msg: str) -> None:
    print(_clr(f"Error: {msg}", "red"), file=sys.stderr)


# ── Tool description for display ─────────────────────────────────────────

def _tool_desc(name: str, inputs: dict) -> str:
    if name == "Read":
        return f"Read({inputs.get('file_path', '')})"
    if name == "Write":
        return f"Write({inputs.get('file_path', '')})"
    if name == "Edit":
        return f"Edit({inputs.get('file_path', '')})"
    if name == "Bash":
        return f"Bash({inputs.get('command', '')[:80]})"
    if name == "Glob":
        return f"Glob({inputs.get('pattern', '')})"
    if name == "Grep":
        return f"Grep({inputs.get('pattern', '')})"
    if name == "WebFetch":
        return f"WebFetch({inputs.get('url', '')[:60]})"
    if name == "WebSearch":
        return f"WebSearch({inputs.get('query', '')})"
    if name == "NotebookEdit":
        return f"NotebookEdit({inputs.get('target_notebook', '')}:{inputs.get('cell_idx', '')})"
    if name == "TodoWrite":
        todos = inputs.get("todos", [])
        return f"TodoWrite({len(todos)} items)"
    if name == "Agent":
        return f"Agent({inputs.get('description', inputs.get('prompt', '')[:40])})"
    return f"{name}(...)"


# ── Permission prompt ─────────────────────────────────────────────────────

def _ask_permission(desc: str, config: dict) -> bool:
    try:
        print()
        ans = input(_clr(f"  Allow: {desc} [y/N/a(ccept-all)] ", "yellow")).strip().lower()
        if ans == "a":
            config["permission_mode"] = "accept-all"
            _ok("  Permission mode set to accept-all for this session.")
            return True
        return ans in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        print()
        return False


# ── Markdown rendering ────────────────────────────────────────────────────

_accumulated_text: list[str] = []


def _stream_text(chunk: str) -> None:
    print(chunk, end="", flush=True)
    _accumulated_text.append(chunk)


def _flush_response() -> None:
    full = "".join(_accumulated_text)
    _accumulated_text.clear()
    if _RICH and full.strip() and any(c in full for c in ("#", "*", "`", "_", "[")):
        print()
        _console.print(Markdown(full))
        return
    print()


# ── Slash commands ────────────────────────────────────────────────────────
# Modeled after nano-claude-code TypeScript commands.ts
#
# Categories:
#   Session    : /clear /new /compact /save /load /export /history
#   Model      : /model /thinking /verbose
#   Navigation : /cwd /files /diff /status /context /cost /todos
#   System     : /permissions /config /init /doctor
#   Agent      : /btw /review
#   Misc       : /copy /bug /help /exit /quit

def _cmd_help(_args: str, _state: AgentState, _config: dict) -> bool:
    print(_clr("\n  Session", "bold"))
    print("    /clear, /new, /reset    Clear conversation history")
    print("    /compact [n]            Keep last n messages (default 4)")
    print("    /save [file]            Save session to file")
    print("    /load [query|file]      Load or search sessions by title")
    print("    /export [file]          Export conversation as markdown")
    print("    /history                Print conversation summary")
    print("    /rename <title>         Set display title for this session")
    print(_clr("\n  Model & Inference", "bold"))
    print("    /model [name]           Show or set model")
    print("    /thinking               Toggle extended thinking mode")
    print("    /verbose                Toggle verbose output")
    print(_clr("\n  Navigation & Context", "bold"))
    print("    /cwd [path]             Show or change working directory")
    print("    /files                  List files mentioned in conversation")
    print("    /diff                   Show uncommitted git changes")
    print("    /status                 Version, model, provider, env info")
    print("    /context                Show token usage estimate")
    print("    /cost                   Show API cost for this session")
    print("    /todos                  Show current task list")
    print(_clr("\n  System & Config", "bold"))
    print("    /permissions [mode]     Get/set: auto, accept-all, manual")
    print("    /config [key=value]     Show/set config")
    print("    /init                   Create CLAUDE.md in current directory")
    print("    /doctor                 Diagnose installation and config")
    print(_clr("\n  Agent & Skills", "bold"))
    print("    /btw <question>         Side question (runs in sub-agent)")
    print("    /review [pr]            Review PR or uncommitted changes")
    print("    /skills                 List available skills")
    print("    /agents                 List sub-agent profiles (Agent tool)")
    print(_clr("\n  Misc", "bold"))
    print("    /copy                   Copy last response to clipboard")
    print("    /bug [text]             Report a bug / feedback")
    print("    /help                   Show this help")
    print("    /exit, /quit            Exit")
    print()
    return True


def _cmd_clear(_args: str, state: AgentState, _config: dict) -> bool:
    state.messages.clear()
    state.turn_count = 0
    _ok("Conversation cleared.")
    return True


def _cmd_model(args: str, _state: AgentState, config: dict) -> bool:
    if not args.strip():
        _info(f"Current model: {config['model']}")
        _info("Available models:\n" + "\n".join(f"  {m}" for m in MODELS))
    else:
        config["model"] = args.strip()
        save_config(config)
        _ok(f"Model set to {config['model']}")
    return True


def _cmd_config(args: str, _state: AgentState, config: dict) -> bool:
    if not args.strip():
        display = {k: v for k, v in config.items() if k != "api_key"}
        print(json.dumps(display, indent=2))
    elif "=" in args:
        key, _, val = args.partition("=")
        key, val_str = key.strip(), val.strip()
        parsed_val: Any
        if val_str.lower() in ("true", "false"):
            parsed_val = val_str.lower() == "true"
        elif val_str.isdigit():
            parsed_val = int(val_str)
        else:
            parsed_val = val_str
        config[key] = parsed_val
        save_config(config)
        _ok(f"Set {key} = {parsed_val}")
    else:
        k = args.strip()
        v = config.get(k, "(not set)")
        _info(f"{k} = {v}")
    return True


def _cmd_save(args: str, state: AgentState, config: dict) -> bool:
    filename = args.strip() or None
    title = config.get("_session_title")
    try:
        path = save_session(
            state.messages, filename=filename,
            turn_count=state.turn_count,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            model=config.get("model", ""),
            title=title,
        )
        _ok(f"Session saved to {path}")
        if title:
            _info(f"  Title: {title}")
    except Exception as e:
        _err(str(e))
    return True


def _cmd_load(args: str, state: AgentState, config: dict) -> bool:
    query = args.strip()
    if not query:
        infos = list_sessions_with_info()
        if not infos:
            _info("No saved sessions found.")
            return True
        _info("Saved sessions:")
        for info in infos[:20]:
            title = info.get("title", "")
            fname = info.get("filename", "?")
            turns = info.get("turn_count", 0)
            msgs = info.get("message_count", 0)
            saved = info.get("saved_at", "")[:16]
            title_str = f"  {_clr(title, 'bold')}" if title else ""
            print(f"  {_clr(fname, 'cyan')}  {msgs}msg {turns}t  {_clr(saved, 'dim')}{title_str}")
        _info(f"\n  Use /load <filename> or /load <search query> to load a session.")
        return True

    if query.endswith(".json"):
        try:
            data = load_session(query)
            state.messages = data.get("messages", [])
            state.turn_count = data.get("turn_count", 0)
            state.total_input_tokens = data.get("total_input_tokens", 0)
            state.total_output_tokens = data.get("total_output_tokens", 0)
            config["_session_title"] = data.get("title", "")
            title = data.get("title", "")
            _ok(f"Session loaded ({len(state.messages)} messages, {state.turn_count} turns)")
            if title:
                _info(f"  Title: {title}")
            return True
        except Exception as e:
            _err(str(e))
            return True

    results = search_sessions(query)
    if not results:
        _info(f"No sessions matching '{query}'.")
        return True

    if len(results) == 1:
        info = results[0]
        try:
            data = load_session(info["filename"])
            state.messages = data.get("messages", [])
            state.turn_count = data.get("turn_count", 0)
            state.total_input_tokens = data.get("total_input_tokens", 0)
            state.total_output_tokens = data.get("total_output_tokens", 0)
            config["_session_title"] = data.get("title", "")
            _ok(f"Session loaded: {info.get('title', info['filename'])}")
            return True
        except Exception as e:
            _err(str(e))
            return True

    _info(f"Multiple sessions match '{query}':")
    for info in results[:15]:
        title = info.get("title", "")
        fname = info.get("filename", "?")
        print(f"  {_clr(fname, 'cyan')}  {_clr(title, 'bold')}")
    _info("  Use /load <filename> to load a specific session.")
    return True


def _cmd_export(args: str, state: AgentState, _config: dict) -> bool:
    """Export conversation as a markdown file."""
    if not state.messages:
        _info("Nothing to export (empty conversation).")
        return True
    filename = args.strip() or f"conversation_{int(__import__('time').time())}.md"
    lines: list[str] = ["# Nano Claw Code — Conversation Export\n"]
    for m in state.messages:
        role = m.get("role", "unknown").upper()
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        parts.append(b.get("text", ""))
                    elif b.get("type") == "tool_use":
                        parts.append(f"**[Tool: {b.get('name', '?')}]**\n```json\n{json.dumps(b.get('input', {}), indent=2)}\n```")
                    elif b.get("type") == "tool_result":
                        parts.append(f"**[Tool Result]**\n```\n{str(b.get('content', ''))[:500]}\n```")
            content = "\n\n".join(parts)
        lines.append(f"## {role}\n\n{content}\n\n---\n")
    try:
        Path(filename).write_text("\n".join(lines))
        _ok(f"Exported to {filename}")
    except Exception as e:
        _err(str(e))
    return True


def _cmd_history(_args: str, state: AgentState, _config: dict) -> bool:
    if not state.messages:
        _info("(empty conversation)")
        return True
    for i, m in enumerate(state.messages):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            print(f"[{i}] {_clr(role.upper(), 'bold')}: {content[:150]}")
        elif isinstance(content, list):
            types = []
            for b in content:
                if isinstance(b, dict):
                    types.append(b.get("type", "?"))
                else:
                    types.append(getattr(b, "type", "?"))
            print(f"[{i}] {_clr(role.upper(), 'bold')}: [{', '.join(types)}]")
    return True


def _cmd_context(_args: str, state: AgentState, config: dict) -> bool:
    msg_chars = sum(len(str(m.get("content", ""))) for m in state.messages)
    est_tokens = msg_chars // 4
    _info(f"Messages: {len(state.messages)}")
    _info(f"Estimated context: ~{est_tokens:,} tokens")
    _info(f"Model: {config['model']}")
    _info(f"Turns: {state.turn_count}")
    return True


def _cmd_cost(_args: str, state: AgentState, config: dict) -> bool:
    cost = calc_cost(config["model"], state.total_input_tokens, state.total_output_tokens)
    _info(f"Input tokens:  {state.total_input_tokens:,}")
    _info(f"Output tokens: {state.total_output_tokens:,}")
    _info(f"Turns: {state.turn_count}")
    _info(f"Est. cost: ${cost:.4f} USD")
    return True


def _cmd_verbose(_args: str, _state: AgentState, config: dict) -> bool:
    config["verbose"] = not config.get("verbose", False)
    _ok(f"Verbose mode: {'ON' if config['verbose'] else 'OFF'}")
    return True


def _cmd_thinking(_args: str, _state: AgentState, config: dict) -> bool:
    config["thinking"] = not config.get("thinking", False)
    _ok(f"Extended thinking: {'ON' if config['thinking'] else 'OFF'}")
    return True


def _cmd_permissions(args: str, _state: AgentState, config: dict) -> bool:
    if not args.strip():
        _info(f"Permission mode: {config.get('permission_mode', 'auto')}")
        _info(f"Available: {', '.join(PERMISSION_MODES)}")
    else:
        mode = args.strip()
        if mode not in PERMISSION_MODES:
            _err(f"Unknown mode: {mode}. Choose: {', '.join(PERMISSION_MODES)}")
        else:
            config["permission_mode"] = mode
            save_config(config)
            _ok(f"Permission mode: {mode}")
    return True


def _cmd_cwd(args: str, _state: AgentState, _config: dict) -> bool:
    if not args.strip():
        _info(f"CWD: {os.getcwd()}")
    else:
        try:
            os.chdir(args.strip())
            clear_skill_cache()
            clear_agent_cache()
            _ok(f"Changed to: {os.getcwd()}")
        except Exception as e:
            _err(str(e))
    return True


def _cmd_todos(_args: str, _state: AgentState, _config: dict) -> bool:
    todos = get_todos()
    if not todos:
        _info("No active todos.")
        return True
    for t in todos:
        status = t.get("status", "pending")
        marker = {"pending": "○", "in_progress": "●", "completed": "✓", "cancelled": "✗"}.get(status, "?")
        print(f"  {marker} [{status}] {t.get('id', '?')}: {t.get('content', '')}")
    return True


def _cmd_compact(args: str, state: AgentState, _config: dict) -> bool:
    keep = 4
    if args.strip().isdigit():
        keep = max(2, int(args.strip()))
    if len(state.messages) <= keep:
        _info("Conversation is already short, nothing to compact.")
        return True
    old_len = len(state.messages)
    state.messages = state.messages[-keep:]
    _ok(f"Compacted: {old_len} → {len(state.messages)} messages (kept last {keep})")
    return True


def _cmd_diff(_args: str, _state: AgentState, _config: dict) -> bool:
    """Show uncommitted git changes."""
    import subprocess
    try:
        staged = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            capture_output=True, text=True, timeout=10,
        )
        unstaged = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, timeout=10,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=10,
        )
        has_output = False
        if staged.stdout.strip():
            print(_clr("\n  Staged changes:", "green", "bold"))
            print(staged.stdout)
            has_output = True
        if unstaged.stdout.strip():
            print(_clr("\n  Unstaged changes:", "yellow", "bold"))
            print(unstaged.stdout)
            has_output = True
        if untracked.stdout.strip():
            print(_clr("\n  Untracked files:", "dim"))
            for f in untracked.stdout.strip().splitlines()[:30]:
                print(f"    {f}")
            has_output = True
        if not has_output:
            _info("Working tree clean — no uncommitted changes.")
    except FileNotFoundError:
        _err("git not found.")
    except subprocess.TimeoutExpired:
        _err("git command timed out.")
    return True


def _cmd_status(_args: str, _state: AgentState, config: dict) -> bool:
    """Show version, model, provider, and environment info."""
    from nano_claw_code import __version__
    api_env = resolve_api_env(config.get("api_key"))
    provider = api_env.get("provider", "unknown")

    print(_clr("\n  Nano Claw Code Status", "bold"))
    print(f"    Version:      v{__version__}")
    print(f"    Python:       {sys.version.split()[0]}")
    print(f"    Provider:     {provider}")
    print(f"    Base URL:     {api_env.get('base_url', '(default)')}")
    print(f"    Model:        {config['model']}")
    print(f"    Permissions:  {config.get('permission_mode', 'auto')}")
    print(f"    Thinking:     {'ON' if config.get('thinking') else 'OFF'}")
    print(f"    CWD:          {os.getcwd()}")

    import subprocess
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        print(f"    Git branch:   {branch}")
    except Exception:
        print("    Git:          (not a repo)")

    dotenv_files = _find_dotenv_files_for_display()
    if dotenv_files:
        print(f"    .env loaded:  {', '.join(dotenv_files)}")
    print()
    return True


def _find_dotenv_files_for_display() -> list[str]:
    from nano_claw_code.config import _find_dotenv_files
    return [str(p) for p in _find_dotenv_files()]


def _cmd_files(_args: str, state: AgentState, _config: dict) -> bool:
    """List files mentioned in the conversation (from tool calls)."""
    files: dict[str, str] = {}
    for m in state.messages:
        content = m.get("content", "")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                inp = b.get("input", {})
                name = b.get("name", "")
                fp = inp.get("file_path") or inp.get("path") or inp.get("target_notebook") or ""
                if fp:
                    files[fp] = name
                pattern = inp.get("pattern") or inp.get("glob") or ""
                if pattern and name in ("Glob", "Grep"):
                    files[pattern] = name
    if not files:
        _info("No files referenced in conversation.")
        return True
    _info(f"Files referenced ({len(files)}):")
    for fp, tool in sorted(files.items()):
        print(f"  {_clr(tool, 'dim'):20s}  {fp}")
    return True


def _cmd_init(_args: str, _state: AgentState, _config: dict) -> bool:
    """Create a CLAUDE.md file in the current directory."""
    target = Path.cwd() / "CLAUDE.md"
    if target.exists():
        _warn(f"CLAUDE.md already exists at {target}")
        try:
            ans = input(_clr("  Overwrite? [y/N] ", "yellow")).strip().lower()
            if ans not in ("y", "yes"):
                _info("Cancelled.")
                return True
        except (KeyboardInterrupt, EOFError):
            print()
            return True

    template = """\
# CLAUDE.md

## Project Overview
<!-- Describe the project in a few sentences -->

## Tech Stack
<!-- List main technologies, frameworks, versions -->

## Key Commands
```bash
# Build
# Test
# Lint
# Run
```

## Code Conventions
<!-- Describe naming conventions, patterns, etc. -->

## Important Notes
<!-- Anything the agent should know about this project -->
"""
    try:
        target.write_text(template)
        _ok(f"Created {target}")
        _info("Edit this file to give the agent context about your project.")
    except Exception as e:
        _err(str(e))
    return True


def _cmd_doctor(_args: str, _state: AgentState, config: dict) -> bool:
    """Diagnose installation, environment, and API connectivity."""
    import subprocess
    from nano_claw_code import __version__

    print(_clr("\n  Doctor — Diagnostics", "bold"))

    print(f"\n  {_clr('Package', 'bold')}")
    print(f"    nano-claw-code v{__version__}")
    print(f"    Python {sys.version.split()[0]}")
    print(f"    Platform: {sys.platform}")

    print(f"\n  {_clr('Dependencies', 'bold')}")
    for mod_name in ["anthropic", "openai", "httpx", "rich"]:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "?")
            print(f"    {mod_name}: {_clr(ver, 'green')}")
        except ImportError:
            label = "(optional)" if mod_name == "rich" else "(MISSING!)"
            print(f"    {mod_name}: {_clr(label, 'red' if 'MISS' in label else 'yellow')}")

    print(f"\n  {_clr('API Configuration', 'bold')}")
    api_env = resolve_api_env(config.get("api_key"))
    provider = api_env.get("provider", "none")
    has_key = bool(api_env.get("api_key"))
    print(f"    Provider: {provider}")
    print(f"    API key:  {'set' if has_key else _clr('NOT SET', 'red')}")
    print(f"    Base URL: {api_env.get('base_url', '(default)')}")

    from nano_claw_code.config import _find_dotenv_files
    dotenv_files = _find_dotenv_files()
    if dotenv_files:
        print(f"    .env files: {', '.join(str(p) for p in dotenv_files)}")
    else:
        print(f"    .env files: {_clr('none found', 'yellow')}")

    print(f"\n  {_clr('Tools', 'bold')}")
    for tool in ["git", "rg", "gh"]:
        try:
            out = subprocess.run(
                [tool, "--version"], capture_output=True, text=True, timeout=5,
            )
            ver = out.stdout.strip().split("\n")[0][:60]
            print(f"    {tool}: {_clr(ver, 'green')}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"    {tool}: {_clr('not found', 'yellow')}")

    if has_key:
        print(f"\n  {_clr('API Connectivity', 'bold')}")
        try:
            if provider == "openai_compat":
                from openai import OpenAI

                oai = OpenAI(api_key=api_env["api_key"], base_url=api_env["base_url"])
                resp = oai.chat.completions.create(
                    model=config["model"],
                    max_tokens=16,
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                )
                snippet = (resp.choices[0].message.content or "").strip()[:80]
                print(f"    {_clr('API connection OK', 'green')} (model: {config['model']})")
                if snippet:
                    print(f"    {_clr('Sample reply:', 'dim')} {snippet!r}")
            else:
                import anthropic as anth

                test_env = {k: v for k, v in api_env.items() if k != "provider"}
                client = anth.Anthropic(**test_env)
                client.messages.create(
                    model=config["model"],
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Say OK"}],
                )
                print(f"    {_clr('API connection OK', 'green')} (model: {config['model']})")
        except Exception as e:
            print(f"    {_clr(f'API error: {e}', 'red')}")

    print()
    return True


def _cmd_btw(args: str, state: AgentState, config: dict) -> bool:
    """Side question — runs in a sub-agent without derailing main conversation."""
    question = args.strip()
    if not question:
        _err("Usage: /btw <your question>")
        return True

    _info(f"[btw] Asking side question...")
    import anthropic as anth
    api_env = resolve_api_env(config.get("api_key"))
    if not api_env.get("api_key"):
        _err("No API key configured.")
        return True
    api_env.pop("provider", None)
    client = anth.Anthropic(**api_env)
    model = config["model"]

    from nano_claw_code.prompts import build_system_prompt
    system = build_system_prompt(cwd=str(Path.cwd()), bare=True)
    tools = anthropic_tool_defs()

    sub_msgs: list[dict[str, Any]] = [{"role": "user", "content": question}]
    max_turns = 5

    print(_clr("\n╭─ /btw ───────────────────────────────", "dim"))
    for _turn in range(max_turns):
        try:
            resp = client.messages.create(
                model=model, max_tokens=config.get("max_tokens", 16384),
                system=system, tools=tools, messages=sub_msgs,
            )
        except Exception as e:
            _err(f"[btw] API error: {e}")
            break

        text_parts = []
        tool_uses = []
        for block in resp.content:
            if getattr(block, "type", "") == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", "") == "tool_use":
                tool_uses.append(block)

        if text_parts:
            text = "\n".join(text_parts)
            print(text)

        if resp.stop_reason != "tool_use" or not tool_uses:
            break

        sub_msgs.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for tu in tool_uses:
            from nano_claw_code.tools_impl import dispatch_tool
            raw_in = tu.input if isinstance(tu.input, dict) else json.loads(tu.input)
            print(_clr(f"  [{_tool_desc(tu.name, raw_in)}]", "dim"))
            result = dispatch_tool(Path.cwd(), tu.name, raw_in)
            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": result[:4000] if isinstance(result, str) else str(result)[:4000],
            })
        sub_msgs.append({"role": "user", "content": tool_results})

    print(_clr("╰───────────────────────────────────────", "dim"))
    print()
    return True


def _cmd_review(args: str, _state: AgentState, _config: dict) -> str:
    """Review a PR or uncommitted changes — returns prompt for the agent."""
    pr_ref = args.strip()
    if pr_ref:
        return f"Review the pull request {pr_ref}. Use `gh pr view {pr_ref}` and `gh pr diff {pr_ref}` to inspect it. Give a thorough code review with specific feedback."
    return "Review my uncommitted changes. Use `git diff` and `git diff --cached` to see the changes. Give a thorough code review with specific feedback on code quality, potential bugs, and suggestions."


def _cmd_copy(_args: str, state: AgentState, _config: dict) -> bool:
    """Copy the last assistant response to clipboard."""
    import subprocess
    last_text = ""
    for m in reversed(state.messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            last_text = content
            break
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            last_text = "\n".join(parts)
            break
    if not last_text:
        _info("No assistant response to copy.")
        return True

    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=last_text, text=True, check=True)
        elif sys.platform == "linux":
            for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
                try:
                    subprocess.run(cmd, input=last_text, text=True, check=True)
                    break
                except FileNotFoundError:
                    continue
            else:
                _err("Install xclip or xsel for clipboard support on Linux.")
                return True
        else:
            _err(f"Clipboard not supported on {sys.platform}")
            return True
        _ok(f"Copied {len(last_text)} chars to clipboard.")
    except Exception as e:
        _err(f"Clipboard error: {e}")
    return True


def _cmd_rename(args: str, state: AgentState, config: dict) -> bool:
    """Set a display title for the current session."""
    new_title = args.strip()
    if not new_title:
        current = config.get("_session_title", "")
        if current:
            _info(f"Current title: {current}")
        else:
            _info("No title set. Usage: /rename <title>")
        return True
    config["_session_title"] = new_title
    _ok(f"Session title set: {new_title}")
    return True


def _cmd_skills(_args: str, _state: AgentState, _config: dict) -> bool:
    """List available skills from .claude/skills/ directories."""
    clear_skill_cache()
    skills = get_skills()
    if not skills:
        _info("No skills found.")
        _info("  Create skills in .claude/skills/<name>/SKILL.md")
        _info("  or in ~/.claude/skills/<name>/SKILL.md")
        return True
    _info(f"Available skills ({len(skills)}):\n")
    for name, s in sorted(skills.items()):
        desc = s.get("description", "")
        ctx = s.get("context", "inline")
        source = Path(s.get("source_dir", "")).name
        invocable = s.get("user_invocable", True)
        marker = "⚡" if ctx == "fork" else "📝"
        line = f"  {marker} /{name}"
        if desc:
            line += f"  — {desc}"
        if not invocable:
            line += _clr("  (model-only)", "dim")
        print(line)
    print()
    _info("  Skills are loaded from .claude/skills/*/SKILL.md directories.")
    _info("  Use /skill-name or the Skill tool to execute them.")
    return True


def _cmd_agents(_args: str, _state: AgentState, _config: dict) -> bool:
    """List Agent tool profiles (built-in + .claude/agents/*.md)."""
    clear_agent_cache()
    agents = get_agents()
    _info(f"Sub-agents ({len(agents)}):\n")

    for _k, ag in sorted(agents.items(), key=lambda x: x[1].agent_type.lower()):
        line = f"  • {ag.agent_type}  — {tools_summary(ag)}"
        if ag.source != "built-in":
            line += _clr(f"  [{ag.source}]", "dim")
        print(line)
        print(_clr(f"    {ag.when_to_use[:200]}{'…' if len(ag.when_to_use) > 200 else ''}", "dim"))
    print()
    _info("  Add markdown agents under .claude/agents/*.md or ~/.claude/agents/.")
    _info("  Use the Agent tool with subagent_type set to the profile name.")
    return True


def _cmd_bug(args: str, _state: AgentState, _config: dict) -> bool:
    """Open a bug report / feedback."""
    _info("Report bugs at: https://github.com/anthropics/claude-code/issues")
    if args.strip():
        _info(f"Your feedback: {args.strip()}")
        _info("(Noted — please also file on GitHub for tracking)")
    return True


def _cmd_exit(_args: str, _state: AgentState, _config: dict) -> bool:
    _ok("Goodbye!")
    sys.exit(0)


# ── Command registry ─────────────────────────────────────────────────────

_COMMANDS: dict[str, Any] = {
    # Session
    "help": _cmd_help,
    "clear": _cmd_clear, "new": _cmd_clear, "reset": _cmd_clear,
    "compact": _cmd_compact,
    "save": _cmd_save, "load": _cmd_load,
    "export": _cmd_export,
    "history": _cmd_history,
    # Model
    "model": _cmd_model,
    "thinking": _cmd_thinking,
    "verbose": _cmd_verbose,
    # Navigation
    "cwd": _cmd_cwd,
    "files": _cmd_files,
    "diff": _cmd_diff,
    "status": _cmd_status,
    "context": _cmd_context,
    "cost": _cmd_cost,
    "todos": _cmd_todos,
    # System
    "permissions": _cmd_permissions, "allowed-tools": _cmd_permissions,
    "config": _cmd_config, "settings": _cmd_config,
    "init": _cmd_init,
    "doctor": _cmd_doctor,
    # Agent
    "btw": _cmd_btw,
    "review": _cmd_review,
    # Skills & sub-agents
    "skills": _cmd_skills,
    "agents": _cmd_agents,
    # Session title
    "rename": _cmd_rename,
    # Misc
    "copy": _cmd_copy,
    "bug": _cmd_bug, "feedback": _cmd_bug,
    "exit": _cmd_exit, "quit": _cmd_exit,
}


def _handle_slash(line: str, state: AgentState, config: dict) -> bool | str:
    """Handle slash commands. Returns True if handled, False if not a command,
    or a string prompt to inject into the conversation."""
    if not line.startswith("/"):
        return False
    parts = line[1:].split(None, 1)
    if not parts:
        return False
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    handler = _COMMANDS.get(cmd)
    if handler:
        return handler(args, state, config)

    # Check if it matches a user-invocable skill name
    skills = get_skills()
    skill = skills.get(cmd)
    if skill and skill.get("user_invocable", True):
        from nano_claw_code.skills import expand_skill_prompt
        prompt = expand_skill_prompt(skill, args)
        _info(f"[skill: /{cmd}]")
        return prompt

    _err(f"Unknown command: /{cmd}. Type /help for commands or /skills to list skills.")
    return True


# ── Slash command metadata (for dropdown display) ────────────────────────

_CMD_META: dict[str, tuple[str, str]] = {
    "help":        ("Misc",       "Show all available commands"),
    "clear":       ("Session",    "Clear conversation history"),
    "new":         ("Session",    "Start a new conversation"),
    "reset":       ("Session",    "Reset conversation (alias for /clear)"),
    "compact":     ("Session",    "Keep last N messages (default 4)"),
    "save":        ("Session",    "Save session to file"),
    "load":        ("Session",    "Load a saved session"),
    "export":      ("Session",    "Export conversation as markdown"),
    "history":     ("Session",    "Print conversation message summary"),
    "model":       ("Model",      "Show or change the model"),
    "thinking":    ("Model",      "Toggle extended thinking mode"),
    "verbose":     ("Model",      "Toggle verbose output"),
    "cwd":         ("Navigate",   "Show or change working directory"),
    "files":       ("Navigate",   "List files mentioned in conversation"),
    "diff":        ("Navigate",   "Show uncommitted git changes"),
    "status":      ("System",     "Version, model, provider info"),
    "context":     ("Navigate",   "Show token usage estimate"),
    "cost":        ("Navigate",   "Show API cost for this session"),
    "todos":       ("Navigate",   "Show current task list"),
    "permissions": ("System",     "Get/set permission mode"),
    "allowed-tools": ("System",   "Alias for /permissions"),
    "config":      ("System",     "Show or set config values"),
    "settings":    ("System",     "Alias for /config"),
    "init":        ("System",     "Create CLAUDE.md template"),
    "doctor":      ("System",     "Diagnose installation and API"),
    "btw":         ("Agent",      "Side question via sub-agent"),
    "review":      ("Agent",      "Review PR or uncommitted changes"),
    "skills":      ("Agent",      "List available skills"),
    "agents":      ("Agent",      "List sub-agent profiles"),
    "rename":      ("Session",    "Set display title for current session"),
    "copy":        ("Misc",       "Copy last response to clipboard"),
    "bug":         ("Misc",       "Report a bug / feedback"),
    "feedback":    ("Misc",       "Alias for /bug"),
    "exit":        ("Misc",       "Exit nano-claw-code"),
    "quit":        ("Misc",       "Exit nano-claw-code"),
}

_ARG_COMPLETIONS: dict[str, list[str]] = {
    "/model": MODELS,
    "/permissions": list(PERMISSION_MODES),
    "/allowed-tools": list(PERMISSION_MODES),
    "/config": ["model=", "permission_mode=", "verbose=", "thinking=", "thinking_budget=", "max_tokens="],
    "/settings": ["model=", "permission_mode=", "verbose=", "thinking=", "thinking_budget=", "max_tokens="],
}


# ── prompt_toolkit-based input with dropdown ──────────────────────────────

def _build_prompt_session():
    """Build a prompt_toolkit PromptSession with slash command dropdown."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style

    ensure_dirs()

    style = Style.from_dict({
        "completion-menu":              "bg:#1a1a2e #e0e0e0",
        "completion-menu.completion":   "bg:#1a1a2e #c0c0c0",
        "completion-menu.completion.current": "bg:#16213e #00d4ff bold",
        "completion-menu.meta":         "bg:#1a1a2e #888888",
        "completion-menu.meta.current": "bg:#16213e #00aacc",
        "scrollbar.background":         "bg:#333333",
        "scrollbar.button":             "bg:#555555",
        "prompt":                       "#888888",
        "prompt.cwd":                   "#666666",
        "prompt.arrow":                 "#00d4ff bold",
    })

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor.lstrip()

            if " " in text and text.startswith("/"):
                cmd = text.split()[0]
                partial = text.split(None, 1)[1] if len(text.split(None, 1)) > 1 else ""
                candidates = _ARG_COMPLETIONS.get(cmd, [])
                for c in candidates:
                    if c.startswith(partial):
                        yield Completion(c, start_position=-len(partial))
                return

            if not text.startswith("/"):
                return

            prefix = text
            seen = set()
            for cmd_name, (category, description) in _CMD_META.items():
                full = f"/{cmd_name}"
                if full in seen:
                    continue
                if full.startswith(prefix):
                    seen.add(full)
                    yield Completion(
                        full,
                        start_position=-len(prefix),
                        display=HTML(f"<b>{full}</b>"),
                        display_meta=HTML(f"<style fg='#888888'>{category}</style>  {description}"),
                    )

    session = PromptSession(
        completer=SlashCompleter(),
        history=FileHistory(str(HISTORY_FILE)),
        style=style,
        complete_while_typing=True,
        complete_in_thread=True,
    )
    return session


def _setup_readline() -> None:
    """Fallback readline setup for when prompt_toolkit is unavailable."""
    ensure_dirs()
    try:
        readline.read_history_file(str(HISTORY_FILE))
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, str(HISTORY_FILE))

    all_cmds = sorted({f"/{c}" for c in _COMMANDS})

    def completer(text: str, idx: int) -> str | None:
        buf = readline.get_line_buffer().lstrip()
        if " " in buf:
            cmd = buf.split()[0]
            arg_prefix = buf.split(None, 1)[1] if len(buf.split(None, 1)) > 1 else ""
            candidates = _ARG_COMPLETIONS.get(cmd, [])
            matches = [a for a in candidates if a.startswith(arg_prefix)]
        else:
            matches = [c for c in all_cmds if c.startswith(text)]
        return matches[idx] if idx < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t")
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


# ── Interactive REPL ──────────────────────────────────────────────────────

def _print_banner(provider: str, provider_label: str, model: str, perm_mode: str,
                  tools_count: int, resume_state: dict | None = None,
                  state: AgentState | None = None,
                  session_title: str = "", skill_count: int = 0) -> None:
    """Print the startup banner with claw mascot."""
    c1 = "\033[38;5;208m"  # warm orange (claw body)
    c2 = "\033[38;5;172m"  # darker orange (claw shadow)
    c3 = "\033[38;5;214m"  # gold highlight (claw tip)
    g1 = "\033[38;5;40m"   # bright green (title text)
    mg = "\033[38;5;176m"  # muted mauve
    wh = "\033[38;5;252m"  # warm white
    b  = "\033[1m"
    r  = "\033[0m"
    d  = "\033[2m"
    y  = "\033[38;5;222m"  # soft gold
    gn = "\033[38;5;114m"  # sage green
    cy = "\033[38;5;116m"  # muted teal
    bx = "\033[38;5;240m"  # box border

    tools_plain = f"{tools_count} built-in"
    tools_label = f"{gn}{tools_count}{r} built-in"
    if skill_count > 0:
        tools_plain += f" + {skill_count} skills"
        tools_label += f" + {mg}{skill_count}{r} skills"

    cwd_str = str(Path.cwd())
    g = " " * 17

    rows: list[tuple[str, str]] = []

    rows.append(("", ""))
    rows.append((
        "     ▄████▄      nano claw code",
        f"     {c2}▄{c1}████{c2}▄{r}      {g1}{b}nano claw code{r}",
    ))
    rows.append((
        "    ██▀  ▀██     ✦ Nano AI Coding Agent ✦",
        f"    {c1}██{c3}▀  ▀{c1}██{r}     {g1}✦ Nano AI Coding Agent ✦{r}",
    ))
    rows.append((
        f"    ██▄  ▄██     Provider     {provider}",
        f"    {c1}██{c3}▄  ▄{c1}██{r}     {wh}Provider{r}     {provider_label}",
    ))
    rows.append((
        f"     ▀████▀      Model        {model}",
        f"     {c2}▀{c1}████{c2}▀{r}      {wh}Model{r}        {cy}{b}{model}{r}",
    ))
    rows.append((
        f"{g}Tools        {tools_plain}",
        f"{g}{wh}Tools{r}        {tools_label}",
    ))
    rows.append((
        f"{g}CWD          {cwd_str}",
        f"{g}{wh}CWD{r}          {d}{cwd_str}{r}",
    ))
    if session_title:
        rows.append((
            f"{g}Session      {session_title}",
            f"{g}{wh}Session{r}      {b}{session_title}{r}",
        ))
    if resume_state and state:
        n_msgs = len(state.messages)
        res_txt = f"{n_msgs} messages, {state.turn_count} turns"
        rows.append((
            f"{g}Resumed      {res_txt}",
            f"{g}{wh}Resumed{r}      {y}{res_txt}{r}",
        ))
    rows.append(("", ""))
    rows.append((
        "  Type / to see commands  ·  Tab to select",
        f"  Type {b}/{r} to see commands  ·  {b}Tab{r} to select",
    ))
    rows.append((
        "  /help for all commands   ·  Ctrl+C to cancel",
        f"  {b}/help{r} for all commands   ·  {b}Ctrl+C{r} to cancel",
    ))

    inner_w = max(len(plain) for plain, _ in rows) + 2
    if inner_w < 50:
        inner_w = 50

    bar = "─" * inner_w

    print()
    print(f"  {bx}╭{bar}╮{r}")
    for plain, colored in rows:
        padding = " " * (inner_w - len(plain))
        print(f"  {bx}│{r}{colored}{padding}{bx}│{r}")
    print(f"  {bx}╰{bar}╯{r}")
    print()


def run_repl(config: dict, resume_state: dict | None = None) -> int:
    import anthropic as anth

    api_env = resolve_api_env(config.get("api_key"))
    if not api_env.get("api_key"):
        _err("No API key found. Create a .env file in the project directory:")
        _err("")
        _err("  # OpenAI-compatible (Azure AI / Kimi / MiniMax, etc.)")
        _err("  OPENAI_COMPAT_BASE_URL=https://....../openai/v1/")
        _err("  OPENAI_COMPAT_API_KEY=...")
        _err("  OPENAI_COMPAT_MODEL=Kimi-K2.5")
        _err("")
        _err("  # Direct Anthropic")
        _err('  ANTHROPIC_API_KEY=sk-ant-xxx')
        _err("")
        _err("  # OR OpenRouter")
        _err('  OPENROUTER_API_KEY=sk-or-xxx')
        _err("")
        _err("Or export the variable in your shell.")
        return 1

    provider = api_env.get("provider", "anthropic")
    openai_client = None
    if provider == "openai_compat":
        from openai import OpenAI

        openai_client = OpenAI(
            api_key=api_env["api_key"],
            base_url=api_env["base_url"],
        )
    else:
        api_env.pop("provider", None)
        client = anth.Anthropic(**api_env)

    if resume_state:
        state = AgentState(
            messages=resume_state.get("messages", []),
            total_input_tokens=resume_state.get("total_input_tokens", 0),
            total_output_tokens=resume_state.get("total_output_tokens", 0),
            turn_count=resume_state.get("turn_count", 0),
        )
    else:
        state = AgentState()
    tools = anthropic_tool_defs()

    # Try prompt_toolkit for rich dropdown, fallback to readline
    pt_session = None
    try:
        pt_session = _build_prompt_session()
    except Exception:
        _setup_readline()

    model = config["model"]
    perm_mode = config.get("permission_mode", "auto")

    # Load session title from resume data or CLI --name
    session_title = config.get("_session_title", "")
    if not session_title and resume_state:
        session_title = resume_state.get("title", "")
        if session_title:
            config["_session_title"] = session_title

    # Count available skills
    try:
        skill_count = len(get_skills())
    except Exception:
        skill_count = 0

    provider_label = {
        "anthropic": _clr("Anthropic", "green"),
        "openrouter": _clr("OpenRouter", "magenta"),
        "openai_compat": _clr("OpenAI-compat", "cyan"),
        "proxy": _clr(api_env.get("base_url", "proxy"), "yellow"),
    }.get(provider, _clr(provider, "yellow"))

    _print_banner(provider, provider_label, model, perm_mode, len(tools),
                  resume_state=resume_state, state=state,
                  session_title=session_title, skill_count=skill_count)

    last_ctrl_c = 0.0
    while True:
        try:
            cwd_short = Path.cwd().name
            if pt_session:
                from prompt_toolkit.formatted_text import HTML
                prompt_html = HTML(f"\n<style fg='#666666'>[{cwd_short}]</style> <style fg='#00d4ff'><b>❯ </b></style>")
                user_input = pt_session.prompt(prompt_html).strip()
            else:
                prompt = _clr(f"\n[{cwd_short}] ", "dim") + _clr("❯ ", "cyan", "bold")
                user_input = input(prompt).strip()
            last_ctrl_c = 0.0
        except EOFError:
            print(_clr("\nGoodbye!", "green"))
            return 0
        except KeyboardInterrupt:
            now = time.time()
            if now - last_ctrl_c < 1.0:
                print(_clr("\nGoodbye!", "green"))
                return 0
            last_ctrl_c = now
            print(_clr("\n  (press Ctrl+C again to exit, or type /quit)", "dim"))
            continue

        if not user_input:
            continue
        slash_result = _handle_slash(user_input, state, config)
        if slash_result is True:
            continue
        if isinstance(slash_result, str):
            user_input = slash_result

        model = config.get("model", model)
        verbose = config.get("verbose", False)
        thinking = config.get("thinking", False)
        perm_mode = config.get("permission_mode", "auto")
        thinking_budget = config.get("thinking_budget", 10_000)

        system_prompt = build_system_prompt(cwd=str(Path.cwd()), bare=config.get("bare", False))

        print(_clr("\n╭─ Assistant ", "dim") + _clr("●", "green") + _clr(" ───────────────────────", "dim"))

        thinking_started = False
        msg_count_before = len(state.messages)
        interrupted = False
        try:
            if openai_client is not None:
                from nano_claw_code.openai_compat import run_streaming_openai

                event_iter = run_streaming_openai(
                    user_input, state,
                    client=openai_client, model=model, system_prompt=system_prompt,
                    tools=tools, cwd=Path.cwd(),
                    thinking=thinking, thinking_budget=thinking_budget,
                    permission_mode=perm_mode,
                )
            else:
                event_iter = run_streaming(
                    user_input, state,
                    client=client, model=model, system_prompt=system_prompt,
                    tools=tools, cwd=Path.cwd(),
                    thinking=thinking, thinking_budget=thinking_budget,
                    permission_mode=perm_mode,
                )
            for event in event_iter:
                if isinstance(event, CompactionNotice):
                    print(_clr(
                        f"\n  [context compacted: {event.old_count} → {event.new_count} messages]",
                        "yellow",
                    ))
                elif isinstance(event, TextChunk):
                    _stream_text(event.text)
                elif isinstance(event, ThinkingChunk):
                    if verbose:
                        if not thinking_started:
                            print(_clr("\n  [thinking]", "dim"))
                            thinking_started = True
                        print(_clr(event.text, "dim"), end="", flush=True)
                elif isinstance(event, ToolStart):
                    _flush_response()
                    desc = _tool_desc(event.name, event.inputs)
                    print(_clr(f"\n  ⚙ {desc}", "dim", "cyan"), flush=True)
                    if verbose:
                        preview = json.dumps(event.inputs, ensure_ascii=False)[:200]
                        print(_clr(f"    inputs: {preview}", "dim"))
                elif isinstance(event, PermissionRequest):
                    event.granted = _ask_permission(event.description, config)
                    perm_mode = config.get("permission_mode", perm_mode)
                elif isinstance(event, ToolEnd):
                    if not event.permitted:
                        print(_clr(f"  ✗ Denied", "dim", "red"), flush=True)
                    elif isinstance(event.result, str) and event.result.startswith("Error:"):
                        print(_clr(f"  ✗ {event.result[:120]}", "dim", "red"), flush=True)
                    else:
                        result_str = event.result if isinstance(event.result, str) else str(event.result)
                        lines = result_str.count("\n") + 1
                        size = len(result_str)
                        print(_clr(f"  ✓ {event.name} → {lines} lines ({size} chars)", "dim", "green"), flush=True)
                    if verbose and event.permitted and isinstance(event.result, str) and not event.result.startswith("Error:"):
                        preview = event.result[:500].replace("\n", "\n    ")
                        print(_clr(f"    {preview}", "dim"))
                elif isinstance(event, TurnDone):
                    if verbose:
                        cost = calc_cost(model, event.input_tokens, event.output_tokens)
                        cache_info = ""
                        if event.cache_read_tokens > 0 or event.cache_creation_tokens > 0:
                            cache_info = (
                                f" | cache: {event.cache_read_tokens} read"
                                f" / {event.cache_creation_tokens} created"
                            )
                        print(_clr(
                            f"\n  [tokens: +{event.input_tokens} in / +{event.output_tokens} out"
                            f"{cache_info} | ~${cost:.4f}]",
                            "dim",
                        ))
        except KeyboardInterrupt:
            interrupted = True
            print(_clr("\n  (interrupted — current turn cancelled)", "yellow"))
            if len(state.messages) > msg_count_before:
                state.messages = state.messages[:msg_count_before]

        _flush_response()
        print(_clr("╰──────────────────────────────────────────────", "dim"))
        print()

        # Auto-generate session title on first real interaction
        if not config.get("_session_title") and state.messages:
            from nano_claw_code.session import generate_session_title
            title = generate_session_title(state.messages)
            if title and title != "(untitled session)":
                config["_session_title"] = title

        auto_save_session(
            state.messages,
            turn_count=state.turn_count,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            model=model,
            title=config.get("_session_title"),
        )


# ── Print mode (non-interactive text) ─────────────────────────────────────

def _run_print_text(prompt: str, config: dict, max_turns: int = 50) -> int:
    import anthropic as anth

    api_env = resolve_api_env(config.get("api_key"))
    if not api_env.get("api_key"):
        _err("No API key found. Create a .env file or export ANTHROPIC_API_KEY.")
        return 1

    provider = api_env.get("provider", "anthropic")
    openai_client = None
    if provider == "openai_compat":
        from openai import OpenAI

        openai_client = OpenAI(api_key=api_env["api_key"], base_url=api_env["base_url"])
    else:
        api_env.pop("provider", None)
        client = anth.Anthropic(**api_env)
    state = AgentState()
    tools = anthropic_tool_defs()
    model = config["model"]

    system_prompt = build_system_prompt(cwd=str(Path.cwd()), bare=config.get("bare", False))

    if openai_client is not None:
        from nano_claw_code.openai_compat import run_streaming_openai

        event_iter = run_streaming_openai(
            prompt, state,
            client=openai_client, model=model, system_prompt=system_prompt,
            tools=tools, cwd=Path.cwd(),
            thinking=config.get("thinking", False),
            thinking_budget=config.get("thinking_budget", 10_000),
            permission_mode="accept-all",
        )
    else:
        event_iter = run_streaming(
            prompt, state,
            client=client, model=model, system_prompt=system_prompt,
            tools=tools, cwd=Path.cwd(),
            thinking=config.get("thinking", False),
            thinking_budget=config.get("thinking_budget", 10_000),
            permission_mode="accept-all",
        )
    for event in event_iter:
        if isinstance(event, TextChunk):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolStart):
            desc = _tool_desc(event.name, event.inputs)
            print(f"\n[{desc}]", file=sys.stderr)
        elif isinstance(event, ToolEnd):
            if event.result.startswith("Error:"):
                print(f"[{event.name} error: {event.result[:120]}]", file=sys.stderr)
        elif isinstance(event, TurnDone):
            if config.get("verbose"):
                print(f"\n[tokens: +{event.input_tokens} in / +{event.output_tokens} out]",
                      file=sys.stderr)

    print()
    return 0


# ── Argument parser ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nano-claw-code",
        description="Nano Claw Code — Python coding agent (ported from nano-claude-code)",
        add_help=True,
    )
    p.add_argument("prompt", nargs="*", help="Initial prompt (positional, for scripting)")
    p.add_argument("-p", "--print", dest="print_prompt", default=None,
                   help="Non-interactive prompt (same flag as Claude Code)")
    p.add_argument("--model", default=None, help="Anthropic model id")
    p.add_argument("--max-turns", type=int, default=50)
    p.add_argument("--output-format", default="text", choices=["text", "json", "stream-json"],
                   help="Output format: text (interactive), json, or stream-json (harness)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--thinking", action="store_true", help="Enable extended thinking")
    p.add_argument("--thinking-budget", type=int, default=10_000,
                   help="Token budget for extended thinking")
    p.add_argument("--dangerously-skip-permissions", action="store_true",
                   help="Bypass all permission checks")
    p.add_argument("--no-session-persistence", action="store_true",
                   help="Disable session persistence (no-op)")
    p.add_argument("--bare", action="store_true",
                   help="Lean system prompt (closer to Claude Code --bare)")
    p.add_argument("--accept-all", action="store_true",
                   help="Never ask permission (accept all operations)")
    p.add_argument("--streaming", action="store_true",
                   help="Use streaming API for print mode")
    p.add_argument("--permission-mode", default=None, choices=PERMISSION_MODES,
                   help="Permission mode: auto, accept-all, manual")
    p.add_argument("--resume", nargs="?", const="latest", default=None,
                   help="Resume a previous session. Use --resume for latest, or --resume <filename>")
    p.add_argument("-n", "--name", default=None,
                   help="Set a display name/title for this session (shown in /load)")
    p.add_argument("--version", action="store_true", help="Print version and exit")
    return p


# ── Main entry ────────────────────────────────────────────────────────────

def _load_resume_session(resume_arg: str) -> dict | None:
    """Load a session for --resume. 'latest' loads the most recent session.

    Also supports searching by title if the arg is not a filename.
    """
    if resume_arg == "latest":
        return load_latest_session()
    # Try exact filename first
    try:
        return load_session(resume_arg)
    except FileNotFoundError:
        pass
    # Try title search
    results = search_sessions(resume_arg)
    if results:
        try:
            return load_session(results[0]["filename"])
        except (FileNotFoundError, KeyError):
            pass
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.version:
        from nano_claw_code import __version__
        print(f"nano-claw-code v{__version__}")
        return 0

    config = load_config()

    if args.model:
        config["model"] = args.model
    if args.verbose:
        config["verbose"] = True
    if args.thinking:
        config["thinking"] = True
    config["thinking_budget"] = args.thinking_budget
    if args.bare:
        config["bare"] = True
    if args.accept_all or args.dangerously_skip_permissions:
        config["permission_mode"] = "accept-all"
    if args.permission_mode:
        config["permission_mode"] = args.permission_mode
    if args.name:
        config["_session_title"] = args.name.strip()

    prompt = args.print_prompt
    if not prompt and args.prompt:
        prompt = " ".join(args.prompt)

    if prompt:
        if args.output_format == "stream-json":
            return run_agent_loop(
                cwd=Path(os.getcwd()), user_prompt=prompt, model=args.model,
                max_turns=args.max_turns, bare=args.bare, verbose=args.verbose,
                streaming=args.streaming, thinking=args.thinking,
                thinking_budget=args.thinking_budget,
            )
        elif args.output_format == "json":
            return run_agent_loop(
                cwd=Path(os.getcwd()), user_prompt=prompt, model=args.model,
                max_turns=args.max_turns, bare=args.bare, verbose=args.verbose,
                streaming=args.streaming, thinking=args.thinking,
                thinking_budget=args.thinking_budget,
            )
        else:
            return _run_print_text(prompt, config, max_turns=args.max_turns)

    resume_state = None
    if args.resume:
        resume_state = _load_resume_session(args.resume)
        if resume_state is None:
            _err("No session found to resume.")
            return 1

    return run_repl(config, resume_state=resume_state)


if __name__ == "__main__":
    raise SystemExit(main())
