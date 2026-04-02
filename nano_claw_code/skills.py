"""Skill discovery, parsing, and execution.

Scans filesystem directories for SKILL.md files, parses YAML-like frontmatter
for metadata, and executes skills either inline (expand into conversation) or
forked (run in a sub-agent loop).

Skill directories searched (in order, later entries override on name collision):
  1. ~/.claude/skills/*/SKILL.md        (global user skills)
  2. ~/.nano_claw/skills/*/SKILL.md     (nano-claw-specific global)
  3. .claude/skills/*/SKILL.md          (project-local, walks up to git root)

Frontmatter (Claude Code–aligned; all optional except conventions):
  name, description, when-to-use / when_to_use
  allowed-tools / tools — comma-separated tool names (* = all for fork mode)
  model — override model for forked execution
  context — inline | fork
  user-invocable, disable-model-invocation
  max-turns / max_turns — fork sub-loop limit (default 10)
  version — shown in listings
  argument-hint — short hint for slash-command args (e.g. path or flags)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from nano_claw_code.config import CONFIG_DIR
from nano_claw_code.frontmatter import (
    meta_bool,
    meta_int,
    parse_comma_list,
    parse_markdown_frontmatter,
)

_SKILL_FILENAME = "SKILL.md"


def _git_toplevel(start: Path) -> Path | None:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start), stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        return Path(out)
    except Exception:
        return None


def _walk_up_for_skills(start: Path) -> list[Path]:
    """Walk from *start* up to git root collecting .claude/skills directories."""
    dirs: list[Path] = []
    top = _git_toplevel(start)
    p = start.resolve()
    for _ in range(20):
        candidate = p / ".claude" / "skills"
        if candidate.is_dir():
            dirs.append(candidate)
        if top and p == top:
            break
        parent = p.parent
        if parent == p:
            break
        p = parent
    return dirs


def _scan_skill_dir(base: Path) -> dict[str, dict[str, Any]]:
    """Scan one skills directory, returning {name: skill_info}."""
    skills: dict[str, dict[str, Any]] = {}
    if not base.is_dir():
        return skills
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        md = entry / _SKILL_FILENAME
        if not md.is_file():
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="replace")[:80_000]
        except OSError:
            continue
        meta, body = parse_markdown_frontmatter(text)
        name = (meta.get("name") or entry.name).lower().replace(" ", "-")
        allowed_raw = (meta.get("allowed-tools") or meta.get("tools") or "").strip()
        allowed_tools: list[str] = []
        if allowed_raw and allowed_raw != "*":
            allowed_tools = parse_comma_list(allowed_raw)

        skills[name] = {
            "name": name,
            "display_name": meta.get("name", entry.name),
            "description": meta.get("description", ""),
            "when_to_use": meta.get("when_to_use", meta.get("when-to-use", "")),
            "allowed_tools": allowed_tools,
            "model": meta.get("model", "").strip(),
            "context": (meta.get("context", "inline") or "inline").lower(),
            "user_invocable": meta_bool(meta, "user-invocable", "user_invocable", default=True),
            "disable_model_invocation": meta_bool(
                meta, "disable-model-invocation", "disable_model_invocation", default=False
            ),
            "max_turns": meta_int(meta, "max-turns", "max_turns"),
            "version": meta.get("version", "").strip(),
            "argument_hint": meta.get("argument-hint", meta.get("argument_hint", "")).strip(),
            "prompt": body,
            "path": str(md),
            "source_dir": str(base),
        }
    return skills


def discover_skills(cwd: str | None = None) -> dict[str, dict[str, Any]]:
    """Discover all available skills, merged from all sources.

    Later sources override earlier ones on name collision, so project-local
    skills take precedence over global ones.
    """
    merged: dict[str, dict[str, Any]] = {}

    global_claude = Path.home() / ".claude" / "skills"
    merged.update(_scan_skill_dir(global_claude))

    nano_claw_skills = CONFIG_DIR / "skills"
    merged.update(_scan_skill_dir(nano_claw_skills))

    start = Path(cwd) if cwd else Path.cwd()
    for project_dir in reversed(_walk_up_for_skills(start)):
        merged.update(_scan_skill_dir(project_dir))

    return merged


_skill_cache: dict[str, dict[str, Any]] | None = None
_skill_cache_cwd: str | None = None


def get_skills(cwd: str | None = None) -> dict[str, dict[str, Any]]:
    """Get cached skills for the current working directory."""
    global _skill_cache, _skill_cache_cwd
    effective_cwd = cwd or os.getcwd()
    if _skill_cache is not None and _skill_cache_cwd == effective_cwd:
        return _skill_cache
    _skill_cache = discover_skills(effective_cwd)
    _skill_cache_cwd = effective_cwd
    return _skill_cache


def clear_skill_cache() -> None:
    global _skill_cache, _skill_cache_cwd
    _skill_cache = None
    _skill_cache_cwd = None


def get_skill_tool_commands(cwd: str | None = None) -> list[dict[str, Any]]:
    """Return skills that can be invoked by the model via the Skill tool."""
    skills = get_skills(cwd)
    return [
        s for s in skills.values()
        if not s.get("disable_model_invocation") and s.get("prompt")
    ]


def get_user_invocable_skills(cwd: str | None = None) -> list[dict[str, Any]]:
    """Return skills that users can invoke via /skill-name."""
    skills = get_skills(cwd)
    return [s for s in skills.values() if s.get("user_invocable", True)]


def format_skill_listing(skills: list[dict[str, Any]], max_chars: int = 2000) -> str:
    """Format skills for display in system prompt or tool listing."""
    if not skills:
        return "(no skills available)"
    lines: list[str] = []
    budget = max_chars
    for s in skills:
        name = s["name"]
        desc = s.get("description", "")
        when = s.get("when_to_use", "")
        ver = s.get("version", "")
        hint = s.get("argument_hint", "")
        line = f"- /{name}"
        if ver:
            line += f" (v{ver})"
        if desc:
            line += f": {desc}"
        if when:
            line += f" (use when: {when})"
        if hint:
            line += f" [args: {hint}]"
        if len(line) > 280:
            line = line[:277] + "..."
        if budget - len(line) < 0:
            lines.append(f"  ... and {len(skills) - len(lines)} more skills")
            break
        lines.append(line)
        budget -= len(line) + 1
    return "\n".join(lines)


def expand_skill_prompt(skill: dict[str, Any], args: str = "") -> str:
    """Expand a skill's prompt template with argument substitution."""
    prompt = skill.get("prompt", "")
    prompt = prompt.replace("$ARGUMENTS", args)
    prompt = prompt.replace("${ARGUMENTS}", args)
    skill_dir = str(Path(skill.get("path", "")).parent)
    prompt = prompt.replace("$CLAUDE_SKILL_DIR", skill_dir)
    prompt = prompt.replace("${CLAUDE_SKILL_DIR}", skill_dir)
    # Common Claude Code-style numbered args
    parts = re.split(r"\s+", args.strip()) if args.strip() else []
    prompt = prompt.replace("$1", parts[0] if len(parts) > 0 else "")
    prompt = prompt.replace("$2", parts[1] if len(parts) > 1 else "")
    prompt = prompt.replace("$3", parts[2] if len(parts) > 2 else "")
    return prompt


def execute_skill_forked(
    skill: dict[str, Any],
    args: str,
    cwd: Path,
) -> str:
    """Execute a skill in forked (sub-agent) mode."""
    from nano_claw_code.config import resolve_api_env
    from nano_claw_code.prompts import build_system_prompt, resolve_model
    from nano_claw_code.tools_impl import anthropic_tool_defs, dispatch_tool
    import anthropic as anth

    api_env = resolve_api_env()
    if not api_env.get("api_key"):
        return "Error: no API key available for skill sub-agent"

    api_env.pop("provider", None)
    client = anth.Anthropic(**api_env)

    model_override = skill.get("model")
    model = resolve_model(model_override if model_override else None)

    system = build_system_prompt(cwd=str(cwd), bare=True)
    prompt_text = expand_skill_prompt(skill, args)

    allowed = skill.get("allowed_tools") or []
    all_tools = [t for t in anthropic_tool_defs() if t["name"] not in ("Agent", "Skill")]
    if allowed:
        all_tools = [t for t in all_tools if t["name"] in allowed]

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt_text}]
    max_turns = skill.get("max_turns") or 10
    if not isinstance(max_turns, int) or max_turns < 1:
        max_turns = 10
    all_text: list[str] = []

    for _turn in range(max_turns):
        try:
            resp = client.messages.create(
                model=model, max_tokens=8192, system=system,
                tools=all_tools, messages=messages,
            )
        except Exception as e:
            return f"Skill sub-agent error: {e}"

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

    result_text = "\n".join(all_text) if all_text else "(skill produced no output)"
    return result_text
