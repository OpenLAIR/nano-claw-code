"""Sub-agent definitions: built-ins (general-purpose, Explore, Plan) + custom .md agents.

Custom agents live in:
  ~/.claude/agents/*.md
  ~/.nano_claw/agents/*.md
  <project>/.claude/agents/*.md (each ancestor directory from root to cwd, later overrides)

Frontmatter (optional):
  name / agent-type: override id (default: filename stem)
  description / when-to-use: shown in listings
  tools: comma-separated allowlist (* = all)
  disallowed-tools: comma-separated denylist
  model: optional model id or empty for default
  max-turns: int
  omit-memory: true — sub-agent uses bare system prompt without CLAUDE hierarchy
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nano_claw_code.config import CONFIG_DIR
from nano_claw_code.frontmatter import (
    meta_bool,
    meta_int,
    parse_comma_list,
    parse_markdown_frontmatter,
)

_AGENT_FILE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*\.md$")


@dataclass
class AgentDefinition:
    agent_type: str
    when_to_use: str
    system_prompt: str
    tools: list[str] | None = None  # None = all except disallowed
    disallowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    max_turns: int | None = None
    omit_memory: bool = False
    source: str = "built-in"


def _builtin_general() -> AgentDefinition:
    body = """You are a sub-agent for Nano Claw Code. Complete the delegated task fully using your tools.

Strengths: searching codebases, reading multiple files, multi-step investigation.

Guidelines:
- Prefer Glob/Grep to locate code; Read when you know the path.
- NEVER create docs (*.md, README) unless the task explicitly requires it.
- Prefer editing existing files over creating new ones.
- When done, reply with a concise report for the parent agent."""
    return AgentDefinition(
        agent_type="general-purpose",
        when_to_use=(
            "General research and multi-step tasks. Use when you need broad codebase search "
            "or exploration without a specialized profile."
        ),
        system_prompt=body,
        tools=None,
        disallowed_tools=[],
        omit_memory=False,
    )


def _builtin_explore() -> AgentDefinition:
    body = """You are a read-only exploration sub-agent. STRICTLY prohibited from:
Write, Edit, NotebookEdit, or any file mutation (including via shell: touch, rm, cp, mv, redirects).

Use Glob, Grep, Read, and Bash only for read-only commands (ls, git status/log/diff, cat, head, tail).
Report findings in your reply; do not create files."""
    return AgentDefinition(
        agent_type="Explore",
        when_to_use=(
            "Fast read-only codebase exploration: find files, grep patterns, answer structural questions."
        ),
        system_prompt=body,
        tools=None,
        disallowed_tools=["Agent", "Write", "Edit", "NotebookEdit", "Skill"],
        omit_memory=True,
    )


def _builtin_plan() -> AgentDefinition:
    body = """You are a read-only planning sub-agent (software architect). Explore the codebase and
produce an implementation plan. You MUST NOT Write, Edit, NotebookEdit, or mutate files.

Process: understand requirements, explore with Glob/Grep/Read and read-only Bash, then output
step-by-step plan and trade-offs.

End with a section:
### Critical Files for Implementation
List 3–5 paths critical to the plan."""
    return AgentDefinition(
        agent_type="Plan",
        when_to_use=(
            "Design implementation plans: steps, dependencies, critical files — without modifying code."
        ),
        system_prompt=body,
        tools=None,
        disallowed_tools=["Agent", "Write", "Edit", "NotebookEdit", "Skill"],
        omit_memory=True,
    )


def builtin_agents() -> dict[str, AgentDefinition]:
    out: dict[str, AgentDefinition] = {}
    for factory in (_builtin_general, _builtin_explore, _builtin_plan):
        a = factory()
        out[_norm_key(a.agent_type)] = a
    return out


def _norm_key(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def _dirs_root_to_cwd(start: Path) -> list[Path]:
    dirs: list[Path] = []
    p = start.resolve()
    for _ in range(64):
        dirs.append(p)
        parent = p.parent
        if parent == p:
            break
        p = parent
    return list(reversed(dirs))


def _parse_agent_file(path: Path) -> AgentDefinition | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:80_000]
    except OSError:
        return None
    meta, body = parse_markdown_frontmatter(text)
    stem = path.stem.lower().replace(" ", "-")
    agent_type = (
        meta.get("agent-type")
        or meta.get("name")
        or stem
    ).strip()
    if not agent_type:
        agent_type = stem
    agent_type = agent_type  # keep Explore/Plan casing if user sets it
    when = (
        meta.get("when-to-use")
        or meta.get("description")
        or meta.get("when_to_use")
        or f"Custom agent loaded from {path.name}"
    )
    tools_raw = meta.get("tools", "").strip()
    tools: list[str] | None
    if not tools_raw or tools_raw == "*":
        tools = None
    else:
        tools = parse_comma_list(tools_raw)
    disallowed = parse_comma_list(
        meta.get("disallowed-tools", "") or meta.get("disallowed_tools", "")
    )
    model_raw = meta.get("model", "").strip()
    model = model_raw if model_raw and model_raw.lower() != "inherit" else None
    max_turns = meta_int(meta, "max-turns", "max_turns")
    omit_memory = meta_bool(meta, "omit-memory", "omit_memory")

    prompt = body.strip()
    if not prompt:
        return None

    return AgentDefinition(
        agent_type=agent_type,
        when_to_use=when,
        system_prompt=prompt,
        tools=tools,
        disallowed_tools=disallowed,
        model=model,
        max_turns=max_turns,
        omit_memory=omit_memory,
        source="file",
    )


def _scan_agents_dir(dir_path: Path) -> dict[str, AgentDefinition]:
    out: dict[str, AgentDefinition] = {}
    if not dir_path.is_dir():
        return out
    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".md":
            continue
        if not _AGENT_FILE.match(entry.name):
            continue
        ag = _parse_agent_file(entry)
        if ag:
            out[_norm_key(ag.agent_type)] = ag
    return out


def discover_agents(cwd: str | None = None) -> dict[str, AgentDefinition]:
    """Merge built-ins, user, nano_claw, and project agents (later overrides)."""
    merged = builtin_agents()

    home_agents = Path.home() / ".claude" / "agents"
    for k, v in _scan_agents_dir(home_agents).items():
        v.source = "user"
        merged[k] = v

    nc = CONFIG_DIR / "agents"
    for k, v in _scan_agents_dir(nc).items():
        v.source = "nano_claw"
        merged[k] = v

    start = Path(cwd or os.getcwd()).resolve()
    for d in _dirs_root_to_cwd(start):
        agents_dir = d / ".claude" / "agents"
        for k, v in _scan_agents_dir(agents_dir).items():
            v.source = "project"
            merged[k] = v

    return merged


_agent_cache: dict[str, AgentDefinition] | None = None
_agent_cache_cwd: str | None = None


def get_agents(cwd: str | None = None) -> dict[str, AgentDefinition]:
    global _agent_cache, _agent_cache_cwd
    effective = cwd or os.getcwd()
    if _agent_cache is not None and _agent_cache_cwd == effective:
        return _agent_cache
    _agent_cache = discover_agents(effective)
    _agent_cache_cwd = effective
    return _agent_cache


def clear_agent_cache() -> None:
    global _agent_cache, _agent_cache_cwd
    _agent_cache = None
    _agent_cache_cwd = None


def resolve_agent(subagent_type: str | None, cwd: str) -> AgentDefinition:
    agents = get_agents(cwd)
    if not subagent_type or not str(subagent_type).strip():
        return agents[_norm_key("general-purpose")]
    key = _norm_key(str(subagent_type).strip())
    if key in agents:
        return agents[key]
    # fuzzy: Explore, Plan exact case in values
    for ag in agents.values():
        if ag.agent_type.lower() == key:
            return ag
    return agents[_norm_key("general-purpose")]


def tools_summary(ag: AgentDefinition) -> str:
    deny = set(ag.disallowed_tools)
    if ag.tools:
        allowed = [t for t in ag.tools if t not in deny]
        return ", ".join(allowed) if allowed else "None"
    if deny:
        return "All except " + ", ".join(sorted(deny))
    return "All"


def format_agent_listing(cwd: str | None = None, max_chars: int = 3500) -> str:
    agents = list(get_agents(cwd).values())
    # Stable: built-in first by type order, then others sorted
    order = {"general-purpose": 0, "explore": 1, "plan": 2}
    agents.sort(key=lambda a: (order.get(_norm_key(a.agent_type), 99), a.agent_type))
    lines: list[str] = []
    budget = max_chars
    for ag in agents:
        line = f"- {ag.agent_type}: {ag.when_to_use} (Tools: {tools_summary(ag)})"
        if len(line) > 400:
            line = line[:397] + "..."
        if budget - len(line) < 0:
            lines.append(f"  ... and {len(agents) - len(lines)} more agents")
            break
        lines.append(line)
        budget -= len(line) + 1
    return "\n".join(lines) if lines else "(no agents)"


def filter_tools_for_agent(
    all_defs: list[dict[str, Any]],
    ag: AgentDefinition,
) -> list[dict[str, Any]]:
    """Filter anthropic tool defs by agent allow/deny lists."""
    names = {d["name"] for d in all_defs}
    deny = set(ag.disallowed_tools)
    if ag.tools:
        allow = set(ag.tools)
        selected = [d for d in all_defs if d["name"] in allow and d["name"] not in deny]
        return selected if selected else [d for d in all_defs if d["name"] not in deny]
    return [d for d in all_defs if d["name"] not in deny]
