"""System prompt construction with CLAUDE.md, git info, and environment context.

Ported from nano-claude-code TypeScript (constants/prompts.ts, getSystemPrompt).
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from typing import Any


def resolve_model(cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    from nano_claw_code.config import resolve_model as _resolve
    return _resolve()


def _get_git_info(cwd: str) -> str:
    parts: list[str] = []
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        parts.append(f"- Git branch: {branch}")
    except Exception:
        return ""
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        if status:
            lines = status.split("\n")[:15]
            parts.append("- Git status:\n" + "\n".join(f"    {l}" for l in lines))
    except Exception:
        pass
    try:
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        if log:
            parts.append("- Recent commits:\n" + "\n".join(f"    {l}" for l in log.split("\n")))
    except Exception:
        pass
    return "\n".join(parts) + "\n" if parts else ""


def _get_claude_md(cwd: str) -> str:
    from nano_claw_code.memory import load_memory_context

    return load_memory_context(cwd)


FULL_SYSTEM_PROMPT = """\
You are Nano Claw Code, an interactive AI coding agent running in the terminal.
You help users with software engineering tasks: writing code, debugging, refactoring, \
running tests, and resolving issues in codebases.

# Available Tools
You have access to these tools to accomplish tasks:

- **Read**: Read file contents with line numbers. Supports offset/limit for large files. \
If given a directory, returns a listing.
- **Write**: Create or overwrite files. Creates parent directories automatically.
- **Edit**: Replace exact text in a file (search/replace). old_string must match exactly \
including whitespace. Include enough context to make old_string unique.
- **Bash**: Execute shell commands (cwd persists across calls). Use for tests, git, \
build tools, package managers, etc.
- **Glob**: Find files by glob pattern (e.g. **/*.py). Auto-prepends **/ for simple patterns. \
Returns paths sorted by modification time.
- **Grep**: Search file contents with regex (uses ripgrep if available). Supports output \
modes: content, files_with_matches, count.
- **WebFetch**: Fetch a URL and return readable text content. HTML is auto-converted. \
Will fail for authenticated URLs.
- **WebSearch**: Search the web for information via DuckDuckGo.
- **NotebookEdit**: Edit or create Jupyter notebook cells.
- **TodoWrite**: Create or update a task list to track progress on multi-step tasks.
- **Agent**: Launch a sub-agent. Set optional **subagent_type** to `general-purpose`, \
`Explore` (read-only search), `Plan` (read-only planning), or a custom id from \
`.claude/agents/*.md`. Omit **subagent_type** for the default general-purpose agent.
- **Skill**: Execute a custom skill loaded from `.claude/skills/` directories. \
Skills are user-defined prompts that extend your capabilities. Use the Skill tool only for \
skills listed in the skill listing — do not guess skill names.

# Guidelines
- Be concise and direct. Lead with the answer, not the process.
- Always read files before editing to understand current state.
- Use absolute paths for file operations when possible.
- Prefer editing existing files over creating new ones.
- Make the smallest change that fixes the issue.
- Do not add unrelated refactors or new files unless necessary.
- After substantive edits, run targeted tests if the repo has a test command.
- For multi-step tasks, use TodoWrite to track progress.
- When using Edit, include enough context in old_string to make it unique.
- Do not add obvious/redundant code comments.
- For Bash, keep commands non-interactive. Prefer tools over raw shell commands.

# Tone and Style
- Be concise. No filler phrases.
- Reference code locations as `path:line` when helpful.
- Use backticks for file/function/class names.
- Output text to communicate; use tools to accomplish tasks.

# Environment
- Working directory: {cwd}
- Date: {date}
- Platform: {platform}
{git_info}{claude_md}\
# Completion
- When finished with the task, end your text reply with the word DONE on its own line.
"""

BARE_SYSTEM_PROMPT = """\
You are Nano Claw Code, a coding agent.

CWD: {cwd}
Date: {date}

Tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch, NotebookEdit, TodoWrite, Agent, Skill.
Make minimal, targeted changes. Skip long explanations. Prefer tools over speculation.
When finished, end with DONE on its own line."""


def _get_agent_listing(cwd: str) -> str:
    try:
        from nano_claw_code.agents import format_agent_listing

        listing = format_agent_listing(cwd)
        if not listing or listing == "(no agents)":
            return ""
        return (
            "\n# Available sub-agents\n"
            "Use the Agent tool with **subagent_type** to select a profile:\n\n"
            f"{listing}\n"
        )
    except Exception:
        return ""


def _get_skill_listing(cwd: str) -> str:
    """Build a skill listing section for the system prompt."""
    try:
        from nano_claw_code.skills import get_skill_tool_commands, format_skill_listing
        skills = get_skill_tool_commands(cwd)
        if not skills:
            return ""
        listing = format_skill_listing(skills)
        return (
            "\n# Available Skills\n"
            "The following skills are available via the Skill tool. "
            "Use /<skill-name> as shorthand or call the Skill tool directly.\n\n"
            f"{listing}\n"
        )
    except Exception:
        return ""


def build_system_prompt(*, cwd: str, bare: bool = False) -> str:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %A UTC")
    if bare:
        return BARE_SYSTEM_PROMPT.format(cwd=cwd, date=date)
    git_info = _get_git_info(cwd)
    claude_md = _get_claude_md(cwd)
    agent_listing = _get_agent_listing(cwd)
    skill_listing = _get_skill_listing(cwd)
    base = FULL_SYSTEM_PROMPT.format(
        cwd=cwd,
        date=date,
        platform=platform.system(),
        git_info=git_info,
        claude_md=claude_md,
    )
    return base + agent_listing + skill_listing


def build_subagent_system_prompt(cwd: str, agent: Any) -> str:
    """System prompt for Agent tool sub-loop (AgentDefinition from agents module)."""
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %A UTC")
    header = f"You are a sub-agent ({agent.agent_type}).\nCWD: {cwd}\nDate: {date} UTC\n\n"
    if getattr(agent, "omit_memory", False):
        return header + agent.system_prompt
    mem = _get_claude_md(cwd)
    role = f"\n# Your instructions\n{agent.system_prompt}\n"
    return header + (mem if mem else "") + role
