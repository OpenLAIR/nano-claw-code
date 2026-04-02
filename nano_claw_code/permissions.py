"""Permission system: auto, accept-all, manual modes with safe command detection.

Ported from nano-claude-code TypeScript (utils/permissions/).
Features: pre-approved WebFetch domains for documentation sites.
"""

from __future__ import annotations

from urllib.parse import urlparse

SAFE_BASH_PREFIXES = (
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "date",
    "which", "type", "env", "printenv", "uname", "whoami", "id",
    "git log", "git status", "git diff", "git show", "git branch",
    "git remote", "git stash list", "git tag",
    "find ", "grep ", "rg ", "ag ", "fd ",
    "python ", "python3 ", "node ", "ruby ", "perl ",
    "pip show", "pip list", "pip3 show", "pip3 list",
    "npm list", "cargo metadata",
    "df ", "du ", "free ", "top -bn", "ps ",
    "curl -I", "curl --head",
    "file ", "stat ", "readlink ",
)

DANGEROUS_BASH_PATTERNS = (
    "rm -rf", "rm -r", "rmdir",
    "mkfs", "dd if=",
    "chmod 777", "chmod -R",
    "> /dev/", "curl | bash", "curl | sh",
    "wget -O - |",
    "eval ", "exec ",
)

# Documentation and reference sites that don't need permission to fetch.
# These are read-only, public resources commonly needed during development.
PREAPPROVED_WEBFETCH_DOMAINS = frozenset({
    # Language / framework docs
    "docs.python.org",
    "docs.rs",
    "doc.rust-lang.org",
    "go.dev",
    "pkg.go.dev",
    "developer.mozilla.org",
    "nodejs.org",
    "docs.npmjs.com",
    "react.dev",
    "vuejs.org",
    "angular.io",
    "nextjs.org",
    "docs.djangoproject.com",
    "flask.palletsprojects.com",
    "fastapi.tiangolo.com",
    "docs.sqlalchemy.org",
    "docs.docker.com",
    "kubernetes.io",
    "docs.github.com",
    "git-scm.com",
    # API references
    "docs.anthropic.com",
    "platform.openai.com",
    "pypi.org",
    "crates.io",
    "www.npmjs.com",
    "rubygems.org",
    # Knowledge bases
    "en.wikipedia.org",
    "stackoverflow.com",
    "www.stackoverflow.com",
    "stackexchange.com",
    # Package registries (read-only)
    "registry.npmjs.org",
    "pypi.python.org",
})


def is_safe_bash(cmd: str) -> bool:
    c = cmd.strip()
    return any(c.startswith(p) or c == p.rstrip() for p in SAFE_BASH_PREFIXES)


def is_dangerous_bash(cmd: str) -> bool:
    c = cmd.strip().lower()
    return any(p in c for p in DANGEROUS_BASH_PATTERNS)


def is_preapproved_url(url: str) -> bool:
    """Check if a URL is for a pre-approved documentation/reference domain."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        return host in PREAPPROVED_WEBFETCH_DOMAINS
    except Exception:
        return False


def needs_permission(tool_name: str, tool_input: dict, permission_mode: str) -> bool:
    """Return True if this tool invocation needs user permission."""
    if permission_mode == "accept-all":
        return False
    if permission_mode == "manual":
        return True

    if tool_name in ("Read", "Glob", "Grep", "WebSearch", "TodoWrite"):
        return False
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        if is_safe_bash(cmd):
            return False
        return True
    if tool_name in ("Write", "Edit", "NotebookEdit"):
        return True
    if tool_name == "WebFetch":
        url = tool_input.get("url", "")
        if is_preapproved_url(url):
            return False
        return True
    if tool_name in ("Agent", "Skill"):
        return False
    return True


def describe_permission(tool_name: str, tool_input: dict) -> str:
    if tool_name == "Bash":
        return f"Run: {tool_input.get('command', '')[:100]}"
    if tool_name == "Write":
        return f"Write to: {tool_input.get('file_path', '')}"
    if tool_name == "Edit":
        return f"Edit: {tool_input.get('file_path', '')}"
    if tool_name == "NotebookEdit":
        return f"Edit notebook: {tool_input.get('target_notebook', '')}"
    if tool_name == "WebFetch":
        return f"Fetch URL: {tool_input.get('url', '')[:80]}"
    return f"{tool_name}: {list(tool_input.values())[:1]}"
