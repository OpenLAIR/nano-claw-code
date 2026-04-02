"""Configuration management, cost tracking, multi-provider API resolution.

Priority (highest → lowest):
  1. Project-level .env file  (./env, ../.env, up to git root)
  2. Shell environment variables
  3. ~/.nano_claw/config.json
  4. Built-in defaults

Supported providers (auto-detected):
  - OpenAI-compatible (Azure AI, Kimi, etc.): OPENAI_COMPAT_BASE_URL + OPENAI_COMPAT_API_KEY
  - Anthropic direct  : ANTHROPIC_API_KEY=sk-ant-*
  - OpenRouter         : OPENROUTER_API_KEY=sk-or-*
  - Generic proxy      : ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".nano_claw"
SESSIONS_DIR = CONFIG_DIR / "sessions"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history"

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_COMPAT_MODEL = "Kimi-K2.5"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"

MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-haiku-3-5-20241022",
]

COST_PER_1K = {
    "claude-sonnet-4-20250514":  {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250514":    {"input": 0.015, "output": 0.075},
    "claude-haiku-3-5-20241022": {"input": 0.0008, "output": 0.004},
}

PERMISSION_MODES = ["auto", "accept-all", "manual"]


# ── .env file loading ─────────────────────────────────────────────────────

_ENV_LINE = re.compile(
    r"""^\s*(?:export\s+)?      # optional 'export '
    ([A-Za-z_][A-Za-z0-9_]*)    # key
    \s*=\s*                     # =
    (?:"([^"]*)"                # "double-quoted value"
    |'([^']*)'                  # 'single-quoted value'
    |([^\s#]*))                 # bare value (up to space/comment)
    """,
    re.VERBOSE,
)


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict. Supports quotes, export prefix, comments."""
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _ENV_LINE.match(line)
            if m:
                key = m.group(1)
                val = m.group(2) if m.group(2) is not None else (
                    m.group(3) if m.group(3) is not None else (m.group(4) or "")
                )
                result[key] = val
    except OSError:
        pass
    return result


def _find_dotenv_files() -> list[Path]:
    """Walk from CWD up to git root (or filesystem root), collecting .env files.

    Returns list ordered from *most specific* (CWD) to *least specific* (root).
    """
    cwd = Path.cwd().resolve()
    git_root = None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            git_root = Path(out.stdout.strip()).resolve()
    except Exception:
        pass

    found: list[Path] = []
    p = cwd
    while True:
        env_file = p / ".env"
        if env_file.is_file():
            found.append(env_file)
        if git_root and p == git_root:
            break
        parent = p.parent
        if parent == p:
            break
        p = parent
    return found


def load_dotenv() -> dict[str, str]:
    """Load .env files from CWD → git root. More-specific files win."""
    merged: dict[str, str] = {}
    for env_file in reversed(_find_dotenv_files()):
        merged.update(_parse_dotenv(env_file))
    return merged


def _env_get(key: str, dotenv: dict[str, str]) -> str:
    """Get a value: .env (highest priority) → shell env → empty string."""
    return dotenv.get(key) or os.environ.get(key) or ""


# ── Provider resolution ───────────────────────────────────────────────────

def resolve_api_env(api_key_override: str | None = None) -> dict[str, Any]:
    """Return ``{"api_key": ..., "base_url": ..., "provider": ...}`` for
    ``anthropic.Anthropic(**kwargs)``.

    Detection order:
      1. If OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_API_KEY are set
         → OpenAI Chat Completions compatible endpoint (Azure AI, Kimi, etc.)
      2. If .env or env has ANTHROPIC_API_KEY starting with ``sk-ant-``
         → direct Anthropic (ignore any OpenRouter base URL)
      3. If .env or env has OPENROUTER_API_KEY (sk-or-*)
         → OpenRouter proxy
      4. If ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL are set
         → generic proxy
      5. Fall back to whatever ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is

    Side effect: for ``sk-ant-*`` keys, if no ``ANTHROPIC_BASE_URL`` key appears in any
    merged ``.env`` file, ``os.environ['ANTHROPIC_BASE_URL']`` is removed. Shell configs
    often set that variable to OpenRouter for other tools; a bare ``Anthropic()`` would
    otherwise send official keys to the wrong host.
    """
    dotenv = load_dotenv()

    openai_compat_base = _env_get("OPENAI_COMPAT_BASE_URL", dotenv)
    openai_compat_key = api_key_override or _env_get("OPENAI_COMPAT_API_KEY", dotenv)
    if openai_compat_base and openai_compat_key:
        return {
            "api_key": openai_compat_key,
            "base_url": openai_compat_base.rstrip("/"),
            "provider": "openai_compat",
        }

    anthropic_key = api_key_override or _env_get("ANTHROPIC_API_KEY", dotenv) or _env_get("ANTHROPIC_AUTH_TOKEN", dotenv)
    if anthropic_key.startswith("sk-ant-") and "ANTHROPIC_BASE_URL" not in dotenv:
        os.environ.pop("ANTHROPIC_BASE_URL", None)

    openrouter_key = _env_get("OPENROUTER_API_KEY", dotenv)
    base_url = _env_get("ANTHROPIC_BASE_URL", dotenv)
    openrouter_base = _env_get("OPENROUTER_BASE_URL", dotenv)

    # ── Case 1: native Anthropic key ──
    if anthropic_key.startswith("sk-ant-"):
        return {
            "api_key": anthropic_key,
            "base_url": ANTHROPIC_DEFAULT_BASE_URL,
            "provider": "anthropic",
        }

    # ── Case 2: OpenRouter key ──
    if openrouter_key:
        return {
            "api_key": openrouter_key,
            "base_url": openrouter_base or OPENROUTER_BASE_URL,
            "provider": "openrouter",
        }

    # ── Case 3: generic proxy (ANTHROPIC_API_KEY + custom BASE_URL) ──
    if anthropic_key and base_url:
        return {
            "api_key": anthropic_key,
            "base_url": base_url,
            "provider": "proxy",
        }

    # ── Case 4: bare ANTHROPIC_API_KEY, no base URL ──
    if anthropic_key:
        return {
            "api_key": anthropic_key,
            "base_url": ANTHROPIC_DEFAULT_BASE_URL,
            "provider": "anthropic",
        }

    return {"api_key": "", "base_url": ANTHROPIC_DEFAULT_BASE_URL, "provider": "none"}


def resolve_model(dotenv: dict[str, str] | None = None, provider: str | None = None) -> str:
    """Resolve model from .env → shell env → default."""
    if dotenv is None:
        dotenv = load_dotenv()
    if provider is None:
        provider = resolve_api_env().get("provider", "none")
    if provider == "openai_compat":
        return (
            _env_get("OPENAI_COMPAT_MODEL", dotenv)
            or _env_get("MODEL", dotenv)
            or DEFAULT_OPENAI_COMPAT_MODEL
        )
    return (
        _env_get("ANTHROPIC_MODEL", dotenv)
        or _env_get("OPENROUTER_MODEL", dotenv)
        or _env_get("MODEL", dotenv)
        or DEFAULT_MODEL
    )


# ── Config load / save ────────────────────────────────────────────────────

def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    ensure_dirs()
    dotenv = load_dotenv()
    api_env = resolve_api_env()

    config: dict[str, Any] = {
        "model": resolve_model(dotenv, api_env.get("provider")),
        "max_tokens": 16_384,
        "permission_mode": "auto",
        "verbose": False,
        "thinking": False,
        "thinking_budget": 10_000,
        "api_key": api_env["api_key"],
        "provider": api_env["provider"],
    }

    if CONFIG_FILE.exists():
        try:
            stored = json.loads(CONFIG_FILE.read_text())
            for k, v in stored.items():
                if k not in ("api_key", "provider"):
                    config[k] = v
        except (json.JSONDecodeError, OSError):
            pass

    return config


def save_config(config: dict[str, Any]) -> None:
    ensure_dirs()
    safe = {k: v for k, v in config.items() if k not in ("api_key", "provider")}
    CONFIG_FILE.write_text(json.dumps(safe, indent=2))


# ── Cost tracking ─────────────────────────────────────────────────────────

def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K.get(model)
    if not rates:
        for key, r in COST_PER_1K.items():
            if key in model or model in key:
                rates = r
                break
    if not rates:
        rates = {"input": 0.003, "output": 0.015}
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]
