"""Configuration management, cost tracking, multi-provider API resolution.

Priority (highest → lowest) for non-secret options:
  1. Shell / project .env (model env vars: MODEL, ANTHROPIC_MODEL, …) — if set, TOML ``model`` is ignored
  2. ~/.nano_claw/config.json (slash-commands; overwrites TOML for keys present)
  3. TOML: ~/.nano_claw/config.toml, then ``.nano_claw/config.toml`` from git root → cwd (later file wins)
  4. Built-in defaults

API keys and base URLs still come only from .env / environment (never from TOML).

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
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

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

# Keys allowed in [nano_claw] or at file root (Codex-style flat keys also accepted at root).
TOML_OPTION_KEYS = frozenset({
    "model",
    "max_tokens",
    "permission_mode",
    "verbose",
    "thinking",
    "thinking_budget",
    "bare",
})

_MODEL_ENV_KEYS = ("MODEL", "ANTHROPIC_MODEL", "OPENROUTER_MODEL", "OPENAI_COMPAT_MODEL")


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


def _git_toplevel() -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            return Path(out.stdout.strip()).resolve()
    except Exception:
        pass
    return None


def _find_dotenv_files() -> list[Path]:
    """Walk from CWD up to git root (or filesystem root), collecting .env files.

    Returns list ordered from *most specific* (CWD) to *least specific* (root).
    """
    cwd = Path.cwd().resolve()
    git_root = _git_toplevel()

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


def _dotenv_or_env_sets_model(dotenv: dict[str, str]) -> bool:
    return any(bool((_env_get(k, dotenv) or "").strip()) for k in _MODEL_ENV_KEYS)


def _parse_toml_options(path: Path) -> dict[str, Any]:
    """Load a single TOML file; return flat options (only TOML_OPTION_KEYS)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except Exception:
        return {}

    if "nano_claw" in data and isinstance(data["nano_claw"], dict):
        table = data["nano_claw"]
    else:
        table = {k: v for k, v in data.items() if not isinstance(v, dict)}

    out: dict[str, Any] = {}
    for k in TOML_OPTION_KEYS:
        if k in table:
            out[k] = table[k]
    return out


def _project_toml_chain_root_to_cwd() -> list[Path]:
    """Paths to .nano_claw/config.toml from git root down to cwd (later overwrites earlier)."""
    cwd = Path.cwd().resolve()
    root = _git_toplevel() or cwd
    found: list[Path] = []
    p = cwd
    while True:
        t = p / ".nano_claw" / "config.toml"
        if t.is_file():
            found.append(t)
        if p == root:
            break
        parent = p.parent
        if parent == p:
            break
        p = parent
    return list(reversed(found))


def load_merged_toml_options() -> dict[str, Any]:
    """Merge TOML options: user ~/.nano_claw/config.toml, then repo root → cwd."""
    merged: dict[str, Any] = {}
    user = CONFIG_DIR / "config.toml"
    if user.is_file():
        merged.update(_parse_toml_options(user))
    for path in _project_toml_chain_root_to_cwd():
        merged.update(_parse_toml_options(path))
    return merged


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

    toml_opts = load_merged_toml_options()
    if not _dotenv_or_env_sets_model(dotenv) and "model" in toml_opts:
        config["model"] = str(toml_opts["model"])
    if "max_tokens" in toml_opts:
        try:
            config["max_tokens"] = int(toml_opts["max_tokens"])
        except (TypeError, ValueError):
            pass
    if "permission_mode" in toml_opts:
        pm = str(toml_opts["permission_mode"])
        if pm in PERMISSION_MODES:
            config["permission_mode"] = pm
    if "verbose" in toml_opts:
        config["verbose"] = bool(toml_opts["verbose"])
    if "thinking" in toml_opts:
        config["thinking"] = bool(toml_opts["thinking"])
    if "thinking_budget" in toml_opts:
        try:
            config["thinking_budget"] = int(toml_opts["thinking_budget"])
        except (TypeError, ValueError):
            pass
    if "bare" in toml_opts:
        config["bare"] = bool(toml_opts["bare"])

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
