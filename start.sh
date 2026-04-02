#!/usr/bin/env bash
#
# Nano Claw Code — Python launcher
#
# Usage:
#   ./start.sh                     # Interactive REPL
#   ./start.sh -p "your prompt"    # Non-interactive mode
#   ./start.sh --help              # Show help
#
# API configuration — put a .env file in the project directory:
#
#   # Direct Anthropic (highest priority if sk-ant-* key detected)
#   ANTHROPIC_API_KEY=sk-ant-xxx
#
#   # OR OpenRouter
#   OPENROUTER_API_KEY=sk-or-xxx
#   OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514
#
#   # OR generic proxy
#   ANTHROPIC_API_KEY=your-key
#   ANTHROPIC_BASE_URL=https://your-proxy.com
#

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# ── Load project .env (highest priority, before any env normalization) ────
# Walk from $DIR upward to git root looking for .env files.
# More-specific (closer to CWD) values win by loading last.
_load_dotenv() {
  local f="$1"
  [ -f "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"                     # strip comments
    line="$(echo "$line" | sed 's/^export //')"  # strip 'export '
    line="$(echo "$line" | xargs)"        # trim whitespace
    [ -z "$line" ] && continue
    case "$line" in
      *=*)
        local key="${line%%=*}"
        local val="${line#*=}"
        val="${val%\"}" ; val="${val#\"}"  # strip double quotes
        val="${val%\'}" ; val="${val#\'}"  # strip single quotes
        export "$key=$val" 2>/dev/null || true
        ;;
    esac
  done < "$f"
}

# Collect .env files from git root → CWD (CWD loaded last = highest priority)
_dotenv_files=()
_walk="$DIR"
_git_root=""
if _git_root="$(git -C "$DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
fi
while true; do
  [ -f "$_walk/.env" ] && _dotenv_files=("$_walk/.env" "${_dotenv_files[@]}")
  [ "$_walk" = "$_git_root" ] && break
  _parent="$(dirname "$_walk")"
  [ "$_parent" = "$_walk" ] && break
  _walk="$_parent"
done
for _ef in "${_dotenv_files[@]}"; do
  _load_dotenv "$_ef"
done

# ── Check auth (after .env is loaded) ────────────────────────────────────

_has_key=0
[ -n "${ANTHROPIC_API_KEY:-}" ] && _has_key=1
[ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] && _has_key=1
[ -n "${OPENROUTER_API_KEY:-}" ] && _has_key=1

if [ "$_has_key" -eq 0 ]; then
  echo "Error: no API credential detected."
  echo ""
  echo "Create a .env file in this directory with one of:"
  echo ""
  echo '  # Direct Anthropic'
  echo '  ANTHROPIC_API_KEY=sk-ant-xxx'
  echo ""
  echo '  # OR OpenRouter'
  echo '  OPENROUTER_API_KEY=sk-or-xxx'
  echo ""
  echo "Or export the variable in your shell."
  exit 1
fi

# ── Find Python and ensure package is installed ───────────────────────────

PYTHON=""
for p in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$p" &>/dev/null; then
    PYTHON="$p"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Error: Python 3.10+ not found. Install Python first."
  exit 1
fi

if [ -f "$DIR/.venv/bin/python" ]; then
  PYTHON="$DIR/.venv/bin/python"
elif ! "$PYTHON" -c "import nano_claw_code" 2>/dev/null; then
  echo "First run: installing nano-claw-code..."
  "$PYTHON" -m pip install -e "$DIR" --quiet 2>/dev/null || \
    "$PYTHON" -m pip install -e "$DIR" --user --quiet 2>/dev/null || \
    echo "Warning: pip install failed. Try: pip install -e $DIR"
fi

# ── Launch ────────────────────────────────────────────────────────────────

exec "$PYTHON" -m nano_claw_code "$@"
