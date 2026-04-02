#!/usr/bin/env bash
# Run SWE-bench Lite with nano-claw-code (Python), then evaluate.
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${OPENROUTER_API_KEY:-}" ] && [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
    export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
fi

if [ -n "${OPENROUTER_BASE_URL:-}" ] && [ -z "${ANTHROPIC_BASE_URL:-}" ]; then
    export ANTHROPIC_BASE_URL="$OPENROUTER_BASE_URL"
fi

if [ -n "${OPENROUTER_MODEL:-}" ] && [ -z "${MODEL:-}" ]; then
    export MODEL="$OPENROUTER_MODEL"
fi

if [ -n "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENROUTER_BASE_URL:-}" ] && [ -z "${ANTHROPIC_BASE_URL:-}" ]; then
    export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
fi

REPO_ROOT="$(cd .. && pwd)"

if ! python3 -c "import nano_claw_code" 2>/dev/null; then
    pip install -e "${REPO_ROOT}"
fi

echo "=== nano-claw-code: Generate predictions ==="
python run_swebench_claude_code.py "$@"

echo ""
echo "=== nano-claw-code: Evaluate predictions ==="
python run_swebench_claude_code.py --evaluate
