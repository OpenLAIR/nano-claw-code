"""Optional live Anthropic API smoke test."""

from __future__ import annotations

import os

import pytest

from nano_claw_code.config import load_dotenv, resolve_api_env, resolve_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def require_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key.startswith("sk-ant-"):
        pytest.skip("Set ANTHROPIC_API_KEY for integration tests")


def test_messages_create_minimal(require_key):
    """Use resolve_api_env() like the CLI — not bare Anthropic().

    A bare ``Anthropic()`` reads ``ANTHROPIC_BASE_URL`` from the process
    environment. Many setups export OpenRouter as that URL for other tools;
    requests then hit OpenRouter (403 \"openrouter.ai/settings/keys\") even when
    ``ANTHROPIC_API_KEY`` is an Anthropic ``sk-ant-*`` key.
    """
    import anthropic

    api = resolve_api_env()
    api.pop("provider", None)
    client = anthropic.Anthropic(**api)
    model = os.environ.get("INTEGRATION_MODEL") or resolve_model(load_dotenv())
    try:
        r = client.messages.create(
            model=model,
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply: OK"}],
        )
    except anthropic.PermissionDeniedError as e:
        pytest.skip(f"API permission/quota: {e}")
    except anthropic.RateLimitError as e:
        pytest.skip(f"Rate limited: {e}")
    text = "".join(getattr(b, "text", "") for b in r.content)
    assert len(text) > 0


def test_dispatch_agent_explore(require_key, tmp_path):
    """Sub-agent Explore + Bash (uses API + tool loop)."""
    from nano_claw_code.tools_impl import dispatch_tool

    (tmp_path / ".git").mkdir()  # optional
    out = dispatch_tool(
        tmp_path,
        "Agent",
        {
            "prompt": "Run Bash: echo AGENT_INTEGRATION_OK",
            "subagent_type": "Explore",
            "description": "echo",
        },
    )
    if isinstance(out, str) and (
        "Sub-agent error" in out or "403" in out or "limit exceeded" in out.lower()
    ):
        pytest.skip(f"Agent tool API error: {out[:200]}")
    assert "AGENT_INTEGRATION_OK" in out
