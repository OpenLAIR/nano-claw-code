"""config: dotenv parse, resolve_api_env, resolve_model, calc_cost."""

from __future__ import annotations

import os
from pathlib import Path

from nano_claw_code import config as cfg


def test_parse_dotenv_quotes(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text('FOO="bar baz"\nexport BAR=x\n', encoding="utf-8")
    d = cfg._parse_dotenv(p)
    assert d["FOO"] == "bar baz"
    assert d["BAR"] == "x"


def test_resolve_model_from_mapping(monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    d = {"MODEL": "claude-test-model"}
    assert cfg.resolve_model(d) == "claude-test-model"


def test_resolve_api_env_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(cfg, "load_dotenv", lambda: {"ANTHROPIC_API_KEY": "sk-ant-test123"})
    out = cfg.resolve_api_env()
    assert out["provider"] == "anthropic"
    assert out["api_key"] == "sk-ant-test123"


def test_resolve_api_env_sk_ant_strips_shell_base_url(monkeypatch):
    """Inherited ANTHROPIC_BASE_URL (e.g. OpenRouter) must not stay in os.environ."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fromshell")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
    monkeypatch.setattr(
        cfg,
        "load_dotenv",
        lambda: {"ANTHROPIC_API_KEY": "sk-ant-fromshell"},
    )
    cfg.resolve_api_env()
    assert os.environ.get("ANTHROPIC_BASE_URL") is None


def test_resolve_api_env_sk_ant_keeps_base_url_when_in_dotenv(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setattr(
        cfg,
        "load_dotenv",
        lambda: {
            "ANTHROPIC_API_KEY": "sk-ant-x",
            "ANTHROPIC_BASE_URL": "https://proxy.example/v1",
        },
    )
    cfg.resolve_api_env()
    assert os.environ.get("ANTHROPIC_BASE_URL") == "https://proxy.example/v1"


def test_calc_cost_unknown_model_uses_default_rate():
    c = cfg.calc_cost("unknown-model-xyz", 1000, 1000)
    assert c > 0


def test_resolve_api_env_openai_compat(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.setattr(
        cfg,
        "load_dotenv",
        lambda: {
            "OPENAI_COMPAT_BASE_URL": "https://example.azure.com/openai/v1/",
            "OPENAI_COMPAT_API_KEY": "azure-secret",
        },
    )
    out = cfg.resolve_api_env()
    assert out["provider"] == "openai_compat"
    assert out["api_key"] == "azure-secret"
    assert out["base_url"] == "https://example.azure.com/openai/v1"


def test_resolve_model_openai_compat(monkeypatch):
    monkeypatch.setattr(
        cfg,
        "load_dotenv",
        lambda: {"OPENAI_COMPAT_MODEL": "Kimi-K2.5"},
    )
    monkeypatch.setattr(
        cfg,
        "resolve_api_env",
        lambda **kw: {"provider": "openai_compat", "api_key": "x", "base_url": "https://x/v1"},
    )
    assert cfg.resolve_model() == "Kimi-K2.5"
