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


def test_parse_toml_options_nano_claw_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[nano_claw]\nmodel = "m1"\nverbose = true\nunknown = 1\n',
        encoding="utf-8",
    )
    d = cfg._parse_toml_options(p)
    assert d["model"] == "m1"
    assert d["verbose"] is True
    assert "unknown" not in d


def test_load_config_toml_project_overrides_user(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cfg, "_git_toplevel", lambda: repo.resolve())
    user = tmp_path / "homecfg"
    user.mkdir()
    (user / "config.toml").write_text('[nano_claw]\nmodel = "user-m"\n', encoding="utf-8")
    (repo / ".nano_claw").mkdir(parents=True)
    (repo / ".nano_claw" / "config.toml").write_text(
        '[nano_claw]\nmodel = "proj-m"\npermission_mode = "manual"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CONFIG_DIR", user)
    monkeypatch.setattr(cfg, "SESSIONS_DIR", user / "sessions")
    monkeypatch.setattr(cfg, "CONFIG_FILE", user / "nope.json")
    monkeypatch.setattr(cfg, "load_dotenv", lambda: {})
    monkeypatch.setattr(
        cfg,
        "resolve_api_env",
        lambda **k: {"api_key": "", "provider": "none", "base_url": cfg.ANTHROPIC_DEFAULT_BASE_URL},
    )
    c = cfg.load_config()
    assert c["model"] == "proj-m"
    assert c["permission_mode"] == "manual"


def test_load_config_env_model_skips_toml_model(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cfg, "_git_toplevel", lambda: repo.resolve())
    (repo / ".nano_claw").mkdir(parents=True)
    (repo / ".nano_claw" / "config.toml").write_text(
        '[nano_claw]\nmodel = "toml-m"\n',
        encoding="utf-8",
    )
    nc = tmp_path / "nc"
    nc.mkdir()
    monkeypatch.setattr(cfg, "CONFIG_DIR", nc)
    monkeypatch.setattr(cfg, "SESSIONS_DIR", nc / "sessions")
    monkeypatch.setattr(cfg, "CONFIG_FILE", nc / "nope.json")
    monkeypatch.setattr(cfg, "load_dotenv", lambda: {"MODEL": "env-m"})
    monkeypatch.setattr(
        cfg,
        "resolve_api_env",
        lambda **k: {"api_key": "", "provider": "none", "base_url": cfg.ANTHROPIC_DEFAULT_BASE_URL},
    )
    c = cfg.load_config()
    assert c["model"] == "env-m"


def test_load_config_json_overrides_toml(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cfg, "_git_toplevel", lambda: repo.resolve())
    (repo / ".nano_claw").mkdir(parents=True)
    (repo / ".nano_claw" / "config.toml").write_text(
        '[nano_claw]\nmodel = "toml-m"\nmax_tokens = 111\n',
        encoding="utf-8",
    )
    nc = tmp_path / "nc"
    nc.mkdir()
    cfg_file = nc / "config.json"
    cfg_file.write_text('{"model": "json-m", "max_tokens": 222}', encoding="utf-8")
    monkeypatch.setattr(cfg, "CONFIG_DIR", nc)
    monkeypatch.setattr(cfg, "SESSIONS_DIR", nc / "sessions")
    monkeypatch.setattr(cfg, "CONFIG_FILE", cfg_file)
    monkeypatch.setattr(cfg, "load_dotenv", lambda: {})
    monkeypatch.setattr(
        cfg,
        "resolve_api_env",
        lambda **k: {"api_key": "", "provider": "none", "base_url": cfg.ANTHROPIC_DEFAULT_BASE_URL},
    )
    c = cfg.load_config()
    assert c["model"] == "json-m"
    assert c["max_tokens"] == 222
