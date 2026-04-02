"""End-to-end CLI + harness workflows (requires working ANTHROPIC_API_KEY)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key.startswith("sk-ant-"):
        pytest.skip("ANTHROPIC_API_KEY (sk-ant-*) required for e2e")


def _run_cli(cwd: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env or {})}
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    return subprocess.run(
        [sys.executable, "-m", "nano_claw_code", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _last_result_line(stdout: str) -> dict | None:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "result":
            return obj
    return None


def _skip_if_subprocess_quota(p: subprocess.CompletedProcess, *, what: str) -> None:
    blob = (p.stderr or "") + (p.stdout or "")
    if p.returncode != 0 and any(
        x in blob for x in ("403", "429", "limit exceeded", "RateLimit", "PermissionDenied")
    ):
        pytest.skip(f"{what}: API/quota error (rc={p.returncode})")


@pytest.mark.e2e
def test_e2e_cli_version():
    p = _run_cli(PROJECT_ROOT, "--version")
    assert p.returncode == 0
    assert "nano-claw-code" in p.stdout.lower() or "0." in p.stdout


@pytest.mark.e2e
@pytest.mark.integration
def test_e2e_stream_json_harness_smoke(tmp_path: Path):
    """run_agent_loop stream-json: model replies without tools."""
    _require_api_key()
    model = os.environ.get("E2E_MODEL", os.environ.get("MODEL", "claude-sonnet-4-20250514"))
    p = _run_cli(
        tmp_path,
        "-p",
        "Reply with exactly this token and nothing else: E2E_STREAM_JSON_TOKEN",
        "--output-format",
        "stream-json",
        "--max-turns",
        "3",
        "--accept-all",
        "--model",
        model,
    )
    _skip_if_subprocess_quota(p, what="stream-json smoke")
    assert p.returncode == 0, (p.stdout, p.stderr)
    res = _last_result_line(p.stdout)
    assert res is not None, p.stdout[-2000:]
    assert res.get("subtype") == "success" or not res.get("is_error", True)
    blob = json.dumps(res)
    assert "E2E_STREAM_JSON_TOKEN" in blob or "E2E_STREAM_JSON_TOKEN" in p.stdout


@pytest.mark.e2e
@pytest.mark.integration
def test_e2e_memory_and_read_tool(tmp_path: Path):
    """CLAUDE.md in cwd is visible; agent uses Read."""
    _require_api_key()
    secret = "memory_marker_e2e_7f3a"
    (tmp_path / "CLAUDE.md").write_text(
        f"# Rules\nThe secret codeword is: {secret}\n", encoding="utf-8"
    )
    model = os.environ.get("E2E_MODEL", os.environ.get("MODEL", "claude-sonnet-4-20250514"))
    p = _run_cli(
        tmp_path,
        "-p",
        "Use the Read tool on the file CLAUDE.md in the current directory. "
        f"In your final text, include the exact substring {secret} once.",
        "--output-format",
        "stream-json",
        "--max-turns",
        "8",
        "--accept-all",
        "--model",
        model,
    )
    _skip_if_subprocess_quota(p, what="memory read e2e")
    assert p.returncode == 0, (p.stdout[-1500:], p.stderr[-500:])
    assert secret in p.stdout, "model output should contain memory-derived secret"


@pytest.mark.e2e
@pytest.mark.integration
def test_e2e_skill_and_agent_project_files(tmp_path: Path):
    """Custom skill markdown + Agent(Explore) bash (best-effort)."""
    _require_api_key()
    (tmp_path / ".claude" / "skills" / "e2e-skill").mkdir(parents=True)
    (tmp_path / ".claude" / "skills" / "e2e-skill" / "SKILL.md").write_text(
        "---\nname: e2e-skill\ndescription: test\ncontext: inline\n---\n"
        "When run, respond with the literal line: SKILL_E2E_INLINE_OK\n",
        encoding="utf-8",
    )
    (tmp_path / ".claude" / "agents").mkdir(parents=True)
    (tmp_path / ".claude" / "agents" / "e2e-bash.md").write_text(
        "---\nagent-type: e2e-bash\ndescription: run echo\ntools: Bash\nmax-turns: 3\n---\n"
        "You may only run Bash. Run: echo AGENT_E2E_BASH_MARKER\n",
        encoding="utf-8",
    )
    model = os.environ.get("E2E_MODEL", os.environ.get("MODEL", "claude-sonnet-4-20250514"))
    p = _run_cli(
        tmp_path,
        "-p",
        "Do two things in order: (1) Use the Skill tool with skill e2e-skill. "
        "(2) Use the Agent tool with subagent_type e2e-bash and prompt that matches your agent file. "
        "Your final summary must contain both SKILL_E2E_INLINE_OK and AGENT_E2E_BASH_MARKER if successful.",
        "--output-format",
        "stream-json",
        "--max-turns",
        "15",
        "--accept-all",
        "--model",
        model,
    )
    _skip_if_subprocess_quota(p, what="skill/agent e2e")
    assert p.returncode == 0, (p.stderr[-800:],)
    out = p.stdout
    assert "SKILL_E2E_INLINE_OK" in out, (
        "Skill inline path should emit marker; "
        f"stdout tail: {out[-1200:]!r}"
    )
    assert "AGENT_E2E_BASH_MARKER" in out, (
        "Agent custom + Bash path should emit marker; "
        f"stdout tail: {out[-1200:]!r}"
    )


@pytest.mark.e2e
@pytest.mark.integration
def test_e2e_print_text_mode_smoke(tmp_path: Path):
    """Non-interactive -p text mode (uses run_streaming; text may be empty if no stream deltas)."""
    _require_api_key()
    model = os.environ.get("E2E_MODEL", os.environ.get("MODEL", "claude-sonnet-4-20250514"))
    p = _run_cli(
        tmp_path,
        "-p",
        "Say only: TEXT_MODE_E2E_OK",
        "--max-turns",
        "2",
        "--accept-all",
        "--model",
        model,
    )
    _skip_if_subprocess_quota(p, what="print text e2e")
    assert p.returncode == 0, p.stderr
