"""agents.discover_agents, filter_tools_for_agent; skills.discover_skills."""

from __future__ import annotations

from pathlib import Path

from nano_claw_code.agents import (
    AgentDefinition,
    builtin_agents,
    discover_agents,
    filter_tools_for_agent,
    resolve_agent,
    tools_summary,
)
from nano_claw_code.skills import discover_skills, expand_skill_prompt


def test_builtin_agents_keys():
    b = builtin_agents()
    assert "general-purpose" in b
    assert "explore" in b
    assert "plan" in b
    assert b["explore"].omit_memory is True


def test_custom_agent_file(tmp_path: Path):
    (tmp_path / ".claude" / "agents").mkdir(parents=True)
    (tmp_path / ".claude" / "agents" / "reviewer.md").write_text(
        "---\nagent-type: code-reviewer\ndescription: reviews\ntools: Read, Grep\n---\nYou review.\n",
        encoding="utf-8",
    )
    agents = discover_agents(str(tmp_path))
    assert "code-reviewer" in agents
    ag = agents["code-reviewer"]
    assert ag.tools == ["Read", "Grep"]


def test_project_overrides_user_name(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude" / "agents").mkdir(parents=True)
    (home / ".claude" / "agents" / "x.md").write_text(
        "---\nname: shared\n---\nfrom user\n", encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".claude" / "agents").mkdir(parents=True)
    (proj / ".claude" / "agents" / "x.md").write_text(
        "---\nname: shared\n---\nfrom project\n", encoding="utf-8"
    )
    agents = discover_agents(str(proj))
    assert "shared" in agents
    assert "from project" in agents["shared"].system_prompt


def test_filter_tools_allow_deny():
    defs = [{"name": "Read"}, {"name": "Write"}, {"name": "Bash"}]
    ag = AgentDefinition(
        agent_type="t",
        when_to_use="w",
        system_prompt="s",
        tools=["Read", "Write"],
        disallowed_tools=["Write"],
    )
    names = {d["name"] for d in filter_tools_for_agent(defs, ag)}
    assert names == {"Read"}


def test_resolve_agent_unknown_defaults_general(tmp_path: Path):
    ag = resolve_agent("nonexistent-xyz", str(tmp_path))
    assert ag.agent_type == "general-purpose"


def test_skill_tools_alias_and_args(tmp_path: Path):
    (tmp_path / ".claude" / "skills" / "s1").mkdir(parents=True)
    (tmp_path / ".claude" / "skills" / "s1" / "SKILL.md").write_text(
        "---\nname: s1\ntools: Read, Bash\nversion: 1.2.3\nargument-hint: PATH\n---\n$ARGUMENTS $1\n",
        encoding="utf-8",
    )
    skills = discover_skills(str(tmp_path))
    assert skills["s1"]["allowed_tools"] == ["Read", "Bash"]
    assert skills["s1"]["version"] == "1.2.3"
    text = expand_skill_prompt(skills["s1"], "hello there")
    assert "hello there" in text
    assert "hello" in text  # $1


def test_tools_summary_all_except():
    ag = AgentDefinition(
        agent_type="x",
        when_to_use="y",
        system_prompt="z",
        tools=None,
        disallowed_tools=["Write"],
    )
    s = tools_summary(ag)
    assert "Write" in s
