"""permissions.needs_permission, describe_permission."""

from __future__ import annotations

from nano_claw_code.permissions import describe_permission, is_preapproved_url, needs_permission


def test_auto_read_free():
    assert needs_permission("Read", {}, "auto") is False


def test_auto_write_needs():
    assert needs_permission("Write", {"file_path": "x"}, "auto") is True


def test_accept_all():
    assert needs_permission("Write", {"file_path": "x"}, "accept-all") is False


def test_manual_all():
    assert needs_permission("Read", {}, "manual") is True


def test_bash_safe_git_status():
    assert needs_permission("Bash", {"command": "git status"}, "auto") is False


def test_bash_rm_needs():
    assert needs_permission("Bash", {"command": "rm -f a"}, "auto") is True


def test_webfetch_docs_preapproved():
    assert is_preapproved_url("https://docs.python.org/3/library/os.html") is True
    assert needs_permission(
        "WebFetch",
        {"url": "https://docs.python.org/3/library/os.html"},
        "auto",
    ) is False


def test_describe_bash():
    s = describe_permission("Bash", {"command": "ls -la"})
    assert "ls" in s
