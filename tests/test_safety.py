"""Guardrail tests: anything outside the allow-lists or the workspace must be refused."""

import os
import sys

import pytest

from voice_os_agent.safety.validator import (
    needs_confirmation,
    normalize_url,
    resolve_in_workspace,
    validate_parsed,
)


def v(**parsed):
    return validate_parsed(parsed)


# ---------- path sandbox ----------
@pytest.mark.parametrize(
    "path",
    [
        "../../etc/passwd",
        "../secret.txt",
        "a/../../b",
        "/etc/passwd",
        "~/.ssh/id_rsa",
        "~",
        "..\\..\\windows\\system32\\evil.dll",
        "C:\\Windows\\System32\\config\\SAM",
        "c:evil.txt",
        "\\\\server\\share\\x",
        "file\x00.txt",
        "",
        "..",
    ],
)
def test_resolve_rejects_escapes(path):
    assert resolve_in_workspace(path) is None


@pytest.mark.parametrize("path", ["report.txt", "sub/dir/notes.md", "./a.txt", "sub\\b.txt"])
def test_resolve_accepts_relative_paths_inside(path, workspace):
    resolved = resolve_in_workspace(path)
    assert resolved is not None and workspace in resolved.parents


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin rights on Windows")
def test_resolve_rejects_symlink_pointing_outside(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, workspace / "link")
    assert resolve_in_workspace("link/secret.txt") is None


@pytest.mark.parametrize("intent", ["delete_file", "create_file", "open_file", "rename_file", "move_file"])
def test_file_intents_refuse_traversal(intent):
    ok, reason, _ = v(intent=intent, filename="../../etc/passwd", newname="x", folder="a")
    assert not ok and reason == "unsafe_path"


@pytest.mark.parametrize("dst", ["../../Desktop", "/tmp", "~/Documents"])
@pytest.mark.parametrize("intent", ["move_file", "copy_file"])
def test_move_copy_refuse_unsafe_destination(intent, dst):
    ok, reason, _ = v(intent=intent, filename="report.txt", folder=dst)
    assert not ok and reason == "unsafe_destination"


@pytest.mark.parametrize("folder", ["../../..", "/etc", "/tmp/evil", "~"])
@pytest.mark.parametrize("intent", ["list_dir", "create_folder", "search_files"])
def test_folder_intents_refuse_escape(intent, folder):
    ok, reason, _ = v(intent=intent, folder=folder)
    assert not ok and reason == "unsafe_path"


@pytest.mark.parametrize("new_name", ["../../escaped.txt", "sub/x.txt", "/tmp/x", ".."])
def test_rename_refuses_unsafe_new_name(new_name):
    ok, reason, _ = v(intent="rename_file", filename="report.txt", newname=new_name)
    assert not ok and reason == "unsafe_new_name"


def test_rename_accepts_model_key_newname(workspace):
    ok, _, p = v(intent="rename_file", filename="report.txt", newname="final.txt")
    assert ok and p["_new_path"] == str(workspace / "final.txt")


@pytest.mark.parametrize(
    "parsed",
    [
        {"filenames": "report.txt", "zipname": "../../out.zip"},
        {"filenames": "../../.ssh/id_rsa", "zipname": "keys.zip"},
        {"filenames": "a.txt, /etc/passwd", "zipname": "x.zip"},
    ],
)
def test_zip_refuses_escape(parsed):
    ok, reason, _ = v(intent="zip_files", **parsed)
    assert not ok and reason == "unsafe_path"


def test_unzip_refuses_escape():
    assert v(intent="unzip_files", zipname="../x.zip")[0] is False


def test_valid_file_operation_is_confined(workspace):
    ok, _, p = v(intent="move_file", filename="report.txt", folder="archive")
    assert ok
    assert p["_safe_path"] == str(workspace / "report.txt")
    assert p["_dst_safe"] == str(workspace / "archive")


def test_validation_has_no_side_effect(workspace):
    v(intent="create_file", filename="deep/nested/file.txt")
    assert not (workspace / "deep").exists()


# ---------- intent / app allow-lists ----------
@pytest.mark.parametrize("intent", ["rm -rf", "start_service", "exec", "run_shell", "", "SHUTDOWN_NOW"])
def test_unknown_intents_are_refused(intent):
    ok, reason, _ = v(intent=intent, service="sudo rm -rf /")
    assert not ok and reason in ("unknown_intent", "no_intent")


@pytest.mark.parametrize("app", ["powershell", "regedit", "bash", "cmd.exe", "rm"])
def test_apps_outside_allow_list_are_refused(app):
    ok, reason, _ = v(intent="open_app", app=app)
    assert not ok and reason == "unknown_app"


def test_fuzzy_app_match():
    ok, _, p = v(intent="open_app", app="Firefx")
    assert ok and p["app"] == "firefox"


@pytest.mark.parametrize("parsed", [None, "not a dict", ["intent", "open_app"], 42])
def test_non_dict_outputs_are_refused(parsed):
    assert validate_parsed(parsed) == (False, "parse_failed", None)


# ---------- URLs ----------
@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "ftp://example.com",
        "example.com & calc.exe",
        "example.com;shutdown",
        "http://exa mple.com",
        "localhost",
        "https://",
        "data:text/html,hi",
        "`whoami`.com",
        "$(id).com",
    ],
)
def test_bad_urls_are_refused(url):
    assert normalize_url(url) is None
    assert v(intent="open_url", url=url)[0] is False
    assert v(intent="open_app_and_go_url", app="chrome", url=url)[0] is False


@pytest.mark.parametrize(
    "url,expected",
    [
        ("github.com", "https://github.com"),
        ("https://arxiv.org/abs/1910.10683", "https://arxiv.org/abs/1910.10683"),
        ("http://example.com/search?q=a+b", "http://example.com/search?q=a+b"),
    ],
)
def test_good_urls_are_normalized(url, expected):
    assert normalize_url(url) == expected


# ---------- confirmation ----------
@pytest.mark.parametrize("intent", ["delete_file", "shutdown", "restart", "logout"])
def test_destructive_intents_need_confirmation(intent):
    assert needs_confirmation(intent, confidence=0.999)


def test_low_confidence_needs_confirmation():
    assert needs_confirmation("open_app", confidence=0.5)
    assert needs_confirmation("open_app", confidence=None)
    assert not needs_confirmation("open_app", confidence=0.99)
