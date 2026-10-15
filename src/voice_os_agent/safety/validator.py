"""Guardrails between the model output and the operating system.

Every model output goes through :func:`validate_parsed` before execution:

1. **Intent allow-list** - only known intents are accepted (``"rm -rf"`` is not an intent).
2. **App allow-list** - applications are fuzzy-matched against a fixed list.
3. **Path sandbox** - every path (source, destination, new name, archive name,
   archive members) is resolved with :func:`resolve_in_workspace`: no absolute
   paths, no ``~``, no drive letters, no ``..`` components, and after symlink
   resolution the result must stay inside the workspace directory.
4. **URL check** - only ``http``/``https`` URLs with a real host, no whitespace
   or shell metacharacters.
5. **Confirmation** - destructive intents and low-confidence parses need an
   explicit confirmation (:func:`needs_confirmation`).

Validated, absolute paths are stored under ``_``-prefixed keys (``_safe_path``,
``_dst_safe``, ...). The executor only ever uses those keys, never raw model text.
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from voice_os_agent import config

APP_PATHS = {
    "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "vscode": os.path.expandvars(r"C:\Users\%USERNAME%\AppData\Local\Programs\Microsoft VS Code\Code.exe"),
    "notepad": r"C:\Windows\System32\notepad.exe",
}
ALLOWED_APPS = set(APP_PATHS) | {"terminal", "calculator", "spotify"}

FILE_INTENTS = {
    "create_file",
    "delete_file",
    "move_file",
    "copy_file",
    "open_file",
    "open_file_with_app",
    "rename_file",
}
APP_INTENTS = {"open_app", "open_app_and_go_url", "open_app_and_search", "open_app_and_open_file_and_search"}
FOLDER_INTENTS = {"create_folder", "list_dir", "search_files"}
SIMPLE_INTENTS = {
    "system_sleep",
    "shutdown",
    "restart",
    "lock_screen",
    "logout",
    "volume_up",
    "volume_down",
    "mute",
    "unmute",
    "play_music",
    "pause_music",
    "resume_music",
    "stop_music",
    "screenshot",
    "take_photo",
    "set_timer",
    "cancel_timer",
    "send_email",
    "translate_text",
    "dictionary_lookup",
    "calculator",
    "weather",
    "show_help",
    "whoami",
}
ALLOWED_INTENTS = (
    FILE_INTENTS
    | APP_INTENTS
    | FOLDER_INTENTS
    | SIMPLE_INTENTS
    | {
        "open_url",
        "search_web",
        "zip_files",
        "unzip_files",
    }
)
DESTRUCTIVE_INTENTS = {"delete_file", "shutdown", "restart", "logout"}

_DRIVE = re.compile(r"^[A-Za-z]:")
_URL_FORBIDDEN = re.compile(r"[\s\"'<>`|&;$\\^]")


def resolve_in_workspace(path: str, base: Path | None = None) -> Path | None:
    """Resolve a user/model-provided relative path inside the workspace.

    Returns the absolute resolved path, or ``None`` if the path is not a plain
    relative path or would escape ``base`` (``..``, absolute, ``~``, drive
    letter, UNC path, NUL byte, or a symlink pointing outside).
    """
    if not path or not isinstance(path, str) or "\x00" in path:
        return None
    path = path.strip()
    if path.startswith(("/", "\\", "~")) or _DRIVE.match(path) or os.path.isabs(path):
        return None
    parts = re.split(r"[\\/]+", path)
    if any(p == ".." for p in parts):
        return None
    base = (base or config.workspace_dir()).resolve()
    candidate = (base / Path(*[p for p in parts if p not in ("", ".")])).resolve()
    if candidate != base and base not in candidate.parents:
        return None
    return candidate


def is_plain_name(name: str) -> bool:
    """A file name without any directory component (used for rename targets)."""
    return bool(name) and not re.search(r"[\\/\x00]", name) and name not in (".", "..")


def normalize_url(url: str) -> str | None:
    """Return a safe ``http(s)`` URL, adding ``https://`` to bare domains; else ``None``."""
    url = (url or "").strip()
    if not url or _URL_FORBIDDEN.search(url):
        return None
    if "://" not in url:
        if ":" in url.split("/")[0] and not re.match(r"^[\w.-]+:\d+", url):
            return None  # "javascript:alert(1)", "file:..." without slashes, etc.
        url = "https://" + url
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or "." not in parts.hostname:
        return None
    return url


def fuzzy_match_app(app_raw: str) -> str | None:
    if not app_raw or not isinstance(app_raw, str):
        return None
    app_low = app_raw.lower().strip()
    if app_low in ALLOWED_APPS:
        return app_low
    close = difflib.get_close_matches(app_low, list(ALLOWED_APPS), n=1, cutoff=0.6)
    return close[0] if close else None


def normalize_parsed(parsed: dict | None) -> dict | None:
    if parsed is None:
        return None
    return {k: v.strip() if isinstance(v, str) else v for k, v in parsed.items()}


def validate_parsed(parsed: dict | None, base: Path | None = None) -> tuple[bool, str, dict | None]:
    """Validate a parsed model output. Returns ``(ok, reason, enriched_parsed)``."""
    if not isinstance(parsed, dict):
        return False, "parse_failed", None
    parsed = normalize_parsed(parsed)
    intent = str(parsed.get("intent") or "").lower()
    if not intent:
        return False, "no_intent", parsed
    if intent not in ALLOWED_INTENTS:
        return False, "unknown_intent", parsed
    parsed["intent"] = intent

    if intent in FILE_INTENTS:
        fname = parsed.get("filename") or parsed.get("path") or ""
        if not fname:
            return False, "no_filename", parsed
        safe = resolve_in_workspace(fname, base)
        if safe is None:
            return False, "unsafe_path", parsed
        parsed["_safe_path"] = str(safe)
        if intent in ("move_file", "copy_file"):
            dst = parsed.get("folder") or parsed.get("dst") or ""
            dst_safe = resolve_in_workspace(dst, base) if dst else (base or config.workspace_dir()).resolve()
            if dst_safe is None:
                return False, "unsafe_destination", parsed
            parsed["_dst_safe"] = str(dst_safe)
        if intent == "rename_file":
            new_name = parsed.get("newname") or parsed.get("new_name") or parsed.get("new_filename") or ""
            if not is_plain_name(new_name):
                return False, "unsafe_new_name", parsed
            parsed["_new_path"] = str(safe.parent / new_name)
        if intent == "open_file_with_app" and parsed.get("app"):
            matched = fuzzy_match_app(parsed["app"])
            if not matched:
                return False, "unknown_app", parsed
            parsed["app"] = matched
        return True, "ok", parsed

    if intent in APP_INTENTS:
        matched = fuzzy_match_app(parsed.get("app") or "")
        if not matched:
            return False, "unknown_app", parsed
        parsed["app"] = matched
        if parsed.get("url"):
            url = normalize_url(parsed["url"])
            if url is None:
                return False, "bad_url", parsed
            parsed["url"] = url
        return True, "ok", parsed

    if intent == "open_url":
        if not parsed.get("url"):
            return False, "no_url", parsed
        url = normalize_url(parsed["url"])
        if url is None:
            return False, "bad_url", parsed
        parsed["url"] = url
        return True, "ok", parsed

    if intent == "search_web":
        if not parsed.get("query"):
            return False, "no_query", parsed
        return True, "ok", parsed

    if intent in FOLDER_INTENTS:
        folder = parsed.get("folder") or ""
        if not folder:
            return False, "no_folder", parsed
        safe = resolve_in_workspace(folder, base)
        if safe is None:
            return False, "unsafe_path", parsed
        parsed["_folder_safe"] = str(safe)
        return True, "ok", parsed

    if intent == "zip_files":
        names = parsed.get("filenames") or parsed.get("filenames_list") or ""
        if not names:
            return False, "no_filenames", parsed
        members = []
        for name in str(names).split(","):
            safe = resolve_in_workspace(name.strip(), base)
            if safe is None:
                return False, "unsafe_path", parsed
            members.append(str(safe))
        zip_safe = resolve_in_workspace(parsed.get("zipname") or "archive.zip", base)
        if zip_safe is None:
            return False, "unsafe_path", parsed
        parsed["_zip_members"] = members
        parsed["_zip_safe"] = str(zip_safe)
        return True, "ok", parsed

    if intent == "unzip_files":
        if not parsed.get("zipname"):
            return False, "no_zipname", parsed
        zip_safe = resolve_in_workspace(parsed["zipname"], base)
        if zip_safe is None:
            return False, "unsafe_path", parsed
        parsed["_zip_safe"] = str(zip_safe)
        return True, "ok", parsed

    return True, "ok", parsed  # SIMPLE_INTENTS: no arguments reach the OS


def needs_confirmation(intent: str, confidence: float | None, threshold: float | None = None) -> bool:
    """Destructive intents and low-confidence parses require explicit confirmation."""
    threshold = config.CONF_THRESHOLD if threshold is None else threshold
    if confidence is None or confidence < threshold:
        return True
    return intent in DESTRUCTIVE_INTENTS
