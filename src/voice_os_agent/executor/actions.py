"""Execution of *validated* intents on the host OS (Windows / macOS / Linux).

Rules followed by every action:
- only paths produced by the validator (``_safe_path``, ``_dst_safe``, ...) are used;
- processes are started with an argument list, never through a shell;
- ``execute=False`` returns a preview and touches nothing (dry run).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from voice_os_agent import config
from voice_os_agent.safety.validator import APP_PATHS

log = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"

#: Names understood by ``open -a`` on macOS.
MACOS_APP_NAMES = {
    "firefox": "Firefox",
    "chrome": "Google Chrome",
    "edge": "Microsoft Edge",
    "vscode": "Visual Studio Code",
    "notepad": "TextEdit",
    "terminal": "Terminal",
    "calculator": "Calculator",
    "spotify": "Spotify",
}
#: Executables tried on Linux.
LINUX_APP_COMMANDS = {
    "firefox": "firefox",
    "chrome": "google-chrome",
    "edge": "microsoft-edge",
    "vscode": "code",
    "notepad": "gedit",
    "terminal": "x-terminal-emulator",
    "calculator": "gnome-calculator",
    "spotify": "spotify",
}


def _ok(**extra) -> dict:
    return {"status": "ok", **extra}


def _error(exc: Exception) -> dict:
    log.warning("Execution error: %s", exc)
    return {"status": "error", "error": str(exc)}


def search_url(query: str) -> str:
    return "https://www.google.com/search?q=" + quote_plus(query or "")


def launch_app(app_name: str, url: str | None = None) -> dict:
    """Start an allow-listed application, optionally with a validated URL."""
    app = app_name.lower()
    args = [url] if url else []
    try:
        if IS_WINDOWS:
            exe = APP_PATHS.get(app)
            if exe and os.path.exists(exe):
                subprocess.Popen([exe, *args])
            else:
                # ShellExecute via os.startfile: no cmd.exe parsing, so no "&" injection.
                os.startfile(app, arguments=" ".join(args)) if args else os.startfile(app)  # type: ignore[attr-defined]
        elif IS_MACOS:
            subprocess.Popen(["open", "-a", MACOS_APP_NAMES.get(app, app), *args])
        else:
            subprocess.Popen([LINUX_APP_COMMANDS.get(app, app), *args])
        return _ok()
    except Exception as exc:
        return _error(exc)


def open_path(path: str) -> None:
    """Open a file with the default application of the platform."""
    if IS_WINDOWS:
        os.startfile(path)  # type: ignore[attr-defined]
    elif IS_MACOS:
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _windows_volume(delta: float | None = None, mute: bool | None = None) -> None:
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    interface = AudioUtilities.GetSpeakers().Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    if delta is not None:
        current = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(min(1.0, max(0.0, current + delta)), None)
    if mute is not None:
        volume.SetMute(1 if mute else 0, None)


def _osascript(script: str) -> None:
    subprocess.run(["osascript", "-e", script], check=False)


# intent -> (preview text, callable doing the work). Each callable receives the validated dict.
def _actions(p: dict) -> dict:
    ws = config.workspace_dir()

    def take_screenshot():
        filepath = os.path.join(ws, f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png")
        if IS_WINDOWS:
            from PIL import ImageGrab

            ImageGrab.grab().save(filepath)
        else:
            subprocess.run(
                ["screencapture", "-x", filepath] if IS_MACOS else ["gnome-screenshot", "-f", filepath], check=True
            )
        return _ok(file=filepath)

    def move_or_copy(op):
        Path(p["_dst_safe"]).mkdir(parents=True, exist_ok=True)
        op(p["_safe_path"], p["_dst_safe"])
        return _ok()

    def zip_files():
        with zipfile.ZipFile(p["_zip_safe"], "w") as zf:
            for member in p["_zip_members"]:
                if os.path.exists(member):
                    zf.write(member, os.path.basename(member))
        return _ok()

    def unzip_files():
        # zipfile.extractall strips absolute paths and ".." from member names (no zip-slip).
        with zipfile.ZipFile(p["_zip_safe"]) as zf:
            zf.extractall(ws)
        return _ok()

    def volume(delta, mac_script):
        _windows_volume(delta=delta) if IS_WINDOWS else _osascript(mac_script)
        return _ok()

    def mute(flag):
        _windows_volume(mute=flag) if IS_WINDOWS else _osascript(f"set volume output muted {str(flag).lower()}")
        return _ok()

    def media_key(mac_command):
        if IS_WINDOWS:
            import pyautogui

            pyautogui.press("playpause")
        else:
            _osascript(f'tell application "Music" to {mac_command}')
        return _ok()

    def power(win_args, unix_args):
        subprocess.run(win_args if IS_WINDOWS else unix_args, check=False)
        return _ok()

    def run(fn):
        fn()
        return _ok()

    app, url = p.get("app"), p.get("url")
    return {
        "open_app": (f"open app {app}", lambda: launch_app(app)),
        "open_url": (f"open url {url}", lambda: run(lambda: webbrowser.open(url))),
        "open_app_and_go_url": (f"open {app} and go to {url}", lambda: launch_app(app, url)),
        "open_app_and_search": (
            f"open {app} and search {p.get('query')}",
            lambda: launch_app(app, url or search_url(p.get("query"))),
        ),
        "open_app_and_open_file_and_search": (
            f"open {app} and search {p.get('query')}",
            lambda: launch_app(app, url or search_url(p.get("query"))),
        ),
        "search_web": (
            f"search web for {p.get('query')}",
            lambda: run(lambda: webbrowser.open(search_url(p.get("query")))),
        ),
        "create_file": (
            f"create file {p.get('_safe_path')}",
            lambda: run(
                lambda: (
                    Path(p["_safe_path"]).parent.mkdir(parents=True, exist_ok=True),
                    open(p["_safe_path"], "a").close(),
                )
            ),
        ),
        "delete_file": (f"delete file {p.get('_safe_path')}", lambda: run(lambda: os.remove(p["_safe_path"]))),
        "open_file": (f"open file {p.get('_safe_path')}", lambda: run(lambda: open_path(p["_safe_path"]))),
        "rename_file": (
            f"rename {p.get('_safe_path')} to {p.get('_new_path')}",
            lambda: run(lambda: os.rename(p["_safe_path"], p["_new_path"])),
        ),
        "move_file": (f"move {p.get('_safe_path')} to {p.get('_dst_safe')}", lambda: move_or_copy(shutil.move)),
        "copy_file": (f"copy {p.get('_safe_path')} to {p.get('_dst_safe')}", lambda: move_or_copy(shutil.copy2)),
        "create_folder": (
            f"create folder {p.get('_folder_safe')}",
            lambda: run(lambda: os.makedirs(p["_folder_safe"], exist_ok=True)),
        ),
        "list_dir": (
            f"list directory {p.get('_folder_safe')}",
            lambda: _ok(files=sorted(os.listdir(p["_folder_safe"]))),
        ),
        "zip_files": (f"zip {p.get('_zip_members')} into {p.get('_zip_safe')}", zip_files),
        "unzip_files": (f"unzip {p.get('_zip_safe')}", unzip_files),
        "screenshot": ("take screenshot", take_screenshot),
        "volume_up": (
            "increase volume",
            lambda: volume(0.1, "set volume output volume (output volume of (get volume settings) + 10)"),
        ),
        "volume_down": (
            "decrease volume",
            lambda: volume(-0.1, "set volume output volume (output volume of (get volume settings) - 10)"),
        ),
        "mute": ("mute audio", lambda: mute(True)),
        "unmute": ("unmute audio", lambda: mute(False)),
        "shutdown": (
            "shutdown system",
            lambda: power(["shutdown", "/s", "/t", "10"], ["sudo", "shutdown", "-h", "now"]),
        ),
        "restart": ("restart system", lambda: power(["shutdown", "/r", "/t", "10"], ["sudo", "shutdown", "-r", "now"])),
        "lock_screen": (
            "lock screen",
            lambda: power(
                ["rundll32.exe", "user32.dll,LockWorkStation"],
                ["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"],
            ),
        ),
        "system_sleep": (
            "put system to sleep",
            lambda: power(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], ["pmset", "sleepnow"]),
        ),
        "play_music": ("play music", lambda: media_key("play")),
        "pause_music": ("pause music", lambda: media_key("pause")),
        "calculator": ("open calculator", lambda: launch_app("calculator")),
    }


def perform_execution(intent: str, parsed: dict, execute: bool = False, confirm: bool = False) -> dict:
    """Run a validated intent. ``execute=False`` only returns a preview."""
    intent = intent.lower()
    actions = _actions(parsed)
    if intent not in actions:
        return {"status": "unsupported_intent", "intent": intent}
    preview, action = actions[intent]
    if intent in ("delete_file", "shutdown", "restart") and not confirm:
        return {"status": "needs_confirm"}
    if not execute:
        return {"status": "simulated", "preview": preview}
    log.info("Executing %s: %s", intent, preview)
    try:
        return action()
    except Exception as exc:
        return _error(exc)
