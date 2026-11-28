"""Tool-Edit Review Phim Studio - hidden launcher (no console window).

Run with pythonw.exe (run.bat / the desktop shortcut do this):
  - starts the FastAPI server in this process (no cmd window),
  - opens the UI as an app window (Edge / Chrome),
  - if the tool is already running, only opens another UI window,
  - stops the server some minutes after the UI window is closed, but never while a job is running.
Server output goes to data/logs/server.log. For a visible console with auto-reload use run_debug.bat.
"""
from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

HOST, PORT = "127.0.0.1", 8765
URL = f"http://{HOST}:{PORT}"
IDLE_SHUTDOWN_SEC = 300      # browsers throttle timers of minimised windows, so keep this generous
NO_WINDOW_SHUTDOWN_SEC = 900  # UI never connected (e.g. browser failed to open)

# pythonw has no stdout/stderr: send everything (uvicorn logs, prints, tracebacks) to a log file.
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_log_path = os.path.join(LOG_DIR, "server.log")
try:
    if os.path.exists(_log_path) and os.path.getsize(_log_path) > 5 * 1024 * 1024:
        os.replace(_log_path, _log_path + ".old")
except OSError:
    pass
_log = open(_log_path, "a", encoding="utf-8", buffering=1)
sys.stdout = _log
sys.stderr = _log


def log(msg: str) -> None:
    print(f"[launcher {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def port_open(timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=timeout):
            return True
    except OSError:
        return False


def open_ui_window() -> None:
    pf, pf86, local = os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", ""), os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(pf86, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(pf86, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(local, "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for exe in candidates:
        if exe and os.path.exists(exe):
            subprocess.Popen(
                [exe, f"--app={URL}", "--window-size=1440,900"],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                close_fds=True,
            )
            log(f"UI opened with {exe}")
            return
    webbrowser.open(URL)
    log("UI opened in default browser")


def wait_then_open_ui(max_wait: float = 90.0) -> None:
    t_end = time.time() + max_wait
    while time.time() < t_end:
        if port_open():
            open_ui_window()
            return
        time.sleep(0.3)
    log("server did not come up in time; see errors above")


def already_running_mutex() -> bool:
    """Named mutex = one server per machine even if the launcher is double-clicked twice quickly."""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        handle = kernel32.CreateMutexW(None, False, "Local\\ToolEditReviewPhimStudio")
        err = ctypes.get_last_error()
        globals()["_MUTEX_HANDLE"] = handle  # keep it alive for the process lifetime
        return err == 183  # ERROR_ALREADY_EXISTS
    except Exception:
        return False


def code_mtime() -> float:
    """Newest modification time of the server code (app/**/*.py)."""
    newest = 0.0
    for root, _dirs, files in os.walk(os.path.join(BASE_DIR, "app")):
        for f in files:
            if f.endswith(".py"):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
    return newest


def replace_outdated_server() -> bool:
    """A running server keeps the code it was started with: double-clicking run.bat only opened another window,
    so code updates (bug fixes) were silently not applied. If the server is idle and older than the code on
    disk, ask it to shut down and return True (the caller then starts a fresh one)."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{URL}/api/system/info", timeout=3) as r:
            info = json.loads(r.read().decode("utf-8"))
    except Exception:
        return False  # very old server without /api/system/info, or not responding
    if info.get("busy") or float(info.get("started") or 0) >= code_mtime():
        return False
    try:
        req = urllib.request.Request(f"{URL}/api/system/shutdown", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=3).read()
    except Exception:
        return False
    for _ in range(60):  # up to 30 s for the old server to release the port
        if not port_open():
            time.sleep(1.0)  # let the old process exit (it owns the single-instance mutex)
            return True
        time.sleep(0.5)
    return False


def main() -> None:
    if already_running_mutex() or port_open():
        if port_open() and replace_outdated_server():
            log("server ran older code than on disk and was idle -> restarted it")
        else:
            log("tool already running -> opening another UI window")
            wait_then_open_ui()
            return

    import uvicorn
    from app import main as app_main

    config = uvicorn.Config(app_main.app, host=HOST, port=PORT, log_level="info")
    server = uvicorn.Server(config)
    started = time.time()

    def watchdog() -> None:
        while not server.should_exit:
            time.sleep(15)
            try:
                idle = app_main.seconds_since_heartbeat()
                busy = app_main.has_running_jobs()
            except Exception as e:  # noqa: BLE001
                log(f"watchdog error: {e}")
                continue
            if busy:
                continue
            if idle is not None and idle > IDLE_SHUTDOWN_SEC:
                log(f"UI window closed for {int(idle)}s and no job running -> shutting down")
                server.should_exit = True
            elif idle is None and time.time() - started > NO_WINDOW_SHUTDOWN_SEC:
                log("UI never connected -> shutting down")
                server.should_exit = True

    def shutdown_watch() -> None:
        while not server.should_exit:
            time.sleep(0.5)
            try:
                if app_main.shutdown_requested():
                    log("shutdown requested by a newer launch (code updated)")
                    server.should_exit = True
            except Exception:
                pass

    threading.Thread(target=wait_then_open_ui, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=shutdown_watch, daemon=True).start()
    log(f"starting server on {URL}")
    server.run()
    log("server stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback

        log("FATAL:\n" + traceback.format_exc())
        try:
            ctypes.windll.user32.MessageBoxW(
                None,
                f"Tool-Edit không khởi động được.\nXem chi tiết lỗi trong:\n{_log_path}",
                "Tool-Edit Review Phim Studio",
                0x10,
            )
        except Exception:
            pass
