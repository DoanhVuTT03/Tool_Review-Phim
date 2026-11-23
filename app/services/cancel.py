"""Per-project cancellation: flags + tracking of child processes (agentapi, agy, ffmpeg) so "Dừng"
can kill them immediately instead of waiting for a long LLM call / render to finish.

The current project id travels through a ContextVar: set it once at the start of a job's asyncio task;
asyncio.create_task and asyncio.to_thread copy the context, so worker threads see it too.
"""
from __future__ import annotations

import contextvars
import subprocess
import threading
from typing import Dict, Optional, Set

current_project: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_project", default=None)

_lock = threading.Lock()
_cancel_flags: Dict[str, threading.Event] = {}
_procs: Dict[str, Set[subprocess.Popen]] = {}


class JobCancelled(Exception):
    """Raised inside a job when the user pressed Dừng."""


def _event(pid: str) -> threading.Event:
    with _lock:
        ev = _cancel_flags.get(pid)
        if ev is None:
            ev = _cancel_flags[pid] = threading.Event()
        return ev


def begin_job(pid: str) -> None:
    """Call at the start of a job task: binds the project id to this context and clears old cancel flag."""
    current_project.set(pid)
    _event(pid).clear()


def is_cancelled(pid: Optional[str] = None) -> bool:
    pid = pid or current_project.get()
    return bool(pid) and _event(pid).is_set()


def check_cancel() -> None:
    if is_cancelled():
        raise JobCancelled("Đã dừng tiến trình theo yêu cầu.")


def register_proc(proc: subprocess.Popen) -> None:
    pid = current_project.get()
    if not pid:
        return
    with _lock:
        _procs.setdefault(pid, set()).add(proc)
    if _event(pid).is_set():
        _kill(proc)


def unregister_proc(proc: subprocess.Popen) -> None:
    pid = current_project.get()
    if not pid:
        return
    with _lock:
        _procs.get(pid, set()).discard(proc)


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        # /T kills the whole tree (agentapi.bat -> language_server, ffmpeg children...)
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def cancel_project(pid: str) -> int:
    """Sets the cancel flag and kills all tracked child processes. Returns number of processes killed."""
    _event(pid).set()
    with _lock:
        procs = list(_procs.get(pid, set()))
    for p in procs:
        _kill(p)
    return len(procs)


def run_tracked(cmd, *, input: Optional[str] = None, timeout: Optional[float] = None, env=None) -> subprocess.CompletedProcess:
    """subprocess.run replacement that can be killed by cancel_project()."""
    check_cancel()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    register_proc(proc)
    try:
        out, err = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(proc)
        proc.communicate()
        raise
    finally:
        unregister_proc(proc)
    check_cancel()
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
