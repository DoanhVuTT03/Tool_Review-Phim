"""Per-video stopwatch: total job time + the duration of every step (translate, voices, OCR pre-scan, render...).

Stored on VideoProject.timing (persisted in projects.json) and returned by /api/projects/{id}/status, so the UI
can show a live clock while a job runs and the final durations afterwards:

    {"title": "Chạy trọn gói A–Z", "started": 1732412345.1, "ended": null, "ok": null,
     "steps": [{"key": "translate", "label": "Dịch SRT", "start": ..., "end": ..., "note": "", "status": "done"}]}

Every call is a no-op for projects that are not in the store (tests, temporary projects).
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_LOCK = threading.Lock()


def _proj(project_id: str):
    from app.services.store import store

    return store.get_project(project_id)


def _save(project) -> None:
    from app.services.store import store

    try:
        store.upsert_project(project)
    except Exception:
        pass


def start_job(project_id: str, title: str) -> None:
    p = _proj(project_id)
    if p is None:
        return
    with _LOCK:
        p.timing = {"title": title, "started": time.time(), "ended": None, "ok": None, "steps": []}
    _save(p)


def begin_step(project_id: str, key: str, label: str, note: str = "") -> Optional[Dict[str, Any]]:
    p = _proj(project_id)
    if p is None or not p.timing:
        return None
    st = {"key": key, "label": label, "start": time.time(), "end": None, "note": note, "status": "running"}
    with _LOCK:
        p.timing.setdefault("steps", []).append(st)
    return st


def end_step(step: Optional[Dict[str, Any]], ok: bool = True, note: Optional[str] = None) -> None:
    if step is None or step.get("end") is not None:
        return
    with _LOCK:
        step["end"] = time.time()
        step["status"] = "done" if ok else "error"
        if note is not None:
            step["note"] = note


@contextmanager
def step(project_id: str, key: str, label: str, note: str = "") -> Iterator[Optional[Dict[str, Any]]]:
    """with step(pid, "render", "Xuất video"): ...  (works around awaits too)."""
    st = begin_step(project_id, key, label, note)
    try:
        yield st
    except BaseException:
        end_step(st, ok=False)
        raise
    end_step(st, ok=True)


def end_job(project_id: str, ok: bool, note: str = "") -> None:
    p = _proj(project_id)
    if p is None or not p.timing or p.timing.get("ended"):
        return
    now = time.time()
    with _LOCK:
        for st in p.timing.get("steps", []):
            if st.get("end") is None:  # e.g. stopped / failed mid-step
                st["end"] = now
                st["status"] = "done" if ok else "error"
        p.timing["ended"] = now
        p.timing["ok"] = ok
        if note:
            p.timing["note"] = note
    _save(p)


def fmt_secs(sec: float) -> str:
    sec = int(round(max(0.0, sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
