"""Separate Python environment for the heavy audio models (reference-voice cleaning: audio-separator Roformer
models + DNSMOS; Whisper checks: faster-whisper). These packages and their CUDA builds already live in
D:\\Download\\Model\\envs\\giong_sach (shared with the gemini_srt_translator tool), so the tool's own venv
stays light. Workers are run as child processes that print progress lines:
    @@P|<a>|<b>   progress            @@XONG|<value>   finished          @@LOI|<message>   failed
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from app.config import BASE_DIR
from app.services.cancel import check_cancel, register_proc, unregister_proc

WORKERS_DIR = BASE_DIR / "app" / "workers"
MODEL_ROOT = Path(os.environ.get("TOOL_MODEL_ROOT", r"D:\Download\Model"))
WHISPER_DIR = MODEL_ROOT / "faster_whisper"
_CANDIDATES = [
    os.environ.get("TOOL_AUDIO_PYTHON", ""),
    str(MODEL_ROOT / "envs" / "giong_sach" / "Scripts" / "python.exe"),
    str(MODEL_ROOT / "envs" / "dub" / "Scripts" / "python.exe"),
    sys.executable,
]
_CHECK: Dict[str, Tuple[bool, bool]] = {}  # python -> (has faster_whisper, has audio_separator)


def _probe(py: str) -> Tuple[bool, bool]:
    if py in _CHECK:
        return _CHECK[py]
    res = (False, False)
    if py and Path(py).exists():
        code = (
            "import importlib.util as u;"
            "print(int(bool(u.find_spec('faster_whisper'))), int(bool(u.find_spec('audio_separator'))))"
        )
        try:
            r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=60,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            a, b = (r.stdout.strip().split() + ["0", "0"])[:2]
            res = (a == "1", b == "1")
        except Exception:
            pass
    _CHECK[py] = res
    return res


def audio_python(need_separator: bool = False) -> Optional[str]:
    """First Python that has faster-whisper (and audio-separator when cleaning a reference voice)."""
    for py in _CANDIDATES:
        fw, sep = _probe(py)
        if fw and (sep or not need_separator):
            return py
    return None


def env_status() -> Dict[str, object]:
    py_check = audio_python()
    py_clean = audio_python(need_separator=True)
    return {
        "python_whisper": py_check or "",
        "python_clean": py_clean or "",
        "whisper_models": WHISPER_DIR.exists(),
        "clean_models": (MODEL_ROOT / "giong_sach_models").exists(),
    }


def run_worker(
    script: str,
    args: List[str],
    *,
    need_separator: bool = False,
    on_progress: Optional[Callable[[str, str], None]] = None,
    timeout: Optional[float] = None,
) -> Tuple[str, List[str]]:
    """Runs app/workers/<script> with the audio Python. Returns (value of @@XONG, all output lines).
    Raises RuntimeError with the worker's @@LOI message. Killed by the project's ⏹ Dừng (cancel_project)."""
    py = audio_python(need_separator)
    if not py:
        what = "audio-separator + faster-whisper" if need_separator else "faster-whisper"
        raise RuntimeError(
            f"Không tìm thấy môi trường Python có {what} (cần D:\\Download\\Model\\envs\\giong_sach, "
            "hoặc đặt biến môi trường TOOL_AUDIO_PYTHON)."
        )
    check_cancel()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", TOOL_WHISPER_DIR=str(WHISPER_DIR))
    proc = subprocess.Popen(
        [py, "-u", str(WORKERS_DIR / script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    register_proc(proc)
    lines: List[str] = []
    done_val = ""
    err = ""
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            if len(lines) > 4000:
                del lines[:2000]
            if line.startswith("@@P|"):
                parts = line.split("|", 2)
                if on_progress and len(parts) == 3:
                    on_progress(parts[1], parts[2])
            elif line.startswith("@@XONG|"):
                done_val = line.split("|", 1)[1]
            elif line.startswith("@@LOI|"):
                err = line.split("|", 1)[1]
        proc.wait(timeout=timeout)
    finally:
        unregister_proc(proc)
    check_cancel()
    if err or proc.returncode != 0:
        tail = " | ".join(l for l in lines[-6:] if l.strip())
        raise RuntimeError(err or f"worker {script} lỗi (mã {proc.returncode}): {tail[-400:]}")
    return done_val, lines
