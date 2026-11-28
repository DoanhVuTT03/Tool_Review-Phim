"""Higgs TTS 3 (Boson AI) server: sglang-omni inside WSL (Ubuntu), port 8000, the same install and scripts as
gemini_srt_translator (D:\\Download\\Model\\higgs3_*.sh / *.yaml). Voice cloning = reference wav + its words.

Speed (RTX 3060 12 GB, measured 17/11/2026 in that project): the server batches up to 48 requests per decode
step (each request has its own text, reference voice, seed and KV cache - nothing is mixed between lines), so
48 parallel requests give ~29 s of audio per second vs ~1.4 s/s one line at a time.
VRAM: the server takes ~11-12 GB -> nothing else may use the GPU meanwhile (OCR / inpaint / Whisper);
app.services.gpu_gate makes the other GPU jobs wait and stops this server before they start.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from app.services.audio_env import MODEL_ROOT

URL = "http://127.0.0.1:8000"
DISTRO = os.environ.get("TOOL_WSL_DISTRO", "Ubuntu")
SCRIPTS_WIN = MODEL_ROOT
MAIN_CONFIGS = ("higgs3_12gb.yaml", "higgs3_12gb_nhe.yaml")  # 48 lines at once; lighter fallback (32)
_STATE_FILE = MODEL_ROOT / "higgs3_dang_chay.txt"  # shared with gemini_srt_translator (same server)
_lock = threading.Lock()


def wsl_path(p: str | Path) -> str:
    p = os.path.abspath(str(p))
    return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")


def _scripts_wsl() -> str:
    return wsl_path(SCRIPTS_WIN)


def _run(*args: str, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl", "-d", DISTRO, "-u", "root", "--", "bash", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        env=dict(os.environ, MSYS_NO_PATHCONV="1"), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def installed() -> Dict[str, Any]:
    """Scripts on disk + model inside WSL (cached by the caller; WSL start-up costs ~1 s)."""
    ok_scripts = all((SCRIPTS_WIN / f).exists() for f in ("higgs3_wsl_bat_nen.sh", "higgs3_wsl_stop.sh", MAIN_CONFIGS[0]))
    ok_model = False
    if ok_scripts:
        try:
            r = _run("-c", "test -f /root/higgs3/model/config.json && test -d /root/higgs3/sglang-omni && echo OK", timeout=40)
            ok_model = "OK" in (r.stdout or "")
        except Exception:
            ok_model = False
    return {"scripts": ok_scripts, "model": ok_model, "ok": ok_scripts and ok_model}


def alive(timeout: float = 4.0) -> bool:
    """Higgs answers on port 8000 (the FishAudio S2-Pro server of the other tool uses the same port -> check id)."""
    import httpx

    try:
        d = httpx.get(URL + "/v1/models", timeout=timeout).json()
        return not any(m.get("id") == "s2-pro" for m in d.get("data", []))
    except Exception:
        return False


def running_config() -> str:
    try:
        return _STATE_FILE.read_text(encoding="utf-8").strip() if alive() else ""
    except Exception:
        return ""


def stop() -> None:
    """Stops the server (gives the ~12 GB of VRAM back)."""
    with _lock:
        try:
            _run(f"{_scripts_wsl()}/higgs3_wsl_stop.sh", timeout=90)
        except Exception:
            pass
        try:
            _STATE_FILE.unlink()
        except Exception:
            pass


def stop_if_running() -> bool:
    if alive(timeout=2.0):
        stop()
        return True
    return False


def start(log: Callable[[str], None] = print) -> str:
    """Starts the server with the 48-line config (lighter 32-line config if VRAM is short). ~1 min.
    Other WSL TTS servers of gemini_srt_translator (ZONOS3, Breeze/Irodori) are stopped first: no VRAM for both."""
    with _lock:
        cur = running_config()
        if cur in MAIN_CONFIGS and alive():
            return cur
        log("Bật máy chủ Higgs TTS 3 trong WSL (~1 phút)...")
        for s in ("zonos3_wsl_stop.sh", "tts_moi_wsl_stop.sh"):
            if (SCRIPTS_WIN / s).exists():
                try:
                    _run(f"{_scripts_wsl()}/{s}", timeout=60)
                except Exception:
                    pass
        out = ""
        for cfg in MAIN_CONFIGS:
            try:
                _run(f"{_scripts_wsl()}/higgs3_wsl_stop.sh", timeout=90)
            except Exception:
                pass
            r = _run(f"{_scripts_wsl()}/higgs3_wsl_bat_nen.sh", f"{_scripts_wsl()}/{cfg}", timeout=1200)
            out = (r.stdout or "") + (r.stderr or "")
            for _ in range(10):
                if alive():
                    break
                time.sleep(1.0)
            if alive():
                try:
                    _STATE_FILE.write_text(cfg, encoding="utf-8")
                except Exception:
                    pass
                return cfg
            log(f"Máy chủ Higgs không bật được với {cfg} → thử cấu hình nhẹ hơn...")
        raise RuntimeError("Không bật được máy chủ Higgs TTS 3 (WSL): " + out[-500:])


# ---------------------------------------------------------------------------------------------- generation
_VOWELS = r"[aeiouyàáâãèéêìíòóôõùúýăđĩũơưạ-ỹ]+"


def token_cap(text: str) -> int:
    """Max decode steps (25 steps = 1 s of audio): ~2.4x the real reading time, so a line is never cut but a
    model that starts rambling / looping stops early (same formula as the patched server)."""
    t = re.sub(r"<\|[^|]*\|>", " ", text or "")
    am = sum(max(1, len(re.findall(_VOWELS, w, re.I))) for w in re.findall(r"[^\W\d_]+", t) if re.search(r"[a-zA-Z]", w))
    other = len(re.findall(r"[^\x00-ɏḀ-ỿ\s\W]", t))
    digits = len(re.findall(r"\d", t))
    return int(min(2048, 12 * am + 14 * other + 30 * digits + 50))


def trim_silence(y: np.ndarray, sr: int) -> np.ndarray:
    """Drops silence at both ends (keeps 0.05 s before, 0.15 s after the speech, 30 ms fade): the clip length
    matches the speech so it fits its subtitle slot. The 0.15 s tail keeps the last syllable whole."""
    pk = float(np.abs(y).max()) + 1e-9
    fr = int(sr * 0.01)
    n = len(y) // fr
    if n < 5:
        return y
    db = 20 * np.log10(np.sqrt((y[: n * fr].reshape(n, fr) ** 2).mean(1)) / pk + 1e-9)
    on = np.where(db > -42)[0]
    if not len(on):
        return y
    a = max(0, on[0] * fr - int(0.05 * sr))
    b = min(len(y), (on[-1] + 1) * fr + int(0.15 * sr))
    out = y[a:b].copy()
    f = min(len(out) // 4, int(0.03 * sr))
    if f > 0:
        out[-f:] *= np.linspace(1, 0, f)
    return out


def generate(text: str, ref: Dict[str, str], seed: int, timeout: float = 180.0,
             temperature: float = 0.8, top_p: float = 0.8, top_k: int = 50) -> Tuple[np.ndarray, int, bool]:
    """One line -> (mono float32 audio, sample rate, hit_cap). hit_cap = the audio reached the length cap
    (probably cut or rambling) -> flagged by the review."""
    import httpx
    import soundfile as sf

    cap = token_cap(text)
    body = {"input": text, "references": [ref], "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "max_new_tokens": cap, "seed": int(seed) % (1 << 31), "response_format": "wav"}
    last: Optional[Exception] = None
    for k in range(3):
        try:
            r = httpx.post(URL + "/v1/audio/speech", json=body, timeout=timeout)
            r.raise_for_status()
            y, sr = sf.read(io.BytesIO(r.content), dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            hit_cap = len(y) / sr >= cap / 25.0 - 0.3
            return np.asarray(y, np.float32), int(sr), hit_cap
        except Exception as e:  # noqa: BLE001
            last = e
            if k < 2 and not alive():
                start()
            time.sleep(1.0 + k)
    raise RuntimeError(f"Higgs không tạo được câu: {last}")


def health_json() -> Dict[str, Any]:
    return {"alive": alive(timeout=2.0), "config": running_config()}


if __name__ == "__main__":  # manual check: python -m app.services.higgs_tts
    print(json.dumps(installed()))
