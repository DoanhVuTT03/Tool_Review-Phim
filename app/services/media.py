from __future__ import annotations
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_ffprobe_path
from app.models import SubtitleCue


_SRT_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    ms_val = int(ms.ljust(3, "0")[:3])
    return int(h) * 3600 + int(m) * 60 + int(s) + ms_val / 1000.0


def format_srt_time(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    h = ms_total // 3_600_000
    ms_total %= 3_600_000
    m = ms_total // 60_000
    ms_total %= 60_000
    s = ms_total // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def probe_video(video_path: str) -> Dict[str, Any]:
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe thất bại: {res.stderr.strip()}")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v_stream:
        raise RuntimeError("Không tìm thấy luồng video trong file.")

    width = int(v_stream.get("width", 1920))
    height = int(v_stream.get("height", 1080))
    # Handle rotation metadata (e.g. mobile vertical videos)
    tags = v_stream.get("tags", {}) or {}
    rotate = str(tags.get("rotate", "0"))
    if rotate in ("90", "270", "-90"):
        width, height = height, width

    fps_str = v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate") or "30/1"
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
    except Exception:
        fps = 30.0

    duration = float(data.get("format", {}).get("duration") or v_stream.get("duration") or 0.0)
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "duration": round(duration, 3),
        "has_audio": a_stream is not None,
    }


def parse_srt(srt_path: str) -> List[SubtitleCue]:
    path = Path(srt_path)
    if not path.exists():
        return []
    raw_bytes = path.read_bytes()
    text = ""
    for enc in ("utf-8-sig", "utf-8", "cp1258", "gb18030", "cp1252"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        text = raw_bytes.decode("utf-8", errors="replace")

    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    cues: List[SubtitleCue] = []
    auto_idx = 1
    for block in blocks:
        lines = [ln.rstrip("\r") for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        time_line_idx = -1
        match = None
        for i, ln in enumerate(lines[:2]):
            m = _SRT_TIME_RE.search(ln)
            if m:
                time_line_idx = i
                match = m
                break
        if match is None:
            continue
        start_sec = _to_seconds(*match.groups()[:4])
        end_sec = _to_seconds(*match.groups()[4:])
        body = "\n".join(lines[time_line_idx + 1 :]).strip()
        if not body:
            continue
        idx = auto_idx
        if time_line_idx == 1 and lines[0].strip().isdigit():
            idx = int(lines[0].strip())
        cues.append(SubtitleCue(index=idx, start_sec=start_sec, end_sec=end_sec, text=body))
        auto_idx += 1
    return cues


def write_srt(cues: List[SubtitleCue], out_path: str) -> None:
    lines: List[str] = []
    for i, c in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(c.start_sec)} --> {format_srt_time(c.end_sec)}")
        lines.append(c.text.strip())
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


_FONTS_CACHE: Optional[List[Dict[str, str]]] = None


def scan_system_fonts(force_refresh: bool = False) -> List[Dict[str, str]]:
    global _FONTS_CACHE
    if _FONTS_CACHE is not None and not force_refresh:
        return _FONTS_CACHE

    dirs = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
    ]
    seen = set()
    fonts: List[Dict[str, str]] = []
    preferred = [
        ("Arial Bold", "arialbd.ttf"),
        ("Arial", "arial.ttf"),
        ("Segoe UI Bold", "segoeuib.ttf"),
        ("Segoe UI", "segoeui.ttf"),
        ("Tahoma Bold", "tahomabd.ttf"),
        ("Tahoma", "tahoma.ttf"),
        ("Roboto / Calibri Bold", "calibrib.ttf"),
        ("Impact", "impact.ttf"),
        ("Times New Roman Bold", "timesbd.ttf"),
        ("Verdana Bold", "verdanab.ttf"),
    ]
    win_fonts = dirs[0]
    for label, fname in preferred:
        fp = win_fonts / fname
        if fp.exists():
            key = str(fp).lower()
            seen.add(key)
            fonts.append({"name": label, "path": str(fp), "filename": fp.name})

    for d in dirs:
        if not d.exists():
            continue
        for fp in sorted(d.iterdir()):
            if fp.suffix.lower() not in (".ttf", ".otf"):
                continue
            key = str(fp).lower()
            if key in seen:
                continue
            seen.add(key)
            fonts.append({"name": fp.stem, "path": str(fp), "filename": fp.name})

    _FONTS_CACHE = fonts
    return fonts


def scan_logo_folder(folder_path: str) -> List[Dict[str, str]]:
    p = Path(folder_path)
    if not folder_path or not p.exists() or not p.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    items = []
    for fp in sorted(p.iterdir()):
        if fp.is_file() and fp.suffix.lower() in exts:
            items.append({"name": fp.name, "path": str(fp)})
    return items


def scan_audio_folder(folder_path: str) -> List[Dict[str, str]]:
    p = Path(folder_path)
    if not folder_path or not p.exists() or not p.is_dir():
        return []
    exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus"}
    items = []
    for fp in sorted(p.iterdir()):
        if fp.is_file() and fp.suffix.lower() in exts:
            items.append({"name": fp.name, "path": str(fp)})
    return items


def open_native_dialog(mode: str = "file", title: str = "Chọn file", filetypes: Optional[List[tuple[str, str]]] = None) -> List[str]:
    """Opens native Windows file/folder dialog via tkinter in an isolated subprocess (safe with FastAPI async)."""
    ft_json = json.dumps(filetypes or [("All files", "*.*")])
    script = f"""
import json, tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
mode = {json.dumps(mode)}
title = {json.dumps(title)}
filetypes = [tuple(x) for x in json.loads({json.dumps(ft_json)})]
if mode == 'folder':
    res = filedialog.askdirectory(title=title)
    out = [res] if res else []
elif mode == 'files':
    res = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    out = list(res) if res else []
else:
    res = filedialog.askopenfilename(title=title, filetypes=filetypes)
    out = [res] if res else []
root.destroy()
print(json.dumps(out))
"""
    import sys

    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if res.returncode != 0 or not res.stdout.strip():
        return []
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        return []
