from __future__ import annotations
import asyncio
import threading
import mimetypes
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    ANTIGRAVITY_MODELS,
    GEM_TXT_PATH,
    OUTPUTS_DIR,
    PROJECTS_DIR,
    WEB_DIR,
    WORLD_LANGUAGES,
)
from app.models import (
    ChannelPreset,
    DownloadVideoRequest,
    EditConfig,
    RenderRequest,
    VideoProject,
    VoiceConfig,
)
from app.services.audio_assembler import assemble_project_audio_sync
from app.services.cancel import begin_job, cancel_project, is_cancelled
from app.services.downloader import (
    clean_srt_rules_sync,
    clean_srt_with_antigravity,
    download_and_clean_video,
    format_download_error,
    strip_ansi,
)
from app.services.gpu_gate import gpu_gate
from app.services.hardware import detect_hardware
from app.services import timing
from app.services.media import (
    open_native_dialog,
    parse_srt,
    probe_video,
    scan_audio_folder,
    scan_logo_folder,
    scan_system_fonts,
)
from app.services.pipeline import pipeline_manager
from app.services.render_engine import render_project_async
from app.services.store import store
from app.services.sub_remover import clean_hardsub_video_sync
from app.services.translator import (
    analyze_style_guide,
    detect_translation_backends,
    load_translate_checkpoint,
    translate_project_srt,
)
from app.services.tts_engine import generate_project_voices, list_all_voices, synthesize_line


app = FastAPI(title="Tool-Edit Review Phim Studio", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

_ACTIVE_TASKS: Dict[str, asyncio.Task] = {}
_DOWNLOAD_JOBS: Dict[str, Dict[str, Any]] = {}


def _ensure_idle(project_id: str) -> None:
    """One job per video at a time: a 2nd click used to start a 2nd task on the same files (two translations
    writing the same checkpoint, two voice jobs writing the same cache...)."""
    t = _ACTIVE_TASKS.get(project_id)
    if (t is not None and not t.done()) or project_id in pipeline_manager.running_project_ids():
        raise HTTPException(
            status_code=409,
            detail="Video này đang có tiến trình chạy (dịch / voice / xuất...). Chờ xong hoặc bấm ⏹ Dừng trước.",
        )


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = WEB_DIR / "index.html"
    html = index_file.read_text(encoding="utf-8")
    # cache-busting: the browser kept a stale app.js after an update (new HTML + old JS = new controls dead)
    for name in ("app.js", "styles.css"):
        f = WEB_DIR / name
        if f.exists():
            html = html.replace(f"/static/{name}\"", f"/static/{name}?v={int(f.stat().st_mtime)}\"")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


# ---------------- Hardware & System ----------------

# Heartbeat from the UI window: the hidden launcher (launcher.pyw) stops the server some time after the
# window is closed, but never while a job is still running.
_LAST_HEARTBEAT: Dict[str, float] = {"t": 0.0}
_SERVER_STARTED = time.time()
_SHUTDOWN_REQUESTED = {"v": False}


def shutdown_requested() -> bool:
    """Polled by launcher.pyw: lets a newer launch replace an idle server that still runs old code."""
    return _SHUTDOWN_REQUESTED["v"]


def has_running_jobs() -> bool:
    if any(not t.done() for t in _ACTIVE_TASKS.values()):
        return True
    if pipeline_manager.status().get("running"):
        return True
    return any(j.get("status") == "downloading" for j in _DOWNLOAD_JOBS.values())


def seconds_since_heartbeat() -> Optional[float]:
    t = _LAST_HEARTBEAT["t"]
    return None if t == 0.0 else time.time() - t


@app.post("/api/system/heartbeat")
async def system_heartbeat():
    _LAST_HEARTBEAT["t"] = time.time()
    return {"ok": True, "busy": has_running_jobs()}


@app.get("/api/system/info")
async def system_info():
    return {"started": _SERVER_STARTED, "busy": has_running_jobs(), "pid": os.getpid()}


@app.post("/api/system/shutdown")
async def system_shutdown():
    """Only when idle: used by launcher.pyw to restart a server that runs older code than what is on disk."""
    if has_running_jobs():
        raise HTTPException(status_code=409, detail="Đang có tiến trình chạy, không tắt server.")
    _SHUTDOWN_REQUESTED["v"] = True
    return {"ok": True}


@app.get("/api/hardware")
async def get_hardware(refresh: bool = False):
    return await asyncio.to_thread(detect_hardware, refresh)


@app.get("/api/config/languages")
async def get_languages():
    return {"languages": WORLD_LANGUAGES}


@app.get("/api/config/models")
async def get_models():
    """Model list for the dropdowns: live from the running Antigravity app (so new models appear and removed
    ones disappear), the static list as a fallback when the app is closed."""
    from app.services.translator import fetch_antigravity_models

    cat = await asyncio.to_thread(fetch_antigravity_models)
    if not cat:
        return {"models": ANTIGRAVITY_MODELS, "live": False}
    models = []
    for f in cat["families"]:
        tag = f" — {f['tag']}" if f.get("tag") and f["tag"] != "New" else (" (New)" if f.get("tag") == "New" else "")
        lv = f"  [{'/'.join(x.capitalize() for x in f['levels'])}]" if f["levels"] else ""
        models.append({"id": f["id"], "label": f"{f['label']}{tag}{lv}", "levels": f["levels"]})
    return {"models": models, "live": True}


@app.get("/api/config/resolve-model")
async def resolve_model(model: str = "", thinking: str = "medium"):
    """What Antigravity will actually run for this model + thinking level (shown under the dropdowns)."""
    from app.services.translator import resolve_agentapi_model

    mid, note = await asyncio.to_thread(resolve_agentapi_model, model, thinking)
    return {"model_id": mid, "note": note}


@app.get("/api/config/fonts")
async def get_fonts():
    return {"fonts": await asyncio.to_thread(scan_system_fonts)}


@app.get("/api/config/gem")
async def get_gem_prompt():
    text = GEM_TXT_PATH.read_text(encoding="utf-8") if GEM_TXT_PATH.exists() else ""
    return {"path": str(GEM_TXT_PATH), "content": text}


class PickRequest(BaseModel):
    mode: str = "file"
    kind: str = "any"


@app.post("/api/system/pick")
async def pick_file(req: PickRequest):
    ft_map = {
        "video": [("Video Files", "*.mp4 *.mkv *.mov *.avi *.webm *.ts"), ("All Files", "*.*")],
        "srt": [("Subtitle Files", "*.srt *.ass *.vtt"), ("All Files", "*.*")],
        "image": [("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All Files", "*.*")],
        "audio": [("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.opus"), ("All Files", "*.*")],
        "font": [("Font Files", "*.ttf *.otf"), ("All Files", "*.*")],
        "any": [("All Files", "*.*")],
    }
    title_map = {
        "video": "Chọn file Video",
        "srt": "Chọn file Phụ đề (.srt)",
        "image": "Chọn file Logo",
        "audio": "Chọn file Nhạc nền (có thể chọn nhiều file)",
        "font": "Chọn file Font (.ttf / .otf)",
        "any": "Chọn file / thư mục",
    }
    paths = await asyncio.to_thread(
        open_native_dialog,
        req.mode,
        title_map.get(req.kind, "Chọn file"),
        ft_map.get(req.kind, ft_map["any"]),
    )
    return {"paths": paths}


@app.get("/api/system/logos")
async def list_logos(folder: str = Query("")):
    return {"logos": scan_logo_folder(folder)}


@app.get("/api/system/audio-folder")
async def list_audio_folder(folder: str = Query("")):
    return {"tracks": scan_audio_folder(folder)}


class OpenFolderRequest(BaseModel):
    path: str = ""


@app.post("/api/system/open-folder")
async def open_folder(req: OpenFolderRequest):
    target = Path(req.path) if req.path else OUTPUTS_DIR
    if target.is_file():
        subprocess.Popen(["explorer", "/select,", str(target)])
    else:
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(str(target))
    return {"ok": True}


# ---------------- Local Media Streaming ----------------


@app.get("/api/media/stream")
async def stream_local_media(request: Request, path: str = Query(...)):
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file media.")

    file_size = file_path.stat().st_size
    mime, _ = mimetypes.guess_type(str(file_path))
    content_type = mime or "application/octet-stream"

    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(str(file_path), media_type=content_type)

    try:
        range_spec = range_header.replace("bytes=", "").strip()
        start_s, end_s = range_spec.split("-", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else min(start + 2 * 1024 * 1024 - 1, file_size - 1)
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            raise HTTPException(status_code=416, detail="Range không hợp lệ")
    except ValueError:
        return FileResponse(str(file_path), media_type=content_type)

    chunk_len = end - start + 1

    def iter_file():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = chunk_len
            while remaining > 0:
                data = f.read(min(65536, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_len),
    }
    return StreamingResponse(iter_file(), status_code=206, headers=headers, media_type=content_type)


# ---------------- YouTube / BiliBili Multi-Threaded Downloader + SRT Cleaner ----------------


_DOWNLOAD_CANCEL: Dict[str, threading.Event] = {}


@app.post("/api/download")
async def start_download(req: DownloadVideoRequest):
    running = [j for j in _DOWNLOAD_JOBS.values() if j["status"] == "downloading"]
    if running:
        # one download at a time: a 2nd one only splits the same ~100 Mbps line
        raise HTTPException(status_code=409, detail="Đang có 1 video đang tải. Chờ tải xong hoặc bấm Dừng tải trước.")
    job_id = uuid.uuid4().hex[:10]
    _DOWNLOAD_CANCEL[job_id] = threading.Event()
    _DOWNLOAD_JOBS[job_id] = {
        "job_id": job_id,
        "status": "downloading",
        "progress": 2.0,
        "message": "Đang phân tích link video...",
        "project": None,
        "error": "",
        "url": req.url,
        "started": time.time(),
        "ended": None,
    }

    def _on_prog(pct: float, msg: str):
        _DOWNLOAD_JOBS[job_id]["progress"] = pct
        _DOWNLOAD_JOBS[job_id]["message"] = msg

    async def _runner():
        try:
            res = await download_and_clean_video(req, on_progress=_on_prog, cancel=_DOWNLOAD_CANCEL[job_id])
            vp = Path(res["video_path"])
            _on_prog(99.5, "Đang đọc thông tin video & thêm vào danh sách...")
            meta = await asyncio.to_thread(probe_video, str(vp))

            cfg = EditConfig()
            if req.preset_id and req.preset_id in store.presets:
                cfg = store.presets[req.preset_id].config.model_copy(deep=True)
            elif store.presets:
                first_preset = next(iter(store.presets.values()))
                cfg = first_preset.config.model_copy(deep=True)

            proj = VideoProject(
                id=uuid.uuid4().hex[:10],
                name=vp.name,
                video_path=str(vp),
                srt_path=res.get("srt_path", ""),
                width=meta["width"],
                height=meta["height"],
                fps=meta["fps"],
                duration=meta["duration"],
                has_audio=meta["has_audio"],
                preset_id=req.preset_id,
                config=cfg,
                status_message=f"Đã tải ({res['threads']} luồng): {res.get('subtitle_source', '')}",
            )
            store.upsert_project(proj)
            _DOWNLOAD_JOBS[job_id]["status"] = "done"
            _DOWNLOAD_JOBS[job_id]["progress"] = 100.0
            _DOWNLOAD_JOBS[job_id]["ended"] = time.time()
            took = _DOWNLOAD_JOBS[job_id]["ended"] - _DOWNLOAD_JOBS[job_id]["started"]
            _DOWNLOAD_JOBS[job_id]["message"] = (
                f"Tải xong trong {timing.fmt_secs(took)}! Đã thêm '{vp.name}' vào danh sách — "
                "giờ chỉnh cấu hình rồi bấm Chạy A–Z (dịch → voice → xóa sub → xuất)."
            )
            _DOWNLOAD_JOBS[job_id]["project"] = proj.model_dump()
            _DOWNLOAD_JOBS[job_id]["clean_stats"] = res.get("clean_stats", {})
        except Exception as e:
            clean_err = format_download_error(e)
            stopped = _DOWNLOAD_CANCEL[job_id].is_set()
            _DOWNLOAD_JOBS[job_id]["status"] = "stopped" if stopped else "error"
            _DOWNLOAD_JOBS[job_id]["error"] = "" if stopped else clean_err
            _DOWNLOAD_JOBS[job_id]["ended"] = time.time()
            _DOWNLOAD_JOBS[job_id]["message"] = (
                "⏹ Đã dừng tải. Bấm Tải lại để tải tiếp (phần đã tải được giữ lại)." if stopped else f"Lỗi tải video: {clean_err}"
            )

    asyncio.create_task(_runner())
    return _DOWNLOAD_JOBS[job_id]


@app.get("/api/download/{job_id}/status")
async def get_download_status(job_id: str):
    if job_id not in _DOWNLOAD_JOBS:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ tải.")
    return {**_DOWNLOAD_JOBS[job_id], "now": time.time()}


@app.get("/api/download/active")
async def get_active_download():
    """The running download (if any), so reopening the dialog re-attaches to it instead of looking idle."""
    for j in reversed(list(_DOWNLOAD_JOBS.values())):
        if j["status"] == "downloading":
            return {"job": {**j, "now": time.time()}}
    return {"job": None}


@app.post("/api/download/{job_id}/cancel")
async def cancel_download(job_id: str):
    ev = _DOWNLOAD_CANCEL.get(job_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ tải.")
    ev.set()  # checked by the yt-dlp progress hooks and between AI cleaning batches
    if job_id in _DOWNLOAD_JOBS and _DOWNLOAD_JOBS[job_id]["status"] == "downloading":
        _DOWNLOAD_JOBS[job_id]["message"] = "Đang dừng tải..."
    return {"ok": True}


class CleanSrtRequest(BaseModel):
    use_ai: bool = True
    model: str = "gemini-3.8-flash"
    thinking_level: str = "low"
    api_key: str = ""
    batch_lines: int = 0  # lines per AI request, 0 = automatic (as many as fit in one request)


@app.post("/api/projects/{project_id}/clean-srt")
async def run_clean_project_srt(project_id: str, req: CleanSrtRequest):
    proj = store.get_project(project_id)
    if not proj or not proj.srt_path or not Path(proj.srt_path).exists():
        raise HTTPException(status_code=400, detail="Video chưa có file SRT gốc để dọn rác.")
    _ensure_idle(project_id)

    proj.status = "translating"
    proj.progress = 5.0
    proj.status_message = "Đang dọn rác file SRT gốc..."
    store.upsert_project(proj)

    def _on_prog(pct: float, msg: str):
        if is_cancelled(project_id):
            return
        p = store.get_project(project_id)
        if p:
            p.progress = pct
            p.status_message = msg

    async def _task():
        begin_job(project_id)
        timing.start_job(project_id, "Dọn rác SRT" + (" bằng AI" if req.use_ai else " (lọc nhanh)"))
        try:
            # Write to <stem>_clean.srt: "Lọc nhanh" used to overwrite the user's own SRT file in place.
            src = Path(proj.srt_path)
            base = src.stem[: -len("_clean")] if src.stem.endswith("_clean") else src.stem
            clean_path = src.with_name(f"{base}_clean.srt")
            with timing.step(project_id, "clean_rules", "Lọc rác nhanh (HTML, dòng trùng, [Nhạc])"):
                _, r_stats = await asyncio.to_thread(clean_srt_rules_sync, str(src), str(clean_path))
            removed = r_stats["removed_count"]
            fixed = 0
            final_srt = str(clean_path)
            if req.use_ai:
                with timing.step(project_id, "clean_ai", f"Antigravity dọn rác & sửa lỗi ({req.model})"):
                    out_p, ai_stats = await clean_srt_with_antigravity(
                        str(clean_path),
                        model=req.model,
                        thinking_level=req.thinking_level,
                        api_key=req.api_key or proj.config.translate.api_key,
                        on_progress=_on_prog,
                        batch_lines=req.batch_lines,
                    )
                final_srt = str(out_p)
                removed += ai_stats["removed_count"]
                fixed = ai_stats.get("fixed_count", 0)
            p = store.get_project(project_id)
            if p:
                p.srt_path = final_srt
                p.status = "ready"
                p.progress = 100.0
                p.status_message = (
                    f"Dọn rác SRT xong! Xóa {removed} dòng rác/trùng" + (f", AI sửa {fixed} dòng" if req.use_ai else "")
                    + f" → {Path(final_srt).name} (file gốc giữ nguyên)."
                )
                store.upsert_project(p)
            timing.end_job(project_id, True)
        except Exception as e:
            p = store.get_project(project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi dọn rác SRT: {e}"
                store.upsert_project(p)
            timing.end_job(project_id, False, "Lỗi")

    _ACTIVE_TASKS[project_id] = asyncio.create_task(_task())
    return {"ok": True, "project_id": project_id}


# ---------------- Channel Presets ----------------


@app.get("/api/presets")
async def get_presets():
    return {"presets": [p.model_dump() for p in store.list_presets()]}


class CreatePresetRequest(BaseModel):
    id: Optional[str] = None
    name: str
    config: EditConfig


@app.post("/api/presets")
async def save_preset(req: CreatePresetRequest):
    pid = req.id or f"preset_{uuid.uuid4().hex[:8]}"
    preset = ChannelPreset(id=pid, name=req.name.strip() or "Preset Mới", config=req.config)
    store.save_preset(preset)
    return {"preset": preset.model_dump()}


@app.delete("/api/presets/{preset_id}")
async def delete_preset(preset_id: str):
    if not store.delete_preset(preset_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy preset")
    return {"ok": True}


# ---------------- Video Projects ----------------


@app.get("/api/projects")
async def list_projects():
    return {"projects": [p.model_dump() for p in store.list_projects()]}


class AddProjectRequest(BaseModel):
    video_path: str
    srt_path: str = ""
    preset_id: Optional[str] = None


@app.post("/api/projects")
async def add_project(req: AddProjectRequest):
    vp = Path(req.video_path)
    if not vp.exists():
        raise HTTPException(status_code=400, detail=f"Không tìm thấy video: {req.video_path}")

    meta = await asyncio.to_thread(probe_video, str(vp))
    srt_path = req.srt_path
    if not srt_path:
        candidate = vp.with_suffix(".srt")
        if candidate.exists():
            srt_path = str(candidate)

    cfg = EditConfig()
    if req.preset_id and req.preset_id in store.presets:
        cfg = store.presets[req.preset_id].config.model_copy(deep=True)
    elif store.presets:
        first_preset = next(iter(store.presets.values()))
        cfg = first_preset.config.model_copy(deep=True)

    proj = VideoProject(
        id=uuid.uuid4().hex[:10],
        name=vp.name,
        video_path=str(vp),
        srt_path=srt_path,
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        duration=meta["duration"],
        has_audio=meta["has_audio"],
        preset_id=req.preset_id,
        config=cfg,
    )
    store.upsert_project(proj)
    return {"project": proj.model_dump()}


class UpdateProjectRequest(BaseModel):
    srt_path: Optional[str] = None
    translated_srt_path: Optional[str] = None
    preset_id: Optional[str] = None
    config: Optional[EditConfig] = None


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, req: UpdateProjectRequest):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    if req.srt_path is not None:
        proj.srt_path = req.srt_path
    if req.translated_srt_path is not None:
        proj.translated_srt_path = req.translated_srt_path
    if req.preset_id is not None:
        proj.preset_id = req.preset_id
    if req.config is not None:
        proj.config = req.config
    store.upsert_project(proj)
    return {"project": proj.model_dump()}


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    if store.get_project(project_id) is not None:
        await _stop_project(project_id)  # otherwise its task kept running (ffmpeg / LLM calls) on a deleted video
    if not store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    return {"ok": True}


@app.get("/api/projects/{project_id}/subtitles")
async def get_project_subtitles(project_id: str):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    srt_file = proj.translated_srt_path or proj.srt_path
    cues = parse_srt(srt_file) if srt_file else []
    return {"srt_path": srt_file, "count": len(cues), "cues": [c.model_dump() for c in cues]}


# ---------------- Phase 2: Antigravity SRT Translation ----------------


@app.get("/api/translate/backends")
async def get_translate_backends(agy_path: str = ""):
    return detect_translation_backends(agy_path)


# ---- Gemini Web (gemini.google.com): one request at a time, sent by the user or by the tool's Chrome ----
@app.get("/api/gemini-web/pending")
async def gemini_web_pending():
    from app.services import gemini_web

    return {"pending": gemini_web.pending(), "history": gemini_web.history()[:5]}


@app.get("/api/gemini-web/prompt/{req_id}")
async def gemini_web_prompt(req_id: str):
    from app.services import gemini_web

    txt = gemini_web.prompt_text(req_id)
    if txt is None:
        raise HTTPException(status_code=404, detail="Yêu cầu này không còn chờ.")
    return Response(content=txt, media_type="text/plain; charset=utf-8")


@app.get("/api/gemini-web/attachment/{req_id}")
async def gemini_web_attachment(req_id: str, txt: bool = False):
    """The file to attach in Gemini with this request (the whole .srt for the style guide). txt=1: same content
    named .txt, for when Gemini refuses the .srt extension."""
    from app.services import gemini_web

    path = gemini_web.attachment_path(req_id)
    if not path:
        raise HTTPException(status_code=404, detail="Yêu cầu này không có file đính kèm hoặc không còn chờ.")
    name = Path(path).name + (".txt" if txt else "")
    return FileResponse(path, filename=name, media_type="text/plain; charset=utf-8")


class GeminiWebSubmit(BaseModel):
    id: str
    text: str
    accept_partial: bool = False


@app.post("/api/gemini-web/submit")
async def gemini_web_submit(req: GeminiWebSubmit):
    from app.services import gemini_web

    return gemini_web.submit(req.id, req.text, req.accept_partial)


@app.post("/api/gemini-web/browser/open")
async def gemini_web_open_browser():
    from app.services import gemini_web_browser as gb

    try:
        return await asyncio.to_thread(gb.launch_browser)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/gemini-web/browser/status")
async def gemini_web_browser_status():
    from app.services import gemini_web_browser as gb

    return await gb.status()


@app.delete("/api/projects/{project_id}/translate-checkpoint")
async def delete_translate_checkpoint(project_id: str, style_guide: bool = False):
    """'Xóa tiến trình dịch': forgets the lines translated so far (next run starts at line 1) WITHOUT starting a
    new translation. The style guide is kept unless style_guide=true; the translated .srt already written stays."""
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    _ensure_idle(project_id)
    cp = load_translate_checkpoint(proj)
    done = len(cp.get("translated", {}))
    cp_file = PROJECTS_DIR / proj.id / "translate_state.json"
    if cp_file.exists():
        cp_file.unlink()
    sg_removed = False
    if style_guide and proj.style_guide_path:
        sg = Path(proj.style_guide_path)
        if sg.exists():
            sg.unlink()
        proj.style_guide_path = ""
        sg_removed = True
    if proj.status == "error":
        proj.status = "ready"
    proj.progress = 0.0
    proj.status_message = f"Đã xóa tiến trình dịch ({done} dòng)" + (" và Style Guide" if sg_removed else "") + "."
    store.upsert_project(proj)
    return {"ok": True, "removed_lines": done, "style_guide_removed": sg_removed}


@app.get("/api/projects/{project_id}/style-guide")
async def get_style_guide(project_id: str):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    content = ""
    if proj.style_guide_path and Path(proj.style_guide_path).exists():
        content = Path(proj.style_guide_path).read_text(encoding="utf-8")
    cp = load_translate_checkpoint(proj)
    return {
        "style_guide_path": proj.style_guide_path,
        "content": content,
        "translated_srt_path": proj.translated_srt_path,
        "checkpoint_done": len(cp.get("translated", {})),
        "checkpoint_total": cp.get("total_cues", 0),
    }


class SaveStyleGuideRequest(BaseModel):
    content: str


@app.put("/api/projects/{project_id}/style-guide")
async def save_style_guide(project_id: str, req: SaveStyleGuideRequest):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    p_dir = PROJECTS_DIR / proj.id
    p_dir.mkdir(parents=True, exist_ok=True)
    sg_path = Path(proj.style_guide_path) if proj.style_guide_path else (p_dir / f"{Path(proj.name).stem}_style_guide.md")
    sg_path.write_text(req.content, encoding="utf-8")
    proj.style_guide_path = str(sg_path)
    store.upsert_project(proj)
    return {"ok": True, "style_guide_path": str(sg_path)}


@app.post("/api/projects/{project_id}/analyze-style")
async def run_analyze_style(project_id: str):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    _ensure_idle(project_id)

    proj.status = "translating"
    proj.progress = 3.0
    proj.status_message = "Đang phân tích Topic, Context & Style Guide..."
    store.upsert_project(proj)

    def _on_prog(pct: float, msg: str):
        if is_cancelled(project_id):
            return
        p = store.get_project(project_id)
        if p:
            p.progress = pct
            p.status_message = msg

    async def _task():
        begin_job(project_id)
        timing.start_job(project_id, "Phân tích Style Guide")
        try:
            with timing.step(project_id, "style", "Phân tích Topic / Nhân vật / Văn phong"):
                sg_path = await analyze_style_guide(proj, on_progress=_on_prog)
            p = store.get_project(project_id)
            if p:
                p.style_guide_path = str(sg_path)
                p.status = "ready"
                p.progress = 100.0
                p.status_message = f"Đã tạo xong Style Guide: {sg_path.name}"
                store.upsert_project(p)
            timing.end_job(project_id, True)
        except Exception as e:
            p = store.get_project(project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi phân tích Style Guide: {e}"
                store.upsert_project(p)
            timing.end_job(project_id, False, "Lỗi")

    _ACTIVE_TASKS[project_id] = asyncio.create_task(_task())
    return {"ok": True, "project_id": project_id}


class TranslateRequest(BaseModel):
    reset_checkpoint: bool = False


@app.post("/api/projects/{project_id}/translate")
async def run_translate(project_id: str, req: TranslateRequest):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    _ensure_idle(project_id)

    proj.status = "translating"
    proj.progress = 2.0
    proj.status_message = (
        "Đang khởi động dịch SRT bằng Gemini Web..."
        if proj.config.translate.provider == "gemini_web"
        else "Đang khởi động dịch SRT bằng Antigravity..."
    )
    store.upsert_project(proj)

    def _on_prog(pct: float, msg: str):
        if is_cancelled(project_id):
            return
        p = store.get_project(project_id)
        if p:
            p.progress = pct
            p.status_message = msg

    async def _task():
        begin_job(project_id)
        timing.start_job(project_id, "Dịch SRT")
        try:
            with timing.step(project_id, "translate", "Dịch SRT (phân tích + dịch theo lô)"):
                out_srt = await translate_project_srt(
                    proj, reset_checkpoint=req.reset_checkpoint, on_progress=_on_prog
                )
            p = store.get_project(project_id)
            if p:
                p.translated_srt_path = str(out_srt)
                p.status = "ready"
                p.progress = 100.0
                p.status_message = f"Dịch SRT hoàn tất: {out_srt.name}"
                store.upsert_project(p)
            timing.end_job(project_id, True)
        except Exception as e:
            p = store.get_project(project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi dịch SRT: {e}"
                store.upsert_project(p)
            timing.end_job(project_id, False, "Lỗi")

    _ACTIVE_TASKS[project_id] = asyncio.create_task(_task())
    return {"ok": True, "project_id": project_id}


# ---------------- Phase 3: CapCut / TikTok Free + Edge-TTS Multilingual + gTTS ----------------


@app.get("/api/voices")
async def get_voices(
    provider: str = "all",
    language: str = "",
    include_multilingual: bool = True,
):
    voices = await list_all_voices(
        provider=provider,
        language=language,
        include_multilingual=include_multilingual,
    )
    return {"count": len(voices), "voices": voices}


class PreviewVoiceRequest(BaseModel):
    config: VoiceConfig
    text: str = "Xin chào các bạn, chào mừng đến với kênh review phim hôm nay!"


@app.post("/api/voices/preview")
async def preview_voice(req: PreviewVoiceRequest):
    if req.config.provider == "higgs3":
        from app.services.higgs_voice import preview_line

        if gpu_gate.busy_text():
            raise HTTPException(status_code=409, detail=f"GPU đang bận ({gpu_gate.busy_text()}), thử lại sau.")
        try:
            wav = await preview_line(req.config, req.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return Response(content=wav, media_type="audio/wav", headers={"X-TTS-Provider-Used": "higgs3"})
    try:
        audio_bytes, used = await synthesize_line(req.text, req.config)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"X-TTS-Provider-Used": used},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/projects/{project_id}/voices")
async def run_generate_voices(project_id: str):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    _ensure_idle(project_id)

    proj.status = "voicing"
    proj.progress = 2.0
    proj.status_message = f"Đang tạo voice song song ({proj.config.voice.concurrency} luồng)..."
    store.upsert_project(proj)

    def _on_prog(pct: float, msg: str):
        if is_cancelled(project_id):
            return
        p = store.get_project(project_id)
        if p:
            p.progress = pct
            p.status_message = msg

    async def _task():
        begin_job(project_id)
        timing.start_job(project_id, "Tạo Voice")
        try:
            v_label = ("Higgs TTS 3, 48 câu cùng lúc + Whisper kiểm tra" if proj.config.voice.provider == "higgs3"
                       else f"{proj.config.voice.concurrency} luồng")
            with timing.step(project_id, "voice", f"Tạo Voice ({v_label})"):
                manifest_path = await generate_project_voices(
                    proj,
                    concurrency=proj.config.voice.concurrency,
                    on_progress=_on_prog,
                )
            with timing.step(project_id, "audio", "Ghép audio (thuyết minh / lồng tiếng)"):
                dub_wav, stats = await asyncio.to_thread(
                    assemble_project_audio_sync,
                    proj,
                    str(manifest_path),
                    lambda pct, m: _on_prog(round(90.0 + pct * 0.10, 1), m),
                )
            p = store.get_project(project_id)
            if p:
                p.voice_dir = str(manifest_path.parent)
                p.dub_audio_path = str(dub_wav)
                p.status = "ready"
                p.progress = 100.0
                p.status_message = (
                    f"Tạo voice & ghép {stats['mode']} xong! "
                    f"({stats['cues_assembled']} câu, tốc độ TB {stats['avg_tempo']}x)"
                )
                store.upsert_project(p)
            timing.end_job(project_id, True)
        except Exception as e:
            p = store.get_project(project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi tạo voice: {e}"
                store.upsert_project(p)
            timing.end_job(project_id, False, "Lỗi")

    _ACTIVE_TASKS[project_id] = asyncio.create_task(_task())
    return {"ok": True, "project_id": project_id}


# ---------------- Higgs TTS 3 (voice clone) + review table + length fit ----------------

_HIGGS_INSTALL: Dict[str, Any] = {}


@app.get("/api/higgs/status")
async def higgs_status(refresh: bool = False):
    from app.services import higgs_tts
    from app.services.audio_env import env_status

    if refresh or not _HIGGS_INSTALL:
        _HIGGS_INSTALL.update(await asyncio.to_thread(higgs_tts.installed))
        _HIGGS_INSTALL["env"] = await asyncio.to_thread(env_status)
    return {**_HIGGS_INSTALL, **(await asyncio.to_thread(higgs_tts.health_json)), "gpu": gpu_gate.busy_text()}


@app.post("/api/higgs/stop")
async def higgs_stop():
    from app.services import higgs_tts

    if gpu_gate.exclusive_held:
        raise HTTPException(status_code=409, detail="Higgs đang tạo voice, bấm ⏹ Dừng ở video đó trước.")
    await asyncio.to_thread(higgs_tts.stop)
    return {"ok": True}


@app.get("/api/higgs/refs")
async def higgs_list_refs(folder: str = Query("")):
    """Audio files of a folder with the words file (.txt, same name) next to each."""
    d = Path(folder)
    if not folder or not d.is_dir():
        return {"refs": []}
    out = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"):
            t = f.with_suffix(".txt")
            out.append({"path": str(f), "name": f.name, "text": t.read_text(encoding="utf-8-sig", errors="replace").strip() if t.exists() else ""})
    return {"refs": out}


class RefPrepareRequest(BaseModel):
    config: VoiceConfig


@app.post("/api/higgs/prepare-ref")
async def higgs_prepare_ref(req: RefPrepareRequest):
    """Cleans + cuts the reference voice now (so it can be heard before / after). Cached."""
    from app.services.higgs_voice import prepare_reference

    if gpu_gate.busy_text():
        raise HTTPException(status_code=409, detail=f"GPU đang bận ({gpu_gate.busy_text()}), thử lại sau.")
    try:
        async with gpu_gate.exclusive():
            ref = await asyncio.to_thread(prepare_reference, req.config, req.config.language, None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    rep = ref.get("report") or {}
    return {"wav": ref["wav"], "text": ref["text"], "full_clean": ref["full_clean"], "source": ref["source"],
            "steps": rep.get("lam_sach", []), "skipped": rep.get("da_thu", []), "warnings": rep.get("canh_bao", []),
            "dnsmos_before": rep.get("dnsmos_goc"), "dnsmos_after": rep.get("dnsmos_ra"), "seconds": rep.get("giay")}


def _proj_or_404(project_id: str) -> VideoProject:
    p = store.get_project(project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    return p


@app.get("/api/projects/{project_id}/voice-review")
async def voice_review_list(project_id: str, show: str = "flagged"):
    from app.services.higgs_voice import review_rows

    return await asyncio.to_thread(review_rows, _proj_or_404(project_id), show)


def _voice_pending_sync(proj: VideoProject) -> Dict[str, Any]:
    """Lines whose current text has no voice yet (shortened / edited after the voice was made)."""
    if proj.config.voice.provider == "higgs3":
        from app.services.higgs_voice import pending_lines

        try:
            return pending_lines(proj)
        except RuntimeError:
            return {"indices": [], "has_voice": False, "total": 0}
    from app.services.tts_engine import _cache_key

    srt = proj.translated_srt_path or proj.srt_path
    if not srt or not Path(srt).exists():
        return {"indices": [], "has_voice": False, "total": 0}
    cues = parse_srt(srt)
    vdir = PROJECTS_DIR / proj.id / "voice_cache"
    has_voice = (vdir / "voice_manifest.json").exists()
    out = []
    for c in cues:
        if not c.text.strip():
            continue
        f = vdir / f"{_cache_key(c.text, proj.config.voice)}.mp3"
        if not f.exists() or f.stat().st_size < 200:
            out.append(c.index)
    return {"indices": out, "has_voice": has_voice, "total": len(cues)}


@app.get("/api/projects/{project_id}/voice-pending")
async def voice_pending(project_id: str):
    r = await asyncio.to_thread(_voice_pending_sync, _proj_or_404(project_id))
    return {"pending": len(r["indices"]), "indices": r["indices"][:500], "has_voice": r["has_voice"], "total": r["total"]}


class ReviewRegenRequest(BaseModel):
    indices: List[int] = []
    all_flagged: bool = False
    edits: Dict[int, str] = {}
    pending: bool = False  # + every line whose text changed since its voice (shortened / edited)
    ai_bad: bool = False  # + every line the AI check judged a real reading error


@app.post("/api/projects/{project_id}/voice-review/regen")
async def voice_review_regen(project_id: str, req: ReviewRegenRequest):
    from app.services.higgs_voice import load_state, regenerate_lines, review_rows

    proj = _proj_or_404(project_id)
    _ensure_idle(project_id)
    idxs = list(req.indices) + [int(k) for k in req.edits]
    if req.all_flagged or req.ai_bad:
        rows = (await asyncio.to_thread(review_rows, proj, "flagged"))["rows"]
        if req.all_flagged:
            idxs += [r["index"] for r in rows]
        if req.ai_bad:
            idxs += [r["index"] for r in rows if (r.get("ai") or {}).get("verdict") == "SAI"]
    if req.pending:
        idxs += (await asyncio.to_thread(_voice_pending_sync, proj))["indices"]
    if not idxs:
        raise HTTPException(status_code=400, detail="Không có câu nào cần tạo voice." if req.pending or req.ai_bad
                            else "Chưa chọn câu nào để tạo lại.")

    def _on_prog(pct: float, msg: str):
        if is_cancelled(project_id):
            return
        p = store.get_project(project_id)
        if p:
            p.progress = pct
            p.status_message = msg

    proj.status = "voicing"
    proj.progress = 1.0
    proj.status_message = f"Tạo lại {len(set(idxs))} câu voice Higgs..."
    store.upsert_project(proj)

    async def _task():
        begin_job(project_id)
        timing.start_job(project_id, f"Tạo lại {len(set(idxs))} câu voice")
        try:
            with timing.step(project_id, "voice", "Higgs tạo lại + Whisper nghe lại"):
                rc = await regenerate_lines(proj, idxs, req.edits, _on_prog)
            p = store.get_project(project_id)
            if p:
                p.status = "ready"
                p.progress = 100.0
                p.status_message = f"Tạo lại xong · còn {rc['flagged']} câu nghi ngờ"
                store.upsert_project(p)
            timing.end_job(project_id, True)
        except Exception as e:
            p = store.get_project(project_id)
            if p and not is_cancelled(project_id):
                p.status = "error"
                p.status_message = f"Lỗi tạo lại voice: {e}"
                store.upsert_project(p)
            timing.end_job(project_id, False, "Lỗi")

    _ACTIVE_TASKS[project_id] = asyncio.create_task(_task())
    return {"ok": True, "count": len(set(idxs))}


class ReviewAiCheckRequest(BaseModel):
    indices: List[int] = []  # empty = every suspicious line not judged yet
    recheck: bool = False


@app.post("/api/projects/{project_id}/voice-review/ai-check")
async def voice_review_ai_check(project_id: str, req: ReviewAiCheckRequest):
    """'🤖 AI soát lỗi Whisper': the suspicious lines go to the translation AI (Antigravity / Gemini) which
    tells a Whisper mis-hearing (-> marked OK) from a real reading error of the voice (-> kept, regenerate)."""
    from app.services.voice_ai_check import ai_check_review

    proj = _proj_or_404(project_id)
    _ensure_idle(project_id)

    def _on_prog(pct: float, msg: str):
        if is_cancelled(project_id):
            return
        p = store.get_project(project_id)
        if p:
            p.progress = pct
            p.status_message = msg

    proj.status = "translating"
    proj.progress = 1.0
    proj.status_message = "AI đang soát bảng kiểm tra voice..."
    store.upsert_project(proj)

    async def _task():
        begin_job(project_id)
        timing.start_job(project_id, "AI soát bảng kiểm tra voice")
        try:
            with timing.step(project_id, "translate", "AI soát lỗi Whisper"):
                res = await ai_check_review(proj, req.indices or None, _on_prog, recheck=req.recheck)
            p = store.get_project(project_id)
            if p:
                p.status = "ready"
                p.progress = 100.0
                if not res["checked"]:
                    p.status_message = "AI soát: không còn câu nghi ngờ nào chưa soát."
                else:
                    kept = res["ok"] - res["marked_ok"]
                    p.status_message = (
                        f"AI soát {res['checked']} câu: ✓ {res['marked_ok']} câu chỉ do Whisper nghe nhầm (đã đánh dấu OK)"
                        + (f" · {kept} câu chữ đúng nhưng có lỗi độ dài (giữ lại để nghe)" if kept else "")
                        + f" · ✗ {res['bad']} câu đọc sai thật · ? {res['unsure']} câu cần nghe"
                        + (f" · {res['missing']} câu AI không trả lời" if res["missing"] else "")
                    )
                store.upsert_project(p)
            timing.end_job(project_id, True)
        except Exception as e:
            p = store.get_project(project_id)
            if p and not is_cancelled(project_id):
                p.status = "error"
                p.status_message = f"Lỗi AI soát bảng kiểm tra: {e}"
                store.upsert_project(p)
            timing.end_job(project_id, False, "Lỗi")

    _ACTIVE_TASKS[project_id] = asyncio.create_task(_task())
    return {"ok": True}


class ReviewMarkRequest(BaseModel):
    indices: List[int] = []
    ok: bool = True


@app.post("/api/projects/{project_id}/voice-review/ok")
async def voice_review_ok(project_id: str, req: ReviewMarkRequest):
    from app.services.higgs_voice import set_ok

    return await asyncio.to_thread(set_ok, _proj_or_404(project_id), req.indices, req.ok)


class ReviewSelectRequest(BaseModel):
    index: int
    t: int


@app.post("/api/projects/{project_id}/voice-review/select")
async def voice_review_select(project_id: str, req: ReviewSelectRequest):
    from app.services.higgs_voice import select_version

    try:
        await asyncio.to_thread(select_version, _proj_or_404(project_id), req.index, req.t)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.post("/api/projects/{project_id}/voice-review/done")
async def voice_review_done(project_id: str, run: bool = True):
    """'Xong kiểm tra → chạy tiếp': remembers the check is done (the A-Z run no longer stops for it) and
    continues the A-Z run (translation / voice come from their caches)."""
    from app.services.higgs_voice import mark_reviewed

    proj = _proj_or_404(project_id)
    await asyncio.to_thread(mark_reviewed, proj, True)
    if not run:
        return {"ok": True}
    _ensure_idle(project_id)
    try:
        return pipeline_manager.start_batch([project_id], do_translate=True, do_voice=True, do_render=True)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/projects/{project_id}/fit-length")
async def run_fit_length(project_id: str):
    """'Rút gọn câu tràn khung': rewrites the translated lines that cannot be read in their time (estimated from
    the syllable count). Then 'Tạo voice' voices only those lines again."""
    from app.services.length_fit import fit_project_lines

    proj = _proj_or_404(project_id)
    _ensure_idle(project_id)
    if not proj.translated_srt_path or not Path(proj.translated_srt_path).exists():
        raise HTTPException(status_code=400, detail="Video chưa có bản dịch.")

    def _on_prog(pct: float, msg: str):
        if is_cancelled(project_id):
            return
        p = store.get_project(project_id)
        if p:
            p.progress = pct
            p.status_message = msg

    proj.status = "translating"
    proj.progress = 1.0
    proj.status_message = "Đang tìm câu dịch dài quá khung thời gian..."
    store.upsert_project(proj)

    async def _task():
        begin_job(project_id)
        timing.start_job(project_id, "Rút gọn câu tràn khung")
        try:
            with timing.step(project_id, "translate", "Rút gọn câu tràn khung (AI)"):
                res = await fit_project_lines(proj, on_progress=_on_prog)
            p = store.get_project(project_id)
            if p:
                p.status = "ready"
                p.progress = 100.0
                p.status_message = (
                    f"Rút gọn xong {len(res['changed'])}/{res['over']} câu dài quá khung (trên {res['checked']} câu). "
                    "Nếu đã có voice: bấm «🔄 Tạo voice cho câu đã đổi chữ» ở Bước 3 (chỉ đọc lại các câu đó)."
                    if res["over"] else
                    f"Cả {res['checked']} câu đều vừa khung thời gian, không cần rút gọn."
                )
                store.upsert_project(p)
            timing.end_job(project_id, True)
        except Exception as e:
            p = store.get_project(project_id)
            if p and not is_cancelled(project_id):
                p.status = "error"
                p.status_message = f"Lỗi rút gọn câu: {e}"
                store.upsert_project(p)
            timing.end_job(project_id, False, "Lỗi")

    _ACTIVE_TASKS[project_id] = asyncio.create_task(_task())
    return {"ok": True}


@app.get("/api/projects/{project_id}/length-report")
async def length_report(project_id: str):
    """How many translated lines do not fit their time (estimate), for the hint in step 2."""
    from app.services.length_fit import project_budgets, speech_units

    proj = _proj_or_404(project_id)
    srt = proj.translated_srt_path
    if not srt or not Path(srt).exists():
        return {"checked": 0, "over": 0}
    cues = await asyncio.to_thread(parse_srt, srt)
    b = project_budgets(proj, cues)
    lang = proj.config.translate.target_lang
    over = [c.index for c in cues if speech_units(c.text, lang) > b[c.index] * 1.12 and speech_units(c.text, lang) > 2]
    tot_units = sum(speech_units(c.text, lang) for c in cues)
    tot_budget = sum(b.values())
    return {"checked": len(cues), "over": len(over), "ratio": round(tot_units / max(1, tot_budget), 2)}


# ---------------- Phase 4: AI Clean Sub + Render + Batch A-Z Pipeline ----------------


@app.post("/api/render")
async def start_render(req: RenderRequest):
    proj = store.get_project(req.project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    _ensure_idle(req.project_id)

    proj.status = "rendering"
    proj.progress = 1.0
    proj.status_message = "Đang khởi tạo pipeline xuất video..."
    store.upsert_project(proj)

    def _on_progress(pct: float, msg: str):
        if is_cancelled(req.project_id):
            return
        p = store.get_project(req.project_id)
        if p:
            p.progress = pct
            p.status_message = msg

    async def _runner():
        begin_job(req.project_id)
        timing.start_job(
            req.project_id, f"Xuất thử {req.preview_seconds:g}s" if req.preview_seconds else "Xuất video full"
        )
        try:
            dub_wav: Optional[str] = None
            manifest_cand = PROJECTS_DIR / proj.id / "voice_cache" / "voice_manifest.json"
            if req.use_voice and manifest_cand.exists():
                _on_progress(5.0, "Đang ghép audio (Voice + Nhạc nền Crossfade + Tiếng gốc)...")
                with timing.step(req.project_id, "audio", "Ghép audio (voice + nhạc nền + tiếng gốc)"):
                    wav_p, _ = await asyncio.to_thread(
                        assemble_project_audio_sync,
                        proj,
                        str(manifest_cand),
                        lambda pct, m: _on_progress(round(5.0 + pct * 0.15, 1), m),
                    )
                dub_wav = str(wav_p)
                proj.dub_audio_path = dub_wav

            # AI Clean Sub runs inside the render (single decode + single encode). The GPU gate waits for a
            # running Higgs TTS 3 job and stops an idle Higgs server first (it holds ~12 GB of VRAM).
            base_pct = 20.0
            if gpu_gate.busy_text():
                _on_progress(base_pct, f"Chờ GPU rảnh ({gpu_gate.busy_text()})...")
            async with gpu_gate.shared():
                out_file = await render_project_async(
                    proj,
                    preview_seconds=req.preview_seconds,
                    voice_wav_path=dub_wav,
                    voice_already_mixed=bool(dub_wav),
                    on_progress=lambda pct, m: _on_progress(round(base_pct + pct * ((100.0 - base_pct) / 100.0), 1), m),
                    ai_clean=req.run_clean_sub,
                )
            p = store.get_project(req.project_id)
            if p:
                p.output_path = str(out_file)
                p.status = "done"
                p.progress = 100.0
                p.status_message = f"Hoàn tất: {out_file.name}"
                store.upsert_project(p)
            timing.end_job(req.project_id, True)
        except Exception as e:
            p = store.get_project(req.project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi render: {e}"
                store.upsert_project(p)
            timing.end_job(req.project_id, False, "Lỗi")

    _ACTIVE_TASKS[req.project_id] = asyncio.create_task(_runner())
    return {"ok": True, "project_id": req.project_id}


class PipelineRunRequest(BaseModel):
    project_ids: List[str] = []
    do_translate: bool = True
    do_voice: bool = True
    do_render: bool = True
    profile: str = "auto"


@app.post("/api/pipeline/run-batch")
async def run_batch_pipeline(req: PipelineRunRequest):
    ids = req.project_ids or [p.id for p in store.list_projects()]
    if not ids:
        raise HTTPException(status_code=400, detail="Danh sách video đang trống.")
    busy = [pid for pid in ids if _ACTIVE_TASKS.get(pid) is not None and not _ACTIVE_TASKS[pid].done()]
    if busy:
        names = ", ".join((store.get_project(pid).name if store.get_project(pid) else pid) for pid in busy)
        raise HTTPException(status_code=409, detail=f"Các video đang chạy tiến trình riêng: {names}. Chờ xong hoặc bấm Dừng trước.")
    try:
        st = pipeline_manager.start_batch(
            ids,
            do_translate=req.do_translate,
            do_voice=req.do_voice,
            do_render=req.do_render,
            profile=req.profile,
        )
        return st
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    return pipeline_manager.status()


_STOPPED_MSG = "⏹ Đã dừng tiến trình. Bấm chạy lại để tiếp tục (phần dịch đã xong được giữ trong checkpoint)."


def _mark_stopped(project_id: str) -> None:
    timing.end_job(project_id, False, "Đã dừng")
    p = store.get_project(project_id)
    if p:
        p.status = "ready"
        p.status_message = _STOPPED_MSG
        store.upsert_project(p)


async def _stop_project(project_id: str) -> bool:
    stopped = bool(await pipeline_manager.cancel([project_id]))
    task = _ACTIVE_TASKS.get(project_id)
    if task and not task.done():
        cancel_project(project_id)  # kill agentapi / ffmpeg children so worker threads unblock
        task.cancel()
        await asyncio.wait([task], timeout=10)
        stopped = True
    if stopped:
        _mark_stopped(project_id)
    return stopped


@app.post("/api/projects/{project_id}/cancel")
async def cancel_project_job(project_id: str):
    if not store.get_project(project_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    stopped = await _stop_project(project_id)
    if not stopped:
        # Nothing running (e.g. stale "translating" status after a server restart) -> just reset the status
        p = store.get_project(project_id)
        if p and p.status in ("translating", "voicing", "cleaning", "rendering"):
            _mark_stopped(project_id)
    return {"ok": True, "stopped": stopped}


@app.post("/api/pipeline/cancel")
async def cancel_pipeline():
    ids = set(pipeline_manager.running_project_ids())
    ids |= {pid for pid, t in _ACTIVE_TASKS.items() if not t.done()}
    for pid in ids:
        await _stop_project(pid)
    return {"ok": True, "stopped": sorted(ids)}


@app.get("/api/projects/{project_id}/status")
async def get_project_status(project_id: str):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    return {
        "id": proj.id,
        "status": proj.status,
        "progress": proj.progress,
        "status_message": proj.status_message,
        "timing": proj.timing,
        "output_path": proj.output_path,
        "srt_path": proj.srt_path,
        "translated_srt_path": proj.translated_srt_path,
        "style_guide_path": proj.style_guide_path,
        "dub_audio_path": proj.dub_audio_path,
    }
