from __future__ import annotations
import asyncio
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
from app.services.hardware import detect_hardware
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


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = WEB_DIR / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


# ---------------- Hardware & System ----------------

# Heartbeat from the UI window: the hidden launcher (launcher.pyw) stops the server some time after the
# window is closed, but never while a job is still running.
_LAST_HEARTBEAT: Dict[str, float] = {"t": 0.0}


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


@app.get("/api/hardware")
async def get_hardware(refresh: bool = False):
    return await asyncio.to_thread(detect_hardware, refresh)


@app.get("/api/config/languages")
async def get_languages():
    return {"languages": WORLD_LANGUAGES}


@app.get("/api/config/models")
async def get_models():
    return {"models": ANTIGRAVITY_MODELS}


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


@app.post("/api/download")
async def start_download(req: DownloadVideoRequest):
    job_id = uuid.uuid4().hex[:10]
    _DOWNLOAD_JOBS[job_id] = {
        "job_id": job_id,
        "status": "downloading",
        "progress": 2.0,
        "message": "Đang khởi tạo tải siêu tốc...",
        "project": None,
        "error": "",
    }

    def _on_prog(pct: float, msg: str):
        _DOWNLOAD_JOBS[job_id]["progress"] = pct
        _DOWNLOAD_JOBS[job_id]["message"] = msg

    async def _runner():
        try:
            res = await download_and_clean_video(req, on_progress=_on_prog)
            vp = Path(res["video_path"])
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
            _DOWNLOAD_JOBS[job_id]["message"] = f"Tải xong! Đã thêm '{vp.name}' vào danh sách."
            _DOWNLOAD_JOBS[job_id]["project"] = proj.model_dump()
            _DOWNLOAD_JOBS[job_id]["clean_stats"] = res.get("clean_stats", {})
        except Exception as e:
            clean_err = format_download_error(e)
            _DOWNLOAD_JOBS[job_id]["status"] = "error"
            _DOWNLOAD_JOBS[job_id]["error"] = clean_err
            _DOWNLOAD_JOBS[job_id]["message"] = f"Lỗi tải video: {clean_err}"

    asyncio.create_task(_runner())
    return _DOWNLOAD_JOBS[job_id]


@app.get("/api/download/{job_id}/status")
async def get_download_status(job_id: str):
    if job_id not in _DOWNLOAD_JOBS:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ tải.")
    return _DOWNLOAD_JOBS[job_id]


class CleanSrtRequest(BaseModel):
    use_ai: bool = True
    model: str = "gemini-3.8-flash"
    thinking_level: str = "low"
    api_key: str = ""


@app.post("/api/projects/{project_id}/clean-srt")
async def run_clean_project_srt(project_id: str, req: CleanSrtRequest):
    proj = store.get_project(project_id)
    if not proj or not proj.srt_path or not Path(proj.srt_path).exists():
        raise HTTPException(status_code=400, detail="Video chưa có file SRT gốc để dọn rác.")

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
        try:
            _, r_stats = await asyncio.to_thread(clean_srt_rules_sync, proj.srt_path)
            removed = r_stats["removed_count"]
            final_srt = proj.srt_path
            if req.use_ai:
                out_p, ai_stats = await clean_srt_with_antigravity(
                    proj.srt_path,
                    model=req.model,
                    thinking_level=req.thinking_level,
                    api_key=req.api_key or proj.config.translate.api_key,
                    on_progress=_on_prog,
                )
                final_srt = str(out_p)
                removed += ai_stats["removed_count"]
            p = store.get_project(project_id)
            if p:
                p.srt_path = final_srt
                p.status = "ready"
                p.progress = 100.0
                p.status_message = f"Dọn rác SRT xong! Đã loại bỏ {removed} dòng rác/trùng."
                store.upsert_project(p)
        except Exception as e:
            p = store.get_project(project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi dọn rác SRT: {e}"
                store.upsert_project(p)

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
        try:
            sg_path = await analyze_style_guide(proj, on_progress=_on_prog)
            p = store.get_project(project_id)
            if p:
                p.style_guide_path = str(sg_path)
                p.status = "ready"
                p.progress = 100.0
                p.status_message = f"Đã tạo xong Style Guide: {sg_path.name}"
                store.upsert_project(p)
        except Exception as e:
            p = store.get_project(project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi phân tích Style Guide: {e}"
                store.upsert_project(p)

    _ACTIVE_TASKS[project_id] = asyncio.create_task(_task())
    return {"ok": True, "project_id": project_id}


class TranslateRequest(BaseModel):
    reset_checkpoint: bool = False


@app.post("/api/projects/{project_id}/translate")
async def run_translate(project_id: str, req: TranslateRequest):
    proj = store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")

    proj.status = "translating"
    proj.progress = 2.0
    proj.status_message = "Đang khởi động dịch SRT bằng Antigravity..."
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
        try:
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
        except Exception as e:
            p = store.get_project(project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi dịch SRT: {e}"
                store.upsert_project(p)

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
        try:
            manifest_path = await generate_project_voices(
                proj,
                concurrency=proj.config.voice.concurrency,
                on_progress=_on_prog,
            )
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
        except Exception as e:
            p = store.get_project(project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi tạo voice: {e}"
                store.upsert_project(p)

    _ACTIVE_TASKS[project_id] = asyncio.create_task(_task())
    return {"ok": True, "project_id": project_id}


# ---------------- Phase 4: AI Clean Sub + Render + Batch A-Z Pipeline ----------------


@app.post("/api/render")
async def start_render(req: RenderRequest):
    proj = store.get_project(req.project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    if req.project_id in _ACTIVE_TASKS and not _ACTIVE_TASKS[req.project_id].done():
        raise HTTPException(status_code=409, detail="Video này đang có tiến trình chạy.")

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
        try:
            dub_wav: Optional[str] = None
            manifest_cand = PROJECTS_DIR / proj.id / "voice_cache" / "voice_manifest.json"
            if req.use_voice and manifest_cand.exists():
                _on_progress(5.0, "Đang ghép audio (Voice + Nhạc nền Crossfade + Tiếng gốc)...")
                wav_p, _ = await asyncio.to_thread(
                    assemble_project_audio_sync,
                    proj,
                    str(manifest_cand),
                    lambda pct, m: _on_progress(round(5.0 + pct * 0.15, 1), m),
                )
                dub_wav = str(wav_p)
                proj.dub_audio_path = dub_wav

            # AI Clean Sub runs inside the render (single decode + single encode)
            base_pct = 20.0
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
        except Exception as e:
            p = store.get_project(req.project_id)
            if p:
                p.status = "error"
                p.status_message = f"Lỗi render: {e}"
                store.upsert_project(p)

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
        "output_path": proj.output_path,
        "srt_path": proj.srt_path,
        "translated_srt_path": proj.translated_srt_path,
        "style_guide_path": proj.style_guide_path,
        "dub_audio_path": proj.dub_audio_path,
    }
