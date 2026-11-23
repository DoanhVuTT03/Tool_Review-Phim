from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import PROJECTS_DIR
from app.models import VideoProject
from app.services.audio_assembler import assemble_project_audio_sync
from app.services.cancel import begin_job, cancel_project, is_cancelled
from app.services.hardware import detect_hardware
from app.services.render_engine import render_project_async
from app.services.store import store
from app.services.translator import translate_project_srt
from app.services.tts_engine import generate_project_voices


class BatchPipelineManager:
    """Producer–Consumer Dual-GPU Pipeline Orchestrator:
    Overlaps I/O stages (Antigravity translation + TTS downloads) with AI GPU (OCR/Inpaint on MX230/RTX)
    and Hardware Video Encoder (Intel QSV / NVIDIA NVENC) across multiple videos simultaneously.
    """

    def __init__(self) -> None:
        self.running: bool = False
        self.total: int = 0
        self.completed: int = 0
        self.failed: int = 0
        self.current_message: str = "Sẵn sàng."
        self._batch_task: Optional[asyncio.Task] = None
        self._project_tasks: Dict[str, asyncio.Task] = {}
        self.cancelled: int = 0

    def _compute_limits(self, profile: str) -> Dict[str, int]:
        hw = detect_hardware()
        tier = hw["tier"]
        if profile == "workstation" or (profile == "auto" and tier == "workstation"):
            return {"io_jobs": 4, "tts_conc": 10, "ai_gpu": 2, "encode_gpu": hw["recommended_render_workers"]}
        if profile == "dual_gpu_max":
            return {"io_jobs": 3, "tts_conc": 8, "ai_gpu": 1, "encode_gpu": 2}
        # Balanced laptop default (protects MX230 2GB VRAM & laptop thermals while overlapping stages)
        return {"io_jobs": 2, "tts_conc": 6, "ai_gpu": 1, "encode_gpu": 1}

    async def run_single_project_az(
        self,
        project_id: str,
        *,
        do_translate: bool = True,
        do_voice: bool = True,
        do_render: bool = True,
        preview_seconds: Optional[float] = None,
        io_sem: Optional[asyncio.Semaphore] = None,
        ai_sem: Optional[asyncio.Semaphore] = None,
        enc_sem: Optional[asyncio.Semaphore] = None,
        tts_conc: int = 6,
    ) -> Dict[str, Any]:
        proj = store.get_project(project_id)
        if not proj:
            raise RuntimeError(f"Không tìm thấy project {project_id}")

        begin_job(project_id)

        def _update(status: str, pct: float, msg: str):
            if is_cancelled(project_id):
                return
            p = store.get_project(project_id)
            if p:
                p.status = status  # type: ignore[assignment]
                p.progress = round(pct, 1)
                p.status_message = msg

        try:
            # 1. Translate SRT (if requested & source SRT exists)
            if do_translate and proj.srt_path and Path(proj.srt_path).exists():
                async with (io_sem or asyncio.Semaphore(1)):
                    _update("translating", 2.0, "[Bước 1/4] Bắt đầu phân tích & dịch SRT...")
                    out_srt = await translate_project_srt(
                        proj,
                        on_progress=lambda p, m: _update("translating", p * 0.25, f"[Dịch SRT] {m}"),
                    )
                    proj.translated_srt_path = str(out_srt)
                    store.upsert_project(proj)

            # 2. Generate Voice (CapCut/TikTok Free + Edge-TTS + gTTS with user concurrency)
            manifest_path: Optional[Path] = None
            if do_voice and (proj.translated_srt_path or proj.srt_path):
                async with (io_sem or asyncio.Semaphore(1)):
                    v_conc = proj.config.voice.concurrency or tts_conc
                    _update("voicing", 28.0, f"[Bước 2/4] Bắt đầu tạo Voice AI ({v_conc} luồng)...")
                    manifest_path = await generate_project_voices(
                        proj,
                        concurrency=v_conc,
                        on_progress=lambda p, m: _update("voicing", 25.0 + p * 0.25, f"[Voice] {m}"),
                    )
                    proj.voice_dir = str(manifest_path.parent)
                    store.upsert_project(proj)
            else:
                # Reuse existing manifest if present
                cand = PROJECTS_DIR / proj.id / "voice_cache" / "voice_manifest.json"
                if cand.exists():
                    manifest_path = cand

            # 3. Assemble Narration / Dubbing Audio WAV
            dub_wav: Optional[Path] = None
            if do_render and manifest_path and manifest_path.exists():
                _update("rendering", 52.0, "[Bước 3/4] Đang ghép timeline Thuyết minh/Lồng tiếng...")
                dub_wav, _ = await asyncio.to_thread(
                    assemble_project_audio_sync,
                    proj,
                    str(manifest_path),
                    lambda p, m: _update("rendering", 50.0 + p * 0.12, f"[Audio] {m}"),
                )
                proj.dub_audio_path = str(dub_wav)
                store.upsert_project(proj)
            elif proj.dub_audio_path and Path(proj.dub_audio_path).exists():
                dub_wav = Path(proj.dub_audio_path)

            # 4. Final Single-Pass Render: AI Clean Sub (OCR on CPU+Iris+MX230, inpaint) is streamed into the
            #    same ffmpeg that overlays subtitles/logo and encodes once with QSV/NVENC.
            if do_render:
                ai_on = proj.config.sub_removal.mode == "ai_clean"
                async with (enc_sem or asyncio.Semaphore(1)):
                    async with (ai_sem if ai_on and ai_sem else asyncio.Semaphore(1)):
                        _update(
                            "rendering", 55.0,
                            "[Bước 4/4] Đang xóa sub AI + xuất video trong 1 lượt..." if ai_on
                            else "[Bước 4/4] Đang xuất video hoàn chỉnh bằng GPU...",
                        )
                        out_mp4 = await render_project_async(
                            proj,
                            preview_seconds=preview_seconds,
                            voice_wav_path=str(dub_wav) if dub_wav else None,
                            voice_already_mixed=bool(dub_wav),
                            on_progress=lambda p, m: _update("rendering", 55.0 + p * 0.45, f"[Xuất Video] {m}"),
                        )
                    proj.output_path = str(out_mp4)
                    proj.status = "done"
                    proj.progress = 100.0
                    proj.status_message = f"Hoàn tất A–Z: {out_mp4.name}"
                    store.upsert_project(proj)
                    return {"ok": True, "project_id": proj.id, "output_path": str(out_mp4)}

            proj.status = "done"
            proj.progress = 100.0
            proj.status_message = "Hoàn tất các bước đã chọn!"
            store.upsert_project(proj)
            return {"ok": True, "project_id": proj.id}

        except Exception as e:
            p = store.get_project(project_id)
            if p and not is_cancelled(project_id):
                p.status = "error"
                p.status_message = f"Lỗi: {e}"
                store.upsert_project(p)
            raise

    def start_batch(
        self,
        project_ids: List[str],
        *,
        do_translate: bool = True,
        do_voice: bool = True,
        do_render: bool = True,
        profile: str = "auto",
    ) -> Dict[str, Any]:
        if self.running and self._batch_task and not self._batch_task.done():
            raise RuntimeError("Đang có tiến trình chạy hàng loạt, vui lòng chờ hoàn tất.")

        limits = self._compute_limits(profile)
        self.running = True
        self.total = len(project_ids)
        self.completed = 0
        self.failed = 0
        self.cancelled = 0
        self.current_message = (
            f"Đang chạy song song {self.total} video "
            f"(I/O={limits['io_jobs']}, AI={limits['ai_gpu']}, Encode={limits['encode_gpu']})..."
        )

        io_sem = asyncio.Semaphore(limits["io_jobs"])
        ai_sem = asyncio.Semaphore(limits["ai_gpu"])
        enc_sem = asyncio.Semaphore(limits["encode_gpu"])

        async def _runner():
            async def _one(pid: str):
                try:
                    await self.run_single_project_az(
                        pid,
                        do_translate=do_translate,
                        do_voice=do_voice,
                        do_render=do_render,
                        io_sem=io_sem,
                        ai_sem=ai_sem,
                        enc_sem=enc_sem,
                        tts_conc=limits["tts_conc"],
                    )
                    self.completed += 1
                except asyncio.CancelledError:
                    self.cancelled += 1
                except Exception:
                    if is_cancelled(pid):
                        self.cancelled += 1
                    else:
                        self.failed += 1
                self.current_message = (
                    f"Tiến độ hàng loạt: Xong {self.completed}/{self.total}"
                    + (f" (Lỗi: {self.failed})" if self.failed else "")
                    + (f" (Đã dừng: {self.cancelled})" if self.cancelled else "")
                )

            self._project_tasks = {pid: asyncio.create_task(_one(pid)) for pid in project_ids}
            try:
                await asyncio.gather(*self._project_tasks.values(), return_exceptions=True)
            finally:
                self.running = False
                self.current_message = (
                    f"Hoàn tất lô {self.total} video: Thành công {self.completed}, Lỗi {self.failed}"
                    + (f", Đã dừng {self.cancelled}" if self.cancelled else "")
                    + "."
                )

        self._batch_task = asyncio.create_task(_runner())
        return self.status()

    def running_project_ids(self) -> List[str]:
        return [pid for pid, t in self._project_tasks.items() if not t.done()]

    async def cancel(self, project_ids: Optional[List[str]] = None) -> List[str]:
        """Stops the given projects (or every project of the running batch). Returns ids actually stopped."""
        targets = [pid for pid in (project_ids or list(self._project_tasks)) if pid in self.running_project_ids()]
        for pid in targets:
            cancel_project(pid)  # kill agentapi / ffmpeg children first so worker threads unblock
            self._project_tasks[pid].cancel()
        if targets:
            await asyncio.wait([self._project_tasks[p] for p in targets], timeout=10)
        return targets

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "running_projects": self.running_project_ids(),
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "message": self.current_message,
        }


pipeline_manager = BatchPipelineManager()
