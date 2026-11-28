from __future__ import annotations
import asyncio
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import PROJECTS_DIR
from app.models import VideoProject
from app.services.audio_assembler import assemble_project_audio_sync
from app.services.cancel import begin_job, cancel_project, is_cancelled
from app.services.gpu_gate import gpu_gate
from app.services.hardware import detect_hardware
from app.services.render_engine import render_project_async
from app.services import timing
from app.services.store import store
from app.services.sub_remover import ai_clean_boxes, prescan_ocr_sync
from app.services.translator import translate_project_srt
from app.services.tts_engine import generate_project_voices


@contextlib.asynccontextmanager
async def _slot(sems, project_id: str, label: str):
    """Acquires the batch semaphores; the time spent waiting for a free slot shows up as its own step."""
    sems = [s for s in sems if s is not None] or [asyncio.Semaphore(1)]
    st = timing.begin_step(project_id, "wait", label) if any(s.locked() for s in sems) else None
    async with contextlib.AsyncExitStack() as stack:
        for s in sems:
            await stack.enter_async_context(s)
        timing.end_step(st)
        yield


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
            # ai_gpu=1: one AI Clean Sub render already saturates the CPU (inpaint threads scale with cores);
            # 2 at once measured the same total fps (i5-12400F + RTX 3060: 290 fps either way) but each video
            # finished later and RAM doubled. Other videos keep translating / voicing / OCR-prescanning meanwhile.
            return {"io_jobs": 4, "tts_conc": 10, "ai_gpu": 1, "encode_gpu": hw["recommended_render_workers"]}
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

        prescan_task: Optional[asyncio.Task] = None
        ok = False
        timing.start_job(project_id, "Chạy trọn gói A–Z" if (do_translate and do_voice and do_render) else "Chạy các bước đã chọn")
        try:
            # 0. AI Clean Sub: OCR-scan the hardsub while translation / TTS wait on the network (CPU + GPU are
            #    idle then). The render later reads the OCR cache instead of scanning (~15% faster render).
            will_io = (do_translate and bool(proj.srt_path) and Path(proj.srt_path).exists()) or (
                do_voice and bool(proj.translated_srt_path or proj.srt_path)
            )
            sr = proj.config.sub_removal
            # Higgs TTS 3 needs the whole GPU: with it, the OCR scan starts only after the voice is done
            higgs = do_voice and proj.config.voice.provider == "higgs3"
            want_prescan = do_render and ai_clean_boxes(proj) and sr.auto_detect_ocr and not preview_seconds

            def _start_prescan(label: str) -> asyncio.Task:
                def _prescan_timed() -> int:
                    st = timing.begin_step(project_id, "ocr_prescan", label)
                    try:
                        n = prescan_ocr_sync(proj)
                    except BaseException:
                        timing.end_step(st, ok=False)
                        raise
                    timing.end_step(st, ok=True, note=f"{n} mẫu OCR" if n else "đã có cache")
                    return n

                async def _gated() -> int:
                    async with gpu_gate.shared():
                        return await asyncio.to_thread(_prescan_timed)

                return asyncio.create_task(_gated())

            if want_prescan and will_io and not higgs:
                prescan_task = _start_prescan("Quét OCR sub gốc (song song lúc dịch / voice)")

            # 1. Translate SRT (if requested & source SRT exists)
            if do_translate and proj.srt_path and Path(proj.srt_path).exists():
                async with _slot([io_sem], project_id, "Chờ lượt dịch (hàng loạt)"):
                    _update("translating", 2.0, "[Bước 1/4] Bắt đầu phân tích & dịch SRT...")
                    with timing.step(project_id, "translate", "Dịch SRT"):
                        out_srt = await translate_project_srt(
                            proj,
                            on_progress=lambda p, m: _update("translating", p * 0.25, f"[Dịch SRT] {m}"),
                        )
                    proj.translated_srt_path = str(out_srt)
                    store.upsert_project(proj)

            # 2. Generate Voice (CapCut/TikTok Free + Edge-TTS + gTTS with user concurrency)
            manifest_path: Optional[Path] = None
            if do_voice and (proj.translated_srt_path or proj.srt_path):
                async with _slot([io_sem], project_id, "Chờ lượt tạo voice (hàng loạt)"):
                    v_conc = proj.config.voice.concurrency or tts_conc
                    v_label = "Higgs TTS 3, 48 câu cùng lúc + Whisper kiểm tra" if higgs else f"{v_conc} luồng"
                    _update("voicing", 28.0, f"[Bước 2/4] Bắt đầu tạo Voice AI ({v_label})...")
                    with timing.step(project_id, "voice", f"Tạo Voice ({v_label})"):
                        manifest_path = await generate_project_voices(
                            proj,
                            concurrency=v_conc,
                            on_progress=lambda p, m: _update("voicing", 25.0 + p * 0.25, f"[Voice] {m}"),
                        )
                    proj.voice_dir = str(manifest_path.parent)
                    store.upsert_project(proj)
                if higgs:
                    from app.services.higgs_voice import needs_review_pause

                    n_bad = needs_review_pause(proj)
                    if n_bad:
                        proj.status = "ready"
                        proj.progress = 100.0
                        proj.status_message = (
                            f"⏸ Tạm dừng: Whisper thấy {n_bad} câu voice nghi ngờ. Mở «Bảng kiểm tra voice» ở Bước 3, "
                            "nghe / tạo lại, rồi bấm «Xong kiểm tra → chạy tiếp»."
                        )
                        store.upsert_project(proj)
                        ok = True
                        return {"ok": True, "project_id": proj.id, "paused_for_review": n_bad}
            else:
                # Reuse existing manifest if present
                cand = PROJECTS_DIR / proj.id / "voice_cache" / "voice_manifest.json"
                if cand.exists():
                    manifest_path = cand

            if want_prescan and prescan_task is None and higgs:
                prescan_task = _start_prescan("Quét OCR sub gốc (sau khi Higgs tạo voice xong)")

            # 3. Assemble Narration / Dubbing Audio WAV
            dub_wav: Optional[Path] = None
            if do_render and manifest_path and manifest_path.exists():
                _update("rendering", 52.0, "[Bước 3/4] Đang ghép timeline Thuyết minh/Lồng tiếng...")
                with timing.step(project_id, "audio", "Ghép audio (thuyết minh / lồng tiếng)"):
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
            if prescan_task is not None:
                wait_st = None
                if not prescan_task.done():
                    _update("rendering", 54.0, "[Bước 4/4] Đang chờ OCR quét sub gốc xong...")
                    wait_st = timing.begin_step(project_id, "wait", "Chờ OCR quét sub gốc xong")
                try:
                    await prescan_task
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass  # the render scans by itself (OCR on the fly) if the pre-scan failed
                timing.end_step(wait_st)
                prescan_task = None

            if do_render:
                ai_on = proj.config.sub_removal.mode == "ai_clean"
                async with _slot([enc_sem, ai_sem if ai_on else None], project_id, "Chờ lượt xuất video (hàng loạt)"), \
                        gpu_gate.shared():
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
                    ok = True
                    return {"ok": True, "project_id": proj.id, "output_path": str(out_mp4)}

            proj.status = "done"
            proj.progress = 100.0
            proj.status_message = "Hoàn tất các bước đã chọn!"
            store.upsert_project(proj)
            ok = True
            return {"ok": True, "project_id": proj.id}

        except Exception as e:
            p = store.get_project(project_id)
            if p and not is_cancelled(project_id):
                p.status = "error"
                p.status_message = f"Lỗi: {e}"
                store.upsert_project(p)
            raise
        finally:
            timing.end_job(project_id, ok, "" if ok else ("Đã dừng" if is_cancelled(project_id) else "Lỗi"))
            if prescan_task is not None and not prescan_task.done():
                # translation/voice failed or was stopped: let the scan thread end (it checks the cancel flag,
                # and its result is only a cache) without leaving an un-awaited task error behind.
                prescan_task.add_done_callback(lambda t: t.cancelled() or t.exception())

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
