from __future__ import annotations
import asyncio
import contextvars
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from app.config import OUTPUTS_DIR, TEMP_DIR, get_ffmpeg_path
from app.models import EditConfig, SubtitleCue, VideoProject
from app.services.cancel import check_cancel, register_proc, unregister_proc
from app.services import timing
from app.services.hardware import detect_hardware
from app.services.media import parse_srt
from app.services.sub_remover import ai_clean_boxes, make_clean_stream
from app.services.subtitle_renderer import build_channel_text_sequence, build_subtitle_concat_sequence


ProgressCallback = Callable[[float, str], None]

# NVDEC -> CUDA overlays -> NVENC graph on NVIDIA (falls back to the CPU graph by itself). Switch for diagnostics.
GPU_GRAPH_ENABLED = True


def _pick_encoder(cfg: EditConfig) -> str:
    hw = detect_hardware()
    if cfg.encoder_override != "auto":
        if hw["encoders"].get(cfg.encoder_override, False):
            return cfg.encoder_override
    return hw["recommended_encoder"]


def _encoder_args(encoder: str, gpu_frames: bool = False) -> List[str]:
    if encoder == "h264_nvenc" and gpu_frames:
        # frames are already yuv420p in VRAM (CUDA graph): no -pix_fmt, NVENC reads them directly
        return ["-c:v", "h264_nvenc", "-preset", "p2", "-rc", "vbr", "-cq", "22"]
    if encoder == "h264_nvenc":
        # p2 vs p4 on RTX 3060 @1080p: 635 vs 387 fps, same SSIM at cq 22 (file ~10% larger). p4 capped the
        # whole render (NVENC was the slowest stage once inpainting scaled with CPU threads).
        return ["-c:v", "h264_nvenc", "-preset", "p2", "-rc", "vbr", "-cq", "22", "-pix_fmt", "yuv420p"]
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "faster", "-global_quality", "23", "-pix_fmt", "nv12"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]


def _output_size_ok(path: Path, vw: int, vh: int) -> bool:
    """The CUDA overlay graph must keep the source frame size (see the overlay_cuda note in the logo step)."""
    from app.config import get_ffprobe_path

    try:
        res = subprocess.run(
            [get_ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        w, h = (int(x) for x in res.stdout.strip().split("x")[:2])
        return (w, h) == (vw, vh)
    except Exception:
        return True  # cannot tell -> keep the result


def _sanitize_hex(color: str) -> str:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", c):
        c = "00FF00"
    return f"0x{c.upper()}"


def _clamp_box_pixels(x: float, y: float, w: float, h: float, vw: int, vh: int) -> tuple[int, int, int, int]:
    px = max(0, min(vw - 2, int(round(x * vw))))
    py = max(0, min(vh - 2, int(round(y * vh))))
    pw = max(2, min(vw - px, int(round(w * vw))))
    ph = max(2, min(vh - py, int(round(h * vh))))
    pw = max(2, pw - (pw % 2))
    ph = max(2, ph - (ph % 2))
    return px, py, pw, ph


def render_project_sync(
    project: VideoProject,
    preview_seconds: Optional[float] = None,
    voice_wav_path: Optional[str] = None,
    source_video_override: Optional[str] = None,
    voice_already_mixed: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    ai_clean: bool = True,
) -> Path:
    """Single-pass hardware-accelerated FFmpeg render pipeline (supports Blur, AI Clean Sub streamed in the same
    pass, rounded-corner subtitles, logo, green-screen CTA, and narration/dubbing mixed WAV)."""
    cfg = project.config
    ffmpeg = get_ffmpeg_path()
    if not Path(ffmpeg).exists() and not shutil.which(ffmpeg):
        raise RuntimeError(f"Không tìm thấy ffmpeg tại: {ffmpeg}")

    src_video = Path(source_video_override) if source_video_override else Path(project.video_path)
    if not src_video.exists():
        raise RuntimeError(f"File video không tồn tại: {src_video}")

    vw, vh = max(16, project.width), max(16, project.height)
    total_dur = project.duration if project.duration > 0 else 10.0
    if preview_seconds and preview_seconds > 0:
        total_dur = min(total_dur, preview_seconds)

    work_dir = TEMP_DIR / f"render_{project.id}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if on_progress:
        on_progress(5.0, "Đang chuẩn bị tài nguyên & lớp phụ đề...")

    inputs: List[str] = ["-y", "-hide_banner", "-loglevel", "error", "-progress", "pipe:1"]
    hwaccel_at = len(inputs)  # GPU graph: "-hwaccel cuda ..." goes right before the main input
    if preview_seconds and preview_seconds > 0:
        inputs.extend(["-t", f"{total_dur:.3f}"])
    inputs.extend(["-i", str(src_video)])
    next_input_idx = 1

    # Subtitle PNG sequence (true rounded corners & independent WxH box matching Preview 1:1)
    sub_input_idx: Optional[int] = None
    sub_box_xy = (0, 0)
    if cfg.subtitle.enabled:
        srt_file = project.translated_srt_path or project.srt_path
        cues: List[SubtitleCue] = []
        if srt_file and Path(srt_file).exists():
            cues = parse_srt(srt_file)
        elif preview_seconds:
            # Show sample subtitle text when testing preview without an SRT
            cues = [
                SubtitleCue(
                    index=1,
                    start_sec=0.0,
                    end_sec=total_dur,
                    text=cfg.subtitle.sample_text or "Phụ đề xem trước",
                )
            ]
        if cues:
            with timing.step(project.id, "subs", f"Tạo phụ đề PNG ({len(cues)} câu)"):
                concat_txt, sx, sy, _, _ = build_subtitle_concat_sequence(
                    cues=cues,
                    cfg=cfg.subtitle,
                    video_w=vw,
                    video_h=vh,
                    video_duration=total_dur,
                    work_dir=work_dir / "subs",
                    fps=float(project.fps or 30.0),
                )
            inputs.extend(["-f", "concat", "-safe", "0", "-i", str(concat_txt)])
            sub_input_idx = next_input_idx
            sub_box_xy = (sx, sy)
            next_input_idx += 1

    # Logo input
    logo_input_idx: Optional[int] = None
    logo_at = -1
    if cfg.logo.enabled and cfg.logo.file_path and Path(cfg.logo.file_path).exists():
        logo_at = len(inputs)  # GPU graph inserts "-loop 1" here (see the logo overlay below)
        inputs.extend(["-i", cfg.logo.file_path])
        logo_input_idx = next_input_idx
        next_input_idx += 1

    # Channel name text (endless input: one looped PNG, or a looped concat of animation frames)
    ch_input_idx: Optional[int] = None
    ch_xy = (0, 0)
    ch = cfg.channel_text
    if ch.enabled and (ch.text or "").strip():
        ch_args, chx, chy = build_channel_text_sequence(ch, vw, vh, float(project.fps or 30.0), work_dir / "channel")
        inputs.extend(ch_args)
        ch_input_idx = next_input_idx
        ch_xy = (chx, chy)
        next_input_idx += 1

    # Green screen CTA input
    gs_input_idx: Optional[int] = None
    if cfg.green_screen.enabled and cfg.green_screen.file_path and Path(cfg.green_screen.file_path).exists():
        if cfg.green_screen.loop_enabled:
            inputs.extend(["-stream_loop", "-1"])
        inputs.extend(["-i", cfg.green_screen.file_path])
        gs_input_idx = next_input_idx
        next_input_idx += 1

    # Optional Voice / Mixed BGM WAV input
    voice_input_idx: Optional[int] = None
    if not voice_wav_path and cfg.bgm.enabled and cfg.bgm.file_paths:
        from app.services.audio_assembler import build_standalone_audio_mix_sync
        with timing.step(project.id, "bgm", "Trộn nhạc nền"):
            bgm_mix = build_standalone_audio_mix_sync(project, total_dur)
        if bgm_mix and bgm_mix.exists():
            voice_wav_path = str(bgm_mix)
            voice_already_mixed = True

    if voice_wav_path and Path(voice_wav_path).exists():
        inputs.extend(["-i", voice_wav_path])
        voice_input_idx = next_input_idx
        next_input_idx += 1

    # AI Clean Sub strips (subtitle box + text-logo / watermark boxes), produced by Python while this same
    # ffmpeg encodes (one decode + one encode)
    use_ai_clean = ai_clean and bool(ai_clean_boxes(project)) and not source_video_override
    ai_stream_idx: Optional[int] = None
    if use_ai_clean:
        if on_progress:
            on_progress(6.0, "Đang khởi động AI xóa sub / chữ logo (OCR + tái tạo nền)...")
        probe = make_clean_stream(project, preview_seconds)
        inputs.extend(probe.ffmpeg_input_args())
        ai_stream_idx = next_input_idx
        next_input_idx += 1

    # Build filter_complex chain. Two variants of the same graph:
    #  - GPU (NVIDIA): NVDEC decode -> frames stay in VRAM -> overlay_cuda for every layer -> NVENC. Only the
    #    small overlay inputs (AI strip, subtitle PNGs, logo, CTA) are prepared on the CPU and uploaded.
    #    RTX 3060 1080p, ffmpeg alone: 505 fps vs 441 fps with CPU decode + CPU overlays, and the CPU is
    #    left to the ROI reader + AI Clean Sub.
    #  - CPU: software decode + overlay filters (any encoder; also the fallback if the GPU graph fails).
    def _ev(v: int) -> int:
        return v - (v % 2)  # yuv420p overlays on the GPU need even offsets (chroma is half size)

    def _video_chain(gpu: bool) -> tuple[List[str], str]:
        vf: List[str] = []
        cur = "0:v"
        ov = "overlay_cuda" if gpu else "overlay"
        to_gpu = ",format=yuva420p,hwupload_cuda" if gpu else ""
        if gpu:
            vf.append("[0:v]scale_cuda=format=yuv420p[v_gpu]")
            cur = "v_gpu"

        # 0. AI Clean Sub overlay (cleaned ROI strip on top of the original frame, frame-exact)
        if ai_stream_idx is not None:
            vf.extend(probe.overlay_filters(cur, ai_stream_idx, "v_after_clean", gpu=gpu))
            cur = "v_after_clean"

        # 1. Sub removal (Blur box mode) - CPU graph only (no CUDA box blur)
        if cfg.sub_removal.mode == "blur":
            bx, by, bw, bh = _clamp_box_pixels(
                cfg.sub_removal.box.x, cfg.sub_removal.box.y, cfg.sub_removal.box.w, cfg.sub_removal.box.h, vw, vh
            )
            # Ensure boxblur radius doesn't exceed chroma plane limits (min(bw, bh)/4)
            max_rad = max(2, min(bw, bh) // 4 - 1)
            rad = min(max(2, cfg.sub_removal.blur_strength), max_rad)
            vf.append(f"[{cur}]split=2[vbase][vblursrc]")
            vf.append(
                f"[vblursrc]crop={bw}:{bh}:{bx}:{by},boxblur=luma_radius={rad}:luma_power=2:chroma_radius={max(1, rad // 2)}:chroma_power=1[vblurred]"
            )
            vf.append(f"[vbase][vblurred]overlay={bx}:{by}[v_after_blur]")
            cur = "v_after_blur"

        # 1b. Text-logo / watermark regions, blur mode (AI mode goes through the AI clean stream above)
        wm = cfg.wm_removal
        if wm.enabled and wm.mode == "blur":
            for i, b in enumerate(wm.boxes):
                bx, by, bw, bh = _clamp_box_pixels(b.x, b.y, b.w, b.h, vw, vh)
                rad = min(max(2, wm.blur_strength), max(2, min(bw, bh) // 4 - 1))
                vf.append(f"[{cur}]split=2[wmb{i}][wms{i}]")
                vf.append(
                    f"[wms{i}]crop={bw}:{bh}:{bx}:{by},boxblur=luma_radius={rad}:luma_power=2:"
                    f"chroma_radius={max(1, rad // 2)}:chroma_power=1[wmz{i}]"
                )
                vf.append(f"[wmb{i}][wmz{i}]overlay={bx}:{by}[v_after_wm{i}]")
                cur = f"v_after_wm{i}"

        # 2. Subtitle overlay
        if sub_input_idx is not None:
            sx, sy = sub_box_xy
            if gpu:
                sx, sy = _ev(sx), _ev(sy)
                vf.append(f"[{sub_input_idx}:v]format=yuva420p,hwupload_cuda[vsub]")
                vf.append(f"[{cur}][vsub]{ov}=x={sx}:y={sy}:eof_action=pass:shortest=0[v_after_sub]")
            else:
                vf.append(f"[{cur}][{sub_input_idx}:v]overlay=x={sx}:y={sy}:eof_action=pass:shortest=0[v_after_sub]")
            cur = "v_after_sub"

        # 3. Logo overlay (independent WxH)
        if logo_input_idx is not None:
            lx, ly, lw, lh = _clamp_box_pixels(cfg.logo.box.x, cfg.logo.box.y, cfg.logo.box.w, cfg.logo.box.h, vw, vh)
            op = max(0.0, min(1.0, cfg.logo.opacity))
            vf.append(
                f"[{logo_input_idx}:v]scale={lw}:{lh}:flags=lanczos,format=rgba,colorchannelmixer=aa={op:.3f}{to_gpu}[vlogo]"
            )
            if gpu:
                # ffmpeg 9 overlay_cuda bug: with eof_action=repeat (needed for a single-image logo) it outputs
                # frames padded to the 32-aligned pool size (1080 -> 1088, 540 -> 544). So the logo image is
                # looped (-loop 1) and overlaid with eof_action=pass + shortest=1 instead: same picture.
                vf.append(f"[{cur}][vlogo]{ov}=x={_ev(lx)}:y={_ev(ly)}:eof_action=pass:shortest=1[v_after_logo]")
            else:
                vf.append(f"[{cur}][vlogo]overlay=x={lx}:y={ly}:format=yuv420[v_after_logo]")
            cur = "v_after_logo"

        # 3b. Channel name text (endless input -> shortest=1 ends it with the video; eof_action=pass also avoids
        #     the overlay_cuda frame-size bug of eof_action=repeat)
        if ch_input_idx is not None:
            cx, cy = ch_xy
            if gpu:
                vf.append(f"[{ch_input_idx}:v]format=yuva420p,hwupload_cuda[vch]")
                vf.append(f"[{cur}][vch]{ov}=x={_ev(cx)}:y={_ev(cy)}:eof_action=pass:shortest=1[v_after_ch]")
            else:
                vf.append(f"[{cur}][{ch_input_idx}:v]overlay=x={cx}:y={cy}:eof_action=pass:shortest=1:format=yuv420[v_after_ch]")
            cur = "v_after_ch"

        # 4. Green-screen CTA overlay (chroma key + independent WxH)
        if gs_input_idx is not None:
            gx, gy, gw, gh = _clamp_box_pixels(
                cfg.green_screen.box.x, cfg.green_screen.box.y, cfg.green_screen.box.w, cfg.green_screen.box.h, vw, vh
            )
            ck_hex = _sanitize_hex(cfg.green_screen.chroma_color)
            sim = max(0.01, min(0.8, cfg.green_screen.similarity))
            bld = max(0.0, min(0.5, cfg.green_screen.blend))
            start_s = max(0.0, cfg.green_screen.start_sec)
            vf.append(
                f"[{gs_input_idx}:v]scale={gw}:{gh}:flags=lanczos,"
                f"chromakey=color={ck_hex}:similarity={sim:.3f}:blend={bld:.3f},"
                f"setpts=PTS-STARTPTS+{start_s:.3f}/TB{to_gpu}[vgs]"
            )
            eof_act = "repeat" if cfg.green_screen.loop_enabled else "pass"
            shortest = 0
            if gpu:
                gx, gy = _ev(gx), _ev(gy)
                # same overlay_cuda repeat bug as the logo: the looped CTA input never ends anyway
                eof_act, shortest = "pass", int(cfg.green_screen.loop_enabled)
            vf.append(
                f"[{cur}][vgs]{ov}=x={gx}:y={gy}:eof_action={eof_act}:shortest={shortest}:enable='gte(t,{start_s:.3f})'[v_after_gs]"
            )
            cur = "v_after_gs"
        return vf, cur

    # 5. Audio routing (Original volume/mute + assembled Narration/Dubbing WAV)
    filters: List[str] = []  # audio filters (video chains are built per variant above)
    audio_map_args: List[str] = []
    orig_vol = 0.0 if cfg.audio.mute_original else max(0.0, cfg.audio.original_volume)
    if voice_input_idx is not None:
        if voice_already_mixed or not project.has_audio or orig_vol <= 0.001 or source_video_override:
            # dub_mix.wav already has auto-ducked original audio mixed cleanly in Python/NumPy
            audio_map_args = ["-map", f"{voice_input_idx}:a", "-c:a", "aac", "-b:a", "192k"]
        else:
            filters.append(f"[0:a]volume={orig_vol:.3f}[a_orig]")
            filters.append(f"[a_orig][{voice_input_idx}:a]amix=inputs=2:duration=longest:dropout_transition=2[a_out]")
            audio_map_args = ["-map", "[a_out]", "-c:a", "aac", "-b:a", "192k"]
    else:
        if project.has_audio and not source_video_override and orig_vol > 0.001:
            if abs(orig_vol - 1.0) > 0.01:
                filters.append(f"[0:a]volume={orig_vol:.3f}[a_out]")
                audio_map_args = ["-map", "[a_out]", "-c:a", "aac", "-b:a", "192k"]
            else:
                audio_map_args = ["-map", "0:a?", "-c:a", "aac", "-b:a", "192k"]
        else:
            audio_map_args = ["-an"]

    encoder = _pick_encoder(cfg)
    suffix = "_preview.mp4" if preview_seconds else "_edited.mp4"
    safe_stem = re.sub(r'[<>:"/\\|?*]+', "_", Path(project.name).stem)
    out_path = OUTPUTS_DIR / f"{safe_stem}_{project.id[:6]}{suffix}"

    hw = detect_hardware()
    use_gpu_graph = (
        GPU_GRAPH_ENABLED
        and encoder == "h264_nvenc"
        and cfg.sub_removal.mode != "blur"
        and not (cfg.wm_removal.enabled and cfg.wm_removal.mode == "blur")
        and bool(hw.get("encoders", {}).get("h264_nvenc"))
    )

    def _build_cmd(gpu: bool, enc: str) -> List[str]:
        vf, cur_v = _video_chain(gpu)
        inp = list(inputs)
        if gpu:
            if logo_at >= 0:
                inp[logo_at:logo_at] = ["-loop", "1"]  # before the hwaccel insert (smaller index) shifts it
            inp[hwaccel_at:hwaccel_at] = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        c = [ffmpeg] + inp
        if vf or filters:
            c.extend(["-filter_complex", ";".join(vf + filters)])
        # Only filter outputs take brackets: an audio-only filter graph (e.g. original volume 15%, no subtitle /
        # logo / sub removal) leaves the video at "0:v", and "-map [0:v]" made ffmpeg abort.
        c.extend(["-map", "0:v" if cur_v == "0:v" else f"[{cur_v}]"])
        c.extend(audio_map_args)
        c.extend(_encoder_args(enc, gpu_frames=gpu))
        if preview_seconds and preview_seconds > 0:
            c.extend(["-t", f"{total_dur:.3f}"])
        c.extend(["-movflags", "+faststart", str(out_path)])
        return c

    cmd = _build_cmd(use_gpu_graph, encoder)

    if on_progress:
        on_progress(12.0, f"Đang xuất video bằng {encoder}...")

    def _execute(run_cmd: List[str], enc_name: str) -> tuple[int, str]:
        stream = make_clean_stream(project, preview_seconds) if use_ai_clean else None
        proc = subprocess.Popen(
            run_cmd,
            stdin=subprocess.PIPE if stream else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        register_proc(proc)
        err_chunks: List[bytes] = []

        def _drain():
            for chunk in iter(lambda: proc.stderr.read(4096), b""):
                err_chunks.append(chunk)
                if len(err_chunks) > 64:
                    del err_chunks[:-32]

        threading.Thread(target=_drain, daemon=True).start()

        feeder: Optional[threading.Thread] = None
        if stream is not None:
            def _feed():
                try:
                    stream.run(proc.stdin.write)
                except BaseException:  # noqa: BLE001 (kept in stream.error)
                    if proc.poll() is None:
                        try:
                            proc.kill()  # never let ffmpeg finish with un-cleaned frames
                        except Exception:
                            pass
                finally:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass

            # copy_context: the feeder (and the ffmpeg reader it starts) must see the job's project id so
            # "Dừng" can flag/kill it directly.
            feeder = threading.Thread(target=contextvars.copy_context().run, args=(_feed,), daemon=True)
            feeder.start()

        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", "replace").strip()
            if line.startswith("out_time_us="):
                val = line.split("=", 1)[1]
                if val.isdigit() and total_dur > 0:
                    sec = int(val) / 1_000_000.0
                    pct = min(98.0, 12.0 + (sec / total_dur) * 86.0)
                    if on_progress:
                        extra = f" · {stream.status_text()}" if stream else ""
                        on_progress(round(pct, 1), f"Đang render ({enc_name}): {sec:.1f}s / {total_dur:.1f}s{extra}")
        ret_code = proc.wait()
        if feeder is not None:
            feeder.join(timeout=30)
        unregister_proc(proc)
        check_cancel()
        if stream is not None and stream.error is not None:
            raise RuntimeError(f"AI xóa sub lỗi: {stream.error}")
        return ret_code, b"".join(err_chunks).decode("utf-8", "replace")

    def _timed(run_cmd: List[str], enc_name: str, label: str) -> tuple[int, str]:
        """_execute + a stopwatch step whose note is the speed reached (fps, x real time)."""
        st = timing.begin_step(project.id, "render", label)
        t0 = time.time()
        try:
            r = _execute(run_cmd, enc_name)
        except BaseException:
            timing.end_step(st, ok=False)
            raise
        dt = max(0.001, time.time() - t0)
        fps = total_dur * max(1.0, float(project.fps or 30.0)) / dt
        timing.end_step(st, ok=r[0] == 0, note=f"{enc_name} · {fps:.0f} fps · {total_dur / dt:.1f}× thời gian thực")
        return r

    main_label = "Xóa sub AI + xuất video" if use_ai_clean else "Xuất video"
    ret, stderr_out = _timed(
        cmd, encoder if not use_gpu_graph else f"{encoder} + NVDEC/CUDA", main_label + (" (GPU)" if use_gpu_graph else "")
    )
    if ret == 0 and use_gpu_graph and not _output_size_ok(out_path, vw, vh):
        ret, stderr_out = 1, "GPU graph produced a wrong frame size"  # safety net -> CPU graph below

    # Fallbacks (OCR results are cached -> no re-scan): GPU graph -> CPU graph (same encoder) -> libx264.
    # The GPU graph fails e.g. for codecs NVDEC cannot decode (frames then arrive in system memory).
    if ret != 0 and use_gpu_graph:
        if on_progress:
            on_progress(14.0, "Đồ họa GPU (NVDEC/CUDA) lỗi với video này, chuyển sang lọc bằng CPU...")
        ret, stderr_out = _timed(_build_cmd(False, encoder), encoder, main_label + " (dự phòng: lọc bằng CPU)")
    if ret != 0 and encoder != "libx264":
        if on_progress:
            on_progress(15.0, f"{encoder} gặp lỗi driver, đang chuyển dự phòng sang libx264...")
        ret2, err2 = _timed(_build_cmd(False, "libx264"), "libx264", main_label + " (dự phòng: libx264)")
        if ret2 != 0:
            raise RuntimeError(f"FFmpeg thất bại ({encoder} & libx264): {err2.strip()[-600:]}")
    elif ret != 0:
        raise RuntimeError(f"FFmpeg thất bại: {stderr_out.strip()[-600:]}")

    shutil.rmtree(work_dir, ignore_errors=True)
    if on_progress:
        on_progress(100.0, f"Hoàn tất! Đã lưu tại {out_path.name}")
    return out_path


async def render_project_async(
    project: VideoProject,
    preview_seconds: Optional[float] = None,
    voice_wav_path: Optional[str] = None,
    source_video_override: Optional[str] = None,
    voice_already_mixed: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    ai_clean: bool = True,
) -> Path:
    return await asyncio.to_thread(
        render_project_sync,
        project,
        preview_seconds,
        voice_wav_path,
        source_video_override,
        voice_already_mixed,
        on_progress,
        ai_clean,
    )
