from __future__ import annotations
import asyncio
import contextvars
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, List, Optional

from app.config import OUTPUTS_DIR, TEMP_DIR, get_ffmpeg_path
from app.models import EditConfig, SubtitleCue, VideoProject
from app.services.cancel import check_cancel, register_proc, unregister_proc
from app.services.hardware import detect_hardware
from app.services.media import parse_srt
from app.services.sub_remover import AICleanStream
from app.services.subtitle_renderer import build_subtitle_concat_sequence


ProgressCallback = Callable[[float, str], None]


def _pick_encoder(cfg: EditConfig) -> str:
    hw = detect_hardware()
    if cfg.encoder_override != "auto":
        if hw["encoders"].get(cfg.encoder_override, False):
            return cfg.encoder_override
    return hw["recommended_encoder"]


def _encoder_args(encoder: str) -> List[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "22", "-pix_fmt", "yuv420p"]
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "faster", "-global_quality", "23", "-pix_fmt", "nv12"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]


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
            concat_txt, sx, sy, _, _ = build_subtitle_concat_sequence(
                cues=cues,
                cfg=cfg.subtitle,
                video_w=vw,
                video_h=vh,
                video_duration=total_dur,
                work_dir=work_dir / "subs",
            )
            inputs.extend(["-f", "concat", "-safe", "0", "-i", str(concat_txt)])
            sub_input_idx = next_input_idx
            sub_box_xy = (sx, sy)
            next_input_idx += 1

    # Logo input
    logo_input_idx: Optional[int] = None
    if cfg.logo.enabled and cfg.logo.file_path and Path(cfg.logo.file_path).exists():
        inputs.extend(["-i", cfg.logo.file_path])
        logo_input_idx = next_input_idx
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
        bgm_mix = build_standalone_audio_mix_sync(project, total_dur)
        if bgm_mix and bgm_mix.exists():
            voice_wav_path = str(bgm_mix)
            voice_already_mixed = True

    if voice_wav_path and Path(voice_wav_path).exists():
        inputs.extend(["-i", voice_wav_path])
        voice_input_idx = next_input_idx
        next_input_idx += 1

    # AI Clean Sub strips, produced by Python while this same ffmpeg encodes (one decode + one encode)
    use_ai_clean = ai_clean and cfg.sub_removal.mode == "ai_clean" and not source_video_override
    ai_stream_idx: Optional[int] = None
    if use_ai_clean:
        if on_progress:
            on_progress(6.0, "Đang khởi động AI xóa sub (OCR CPU + GPU)...")
        probe = AICleanStream(project, preview_seconds)
        inputs.extend(probe.ffmpeg_input_args())
        ai_stream_idx = next_input_idx
        next_input_idx += 1

    # Build filter_complex chain
    filters: List[str] = []
    cur_v = "0:v"

    # 0. AI Clean Sub overlay (cleaned ROI strip on top of the original frame, frame-exact)
    if ai_stream_idx is not None:
        filters.extend(probe.overlay_filters(cur_v, ai_stream_idx, "v_after_clean"))
        cur_v = "v_after_clean"

    # 1. Sub removal (Blur box mode)
    if cfg.sub_removal.mode == "blur":
        bx, by, bw, bh = _clamp_box_pixels(
            cfg.sub_removal.box.x,
            cfg.sub_removal.box.y,
            cfg.sub_removal.box.w,
            cfg.sub_removal.box.h,
            vw,
            vh,
        )
        # Ensure boxblur radius doesn't exceed chroma plane limits (min(bw, bh)/4)
        max_rad = max(2, min(bw, bh) // 4 - 1)
        rad = min(max(2, cfg.sub_removal.blur_strength), max_rad)
        filters.append(f"[{cur_v}]split=2[vbase][vblursrc]")
        filters.append(
            f"[vblursrc]crop={bw}:{bh}:{bx}:{by},boxblur=luma_radius={rad}:luma_power=2:chroma_radius={max(1, rad // 2)}:chroma_power=1[vblurred]"
        )
        filters.append(f"[vbase][vblurred]overlay={bx}:{by}[v_after_blur]")
        cur_v = "v_after_blur"

    # 2. Subtitle overlay
    if sub_input_idx is not None:
        sx, sy = sub_box_xy
        filters.append(
            f"[{cur_v}][{sub_input_idx}:v]overlay=x={sx}:y={sy}:eof_action=pass:shortest=0[v_after_sub]"
        )
        cur_v = "v_after_sub"

    # 3. Logo overlay (independent WxH)
    if logo_input_idx is not None:
        lx, ly, lw, lh = _clamp_box_pixels(
            cfg.logo.box.x, cfg.logo.box.y, cfg.logo.box.w, cfg.logo.box.h, vw, vh
        )
        op = max(0.0, min(1.0, cfg.logo.opacity))
        filters.append(
            f"[{logo_input_idx}:v]scale={lw}:{lh}:flags=lanczos,format=rgba,colorchannelmixer=aa={op:.3f}[vlogo]"
        )
        filters.append(f"[{cur_v}][vlogo]overlay=x={lx}:y={ly}:format=auto[v_after_logo]")
        cur_v = "v_after_logo"

    # 4. Green-screen CTA overlay (chroma key + independent WxH)
    if gs_input_idx is not None:
        gx, gy, gw, gh = _clamp_box_pixels(
            cfg.green_screen.box.x,
            cfg.green_screen.box.y,
            cfg.green_screen.box.w,
            cfg.green_screen.box.h,
            vw,
            vh,
        )
        ck_hex = _sanitize_hex(cfg.green_screen.chroma_color)
        sim = max(0.01, min(0.8, cfg.green_screen.similarity))
        bld = max(0.0, min(0.5, cfg.green_screen.blend))
        start_s = max(0.0, cfg.green_screen.start_sec)
        gs_chain = (
            f"[{gs_input_idx}:v]scale={gw}:{gh}:flags=lanczos,"
            f"chromakey=color={ck_hex}:similarity={sim:.3f}:blend={bld:.3f},"
            f"setpts=PTS-STARTPTS+{start_s:.3f}/TB[vgs]"
        )
        filters.append(gs_chain)
        eof_act = "repeat" if cfg.green_screen.loop_enabled else "pass"
        filters.append(
            f"[{cur_v}][vgs]overlay=x={gx}:y={gy}:eof_action={eof_act}:shortest=0:enable='gte(t,{start_s:.3f})'[v_after_gs]"
        )
        cur_v = "v_after_gs"

    # 5. Audio routing (Original volume/mute + assembled Narration/Dubbing WAV)
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

    cmd = [ffmpeg] + inputs
    if filters:
        cmd.extend(["-filter_complex", ";".join(filters)])
        cmd.extend(["-map", f"[{cur_v}]"])
    else:
        cmd.extend(["-map", "0:v"])

    cmd.extend(audio_map_args)
    cmd.extend(_encoder_args(encoder))
    if preview_seconds and preview_seconds > 0:
        cmd.extend(["-t", f"{total_dur:.3f}"])
    cmd.extend(["-movflags", "+faststart", str(out_path)])

    if on_progress:
        on_progress(12.0, f"Đang xuất video bằng {encoder}...")

    def _execute(run_cmd: List[str], enc_name: str) -> tuple[int, str]:
        stream = AICleanStream(project, preview_seconds) if use_ai_clean else None
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

    ret, stderr_out = _execute(cmd, encoder)

    # Automatic safe fallback to libx264 if a hardware driver hiccup occurs
    if ret != 0 and encoder != "libx264":
        if on_progress:
            on_progress(15.0, f"{encoder} gặp lỗi driver, đang chuyển dự phòng sang libx264...")
        fallback_cmd = cmd.copy()
        enc_idx = fallback_cmd.index("-c:v")
        # Replace encoder args slice up to -t or -movflags
        end_idx = fallback_cmd.index("-movflags")
        if "-t" in fallback_cmd[enc_idx:end_idx]:
            end_idx = fallback_cmd.index("-t", enc_idx)
        fallback_cmd[enc_idx:end_idx] = _encoder_args("libx264")
        ret2, err2 = _execute(fallback_cmd, "libx264")  # OCR results are cached -> no re-scan
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
