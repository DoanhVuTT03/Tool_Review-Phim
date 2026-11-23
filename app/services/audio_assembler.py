from __future__ import annotations
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from app.config import PROJECTS_DIR, get_ffmpeg_path
from app.models import AudioConfig, BackgroundMusicConfig, VideoProject


ProgressCallback = Callable[[float, str], None]

SAMPLE_RATE = 44100


def _decode_to_mono_f32(audio_path: str, sr: int = SAMPLE_RATE, extra_af: str = "") -> np.ndarray:
    """Decodes any audio/video file to 1D float32 numpy array at `sr` Hz via FFmpeg."""
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", audio_path]
    if extra_af:
        cmd.extend(["-af", extra_af])
    cmd.extend(["-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"])
    res = subprocess.run(
        cmd,
        capture_output=True,
        timeout=90,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if res.returncode != 0 or not res.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def _trim_silence(pcm: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = -38.0, pad_ms: int = 25) -> np.ndarray:
    """Removes leading and trailing silence cleanly with tiny soft fade so speech starts on beat."""
    if pcm.size == 0:
        return pcm
    thresh = 10.0 ** (threshold_db / 20.0)
    idx = np.where(np.abs(pcm) > thresh)[0]
    if idx.size == 0:
        return pcm
    pad = int(sr * pad_ms / 1000)
    start = max(0, int(idx[0]) - pad)
    end = min(pcm.size, int(idx[-1]) + pad)
    trimmed = pcm[start:end].copy()

    fade_n = min(int(sr * 0.008), trimmed.size // 4)
    if fade_n > 1:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        trimmed[:fade_n] *= ramp
        trimmed[-fade_n:] *= ramp[::-1]
    return trimmed


def _time_stretch_pcm(audio_path: str, tempo: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Applies pitch-preserving time stretch via FFmpeg librubberband (with atempo fallback)."""
    t = max(0.5, min(2.5, tempo))
    pcm = _decode_to_mono_f32(audio_path, sr=sr, extra_af=f"rubberband=tempo={t:.4f}")
    if pcm.size == 0:
        chain = []
        rem = tempo
        while rem > 2.0:
            chain.append("atempo=2.0")
            rem /= 2.0
        while rem < 0.5:
            chain.append("atempo=0.5")
            rem /= 0.5
        chain.append(f"atempo={rem:.4f}")
        pcm = _decode_to_mono_f32(audio_path, sr=sr, extra_af=",".join(chain))
    return _trim_silence(pcm, sr=sr)


def _build_ducking_envelope(
    total_samples: int,
    speech_intervals: List[Tuple[int, int]],
    base_vol: float,
    duck_vol: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Smooth attack (100ms) & release (240ms) auto-ducking envelope."""
    env = np.full(total_samples, base_vol, dtype=np.float32)
    attack = int(sr * 0.10)
    release = int(sr * 0.24)
    target = min(base_vol, duck_vol)

    for s, e in speech_intervals:
        s = max(0, min(total_samples - 1, s))
        e = max(s, min(total_samples, e))
        env[s:e] = np.minimum(env[s:e], target)
        a0 = max(0, s - attack)
        if s > a0:
            ramp_down = np.linspace(base_vol, target, s - a0, dtype=np.float32)
            env[a0:s] = np.minimum(env[a0:s], ramp_down)
        r1 = min(total_samples, e + release)
        if r1 > e:
            ramp_up = np.linspace(target, base_vol, r1 - e, dtype=np.float32)
            env[e:r1] = np.minimum(env[e:r1], ramp_up)
    return env


def build_crossfaded_bgm_pcm(
    bgm_cfg: BackgroundMusicConfig,
    total_samples: int,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Builds a seamless looped Background Music track covering `total_samples`.
    - 1 track: smoothly loops with equal-power cosine crossfade (tail fade-out overlapped with head fade-in).
    - N tracks: plays Track 1 -> crossfade -> Track 2 -> ... -> Track N -> crossfade -> Track 1 repeating seamlessly.
    - Applies smooth fade-in at video start and fade-out at video end.
    """
    valid_paths = [p for p in (bgm_cfg.file_paths or []) if p and Path(p).exists()]
    if not bgm_cfg.enabled or not valid_paths or total_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    tracks: List[np.ndarray] = []
    for fp in valid_paths:
        arr = _decode_to_mono_f32(fp, sr=sr)
        if arr.size > sr * 0.5:
            # Normalize each track gently so playlist volume stays balanced across songs
            rms = float(np.sqrt(np.mean(arr**2)) + 1e-6)
            arr = arr * min(2.0, 0.14 / rms)
            tracks.append(arr.astype(np.float32))
    if not tracks:
        return np.zeros(0, dtype=np.float32)

    out = np.zeros(total_samples, dtype=np.float32)
    req_xfade = int(round(max(0.2, bgm_cfg.crossfade_sec) * sr))

    pos = 0
    track_idx = 0
    is_first = True

    while pos < total_samples:
        trk = tracks[track_idx % len(tracks)]
        track_idx += 1

        if is_first:
            copy_n = min(trk.size, total_samples - pos)
            out[pos : pos + copy_n] = trk[:copy_n]
            pos += copy_n
            is_first = False
            continue

        # Equal-power cosine crossfade (fade-out tail of previous + fade-in head of next)
        xfade = min(req_xfade, trk.size // 3, pos)
        if xfade > 16:
            overlap_start = pos - xfade
            theta = np.linspace(0.0, np.pi * 0.5, xfade, dtype=np.float32)
            fade_out = np.cos(theta)  # 1 -> 0
            fade_in = np.sin(theta)   # 0 -> 1

            out[overlap_start:pos] = (
                out[overlap_start:pos] * fade_out + trk[:xfade] * fade_in
            )
            rem_trk = trk[xfade:]
        else:
            rem_trk = trk

        if rem_trk.size > 0 and pos < total_samples:
            copy_n = min(rem_trk.size, total_samples - pos)
            out[pos : pos + copy_n] = rem_trk[:copy_n]
            pos += copy_n

    # Global smooth intro fade-in & outro fade-out
    fin_n = min(int(round(bgm_cfg.fade_in_sec * sr)), total_samples // 4)
    if fin_n > 1:
        out[:fin_n] *= np.linspace(0.0, 1.0, fin_n, dtype=np.float32)
    fout_n = min(int(round(bgm_cfg.fade_out_sec * sr)), total_samples // 4)
    if fout_n > 1:
        out[-fout_n:] *= np.linspace(1.0, 0.0, fout_n, dtype=np.float32)

    return out


def build_standalone_audio_mix_sync(
    project: VideoProject,
    total_dur: float,
) -> Optional[Path]:
    """Builds a standalone mixed WAV (Original Audio + Crossfaded BGM) when rendering WITHOUT Voice AI."""
    bgm_cfg = project.config.bgm
    if not bgm_cfg.enabled or not bgm_cfg.file_paths:
        return None
    total_samples = int(np.ceil(max(0.5, total_dur) * SAMPLE_RATE))
    bgm_pcm = build_crossfaded_bgm_pcm(bgm_cfg, total_samples, sr=SAMPLE_RATE)
    if bgm_pcm.size == 0:
        return None

    mix = bgm_pcm * float(bgm_cfg.volume)
    audio_cfg = project.config.audio
    orig_vol = 0.0 if audio_cfg.mute_original else max(0.0, float(audio_cfg.original_volume))
    if project.has_audio and orig_vol > 0.001 and Path(project.video_path).exists():
        orig = _decode_to_mono_f32(project.video_path, sr=SAMPLE_RATE)
        n = min(total_samples, orig.size)
        if n > 0:
            mix[:n] += orig[:n] * orig_vol

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.96:
        mix = np.tanh(mix / max(1.0, peak) * 1.1) * 0.94

    out_dir = PROJECTS_DIR / project.id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "bgm_only_mix.wav"
    sf.write(str(out_wav), np.stack([mix, mix], axis=-1), SAMPLE_RATE, subtype="PCM_16")
    return out_wav


def assemble_project_audio_sync(
    project: VideoProject,
    manifest_path: str,
    on_progress: Optional[ProgressCallback] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Assembles TTS clips + Original Audio + Crossfaded Background Music Playlist into 1 unified stereo WAV."""
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise RuntimeError("Không tìm thấy voice_manifest.json. Hãy tạo voice ở Bước 3 trước.")

    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = manifest.get("items", [])
    if not items:
        raise RuntimeError("Danh sách voice rỗng.")

    cfg: AudioConfig = project.config.audio
    bgm_cfg: BackgroundMusicConfig = project.config.bgm
    total_dur = max(project.duration, max(i["end_sec"] for i in items) + 0.5)
    total_samples = int(np.ceil(total_dur * SAMPLE_RATE))

    if on_progress:
        on_progress(10.0, f"Đang giải mã & cắt khoảng lặng cho {len(items)} file voice...")

    def _prep_item(idx: int) -> Tuple[int, np.ndarray, float]:
        item = items[idx]
        ap = item.get("audio_path", "")
        if not ap or not Path(ap).exists():
            return idx, np.zeros(0, dtype=np.float32), 0.0
        raw = _decode_to_mono_f32(ap, sr=SAMPLE_RATE)
        raw_dur = raw.size / SAMPLE_RATE
        trimmed = _trim_silence(raw, sr=SAMPLE_RATE)
        saved = max(0.0, raw_dur - (trimmed.size / SAMPLE_RATE))
        return idx, trimmed, saved

    decoded: Dict[int, np.ndarray] = {}
    total_saved_sec = 0.0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for idx, pcm, saved in pool.map(_prep_item, range(len(items))):
            decoded[idx] = pcm
            total_saved_sec += saved

    if on_progress:
        mode_vn = "Thuyết minh" if cfg.edit_mode == "narration" else "Lồng tiếng"
        on_progress(45.0, f"Đang căn chỉnh timing chế độ {mode_vn}...")

    voice_track = np.zeros(total_samples, dtype=np.float32)
    speech_intervals: List[Tuple[int, int]] = []
    tempos_used: List[float] = []
    tight_cues = 0
    cursor_end_sample = 0

    for i, item in enumerate(items):
        pcm = decoded.get(i, np.zeros(0, dtype=np.float32))
        if pcm.size == 0:
            continue

        start_s = float(item["start_sec"])
        end_s = float(item["end_sec"])
        slot_dur = max(0.25, end_s - start_s)
        clip_dur = pcm.size / SAMPLE_RATE

        if cfg.edit_mode == "narration":
            next_start = float(items[i + 1]["start_sec"]) - 0.08 if i + 1 < len(items) else total_dur
            allowed_dur = max(slot_dur, min(next_start - start_s, slot_dur + 1.8))
            max_speed = max(1.05, cfg.max_atempo)
        else:
            allowed_dur = slot_dur
            max_speed = max(1.35, cfg.max_atempo)

        tempo = 1.0
        if clip_dur > allowed_dur * 1.03:
            needed = clip_dur / max(0.2, allowed_dur)
            tempo = min(max_speed, needed)
            if tempo > 1.35:
                tight_cues += 1
            pcm = _time_stretch_pcm(item["audio_path"], tempo, sr=SAMPLE_RATE)
        tempos_used.append(tempo)

        start_sample = int(round(start_s * SAMPLE_RATE))
        if start_sample < cursor_end_sample:
            start_sample = cursor_end_sample + int(0.02 * SAMPLE_RATE)

        end_sample = min(total_samples, start_sample + pcm.size)
        if end_sample <= start_sample:
            continue

        seg = pcm[: end_sample - start_sample] * float(cfg.voice_gain)
        voice_track[start_sample:end_sample] += seg
        speech_intervals.append((start_sample, end_sample))
        cursor_end_sample = end_sample

    final_mix = voice_track.copy()

    # Mix original video audio
    orig_vol = 0.0 if cfg.mute_original else max(0.0, float(cfg.original_volume))
    if project.has_audio and orig_vol > 0.001 and Path(project.video_path).exists():
        if on_progress:
            on_progress(72.0, "Đang hòa âm với âm thanh gốc...")
        orig_pcm = _decode_to_mono_f32(project.video_path, sr=SAMPLE_RATE)
        if orig_pcm.size > 0:
            n = min(total_samples, orig_pcm.size)
            if cfg.edit_mode == "narration" and cfg.auto_ducking:
                env = _build_ducking_envelope(
                    n,
                    speech_intervals,
                    base_vol=orig_vol,
                    duck_vol=min(orig_vol, float(cfg.duck_level)),
                    sr=SAMPLE_RATE,
                )
                final_mix[:n] += orig_pcm[:n] * env
            else:
                final_mix[:n] += orig_pcm[:n] * orig_vol

    # Mix Crossfaded Background Music Playlist (with optional Auto-Ducking under voice!)
    bgm_used = False
    if bgm_cfg.enabled and bgm_cfg.file_paths:
        if on_progress:
            on_progress(85.0, f"Đang nối chéo nhạc nền ({len(bgm_cfg.file_paths)} bài, crossfade {bgm_cfg.crossfade_sec}s)...")
        bgm_pcm = build_crossfaded_bgm_pcm(bgm_cfg, total_samples, sr=SAMPLE_RATE)
        if bgm_pcm.size > 0:
            bgm_used = True
            if bgm_cfg.auto_ducking and speech_intervals:
                b_env = _build_ducking_envelope(
                    total_samples,
                    speech_intervals,
                    base_vol=float(bgm_cfg.volume),
                    duck_vol=min(float(bgm_cfg.volume), float(bgm_cfg.duck_level)),
                    sr=SAMPLE_RATE,
                )
                final_mix += bgm_pcm * b_env
            else:
                final_mix += bgm_pcm * float(bgm_cfg.volume)

    # Soft broadcast peak limiter
    peak = float(np.max(np.abs(final_mix))) if final_mix.size else 0.0
    if peak > 0.96:
        final_mix = np.tanh(final_mix / max(1.0, peak) * 1.1) * 0.94

    stereo = np.stack([final_mix, final_mix], axis=-1)
    out_wav = PROJECTS_DIR / project.id / f"dub_mix_{cfg.edit_mode}.wav"
    sf.write(str(out_wav), stereo, SAMPLE_RATE, subtype="PCM_16")

    stats = {
        "mode": cfg.edit_mode,
        "cues_assembled": len(speech_intervals),
        "avg_tempo": round(float(np.mean(tempos_used)) if tempos_used else 1.0, 3),
        "tight_cues": tight_cues,
        "silence_trimmed_sec": round(total_saved_sec, 2),
        "bgm_mixed": bgm_used,
        "wav_path": str(out_wav),
    }
    if on_progress:
        bgm_txt = " + Nhạc nền Crossfade" if bgm_used else ""
        on_progress(
            100.0,
            f"Ghép audio xong ({stats['cues_assembled']} câu{bgm_txt}, tốc độ TB {stats['avg_tempo']}x)",
        )
    return out_wav, stats
