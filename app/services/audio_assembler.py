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


def _decode_to_mono_f32(
    audio_path: str, sr: int = SAMPLE_RATE, extra_af: str = "", timeout: float = 600.0
) -> np.ndarray:
    """Decodes a (short) audio file to 1D float32 numpy array at `sr` Hz via FFmpeg. Long sources (the movie's
    own audio track) are streamed with _PcmStream instead: an 8 h track is 5 GB of float32."""
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", audio_path, "-vn"]
    if extra_af:
        cmd.extend(["-af", extra_af])
    cmd.extend(["-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"])
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return np.zeros(0, dtype=np.float32)
    if res.returncode != 0 or not res.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


class _PcmStream:
    """Streams a media file's audio as mono float32 blocks (constant memory, no timeout)."""

    def __init__(self, path: str, sr: int = SAMPLE_RATE) -> None:
        self.proc = subprocess.Popen(
            [get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path, "-vn",
             "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.eof = False

    def read(self, n: int) -> np.ndarray:
        """Exactly n samples (zero padded after the end of the audio)."""
        out = np.zeros(n, dtype=np.float32)
        if self.eof or n <= 0:
            return out
        want, got = n * 4, bytearray()
        while len(got) < want:
            chunk = self.proc.stdout.read(want - len(got))
            if not chunk:
                self.eof = True
                break
            got += chunk
        k = len(got) // 4
        if k:
            out[:k] = np.frombuffer(bytes(got[: k * 4]), dtype=np.float32)
        return out

    def close(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass


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


# ----------------------------------------------------------------------------------------------------------
# Block-wise (streaming) mixing
#
# The old code built every track for the whole video in RAM: an 8 h movie = 1.28e9 samples = 5.1 GB per
# float32 track, and voice + original + mix + ducking envelope + stereo copy needed > 30 GB (and the original
# audio decode had a 90 s timeout, after which the movie's own sound was silently dropped). Now the mix is
# produced in 30 s blocks with the exact same math (placement, ducking ramps, crossfades), and only the
# trimmed voice clips are kept (int16).
# ----------------------------------------------------------------------------------------------------------

_BLOCK_SEC = 30
_ATTACK_SEC = 0.10
_RELEASE_SEC = 0.24


def _soft_limit(x: np.ndarray) -> np.ndarray:
    """Per-sample soft limiter (block-safe). Leaves |x| <= 0.9 untouched and rounds peaks off below ~0.98.
    (The old limiter divided the WHOLE track by its global peak, which needs the full track in memory.)"""
    ax = np.abs(x)
    over = ax > 0.9
    if np.any(over):
        x = x.copy()
        x[over] = np.sign(x[over]) * (0.9 + 0.08 * np.tanh((ax[over] - 0.9) / 0.08))
    return x


class _DuckingEnvelope:
    """Same envelope as _build_ducking_envelope (100 ms attack, 240 ms release), computed per block."""

    def __init__(self, total: int, intervals: List[Tuple[int, int]], base: float, duck: float, sr: int = SAMPLE_RATE):
        self.total, self.base, self.target = total, float(base), float(min(base, duck))
        self.attack, self.release = int(sr * _ATTACK_SEC), int(sr * _RELEASE_SEC)
        self.iv = sorted(intervals)
        self.starts = [s for s, _ in self.iv]

    def block(self, a: int, b: int) -> np.ndarray:
        env = np.full(b - a, self.base, dtype=np.float32)
        if not self.iv or self.target >= self.base:
            return env
        import bisect

        # intervals that can touch [a, b): start < b + attack  and  end + release > a
        hi = bisect.bisect_left(self.starts, b + self.attack)
        lo = max(0, bisect.bisect_left(self.starts, a - self.release - 1) - 64)
        for s, e in self.iv[lo:hi]:
            s = max(0, min(self.total - 1, s))
            e = max(s, min(self.total, e))
            if e + self.release <= a or s - self.attack >= b:
                continue
            x0, x1 = max(a, s), min(b, e)
            if x1 > x0:
                env[x0 - a : x1 - a] = np.minimum(env[x0 - a : x1 - a], self.target)
            a0 = max(0, s - self.attack)
            if s > a0:
                r0, r1 = max(a, a0), min(b, s)
                if r1 > r0:
                    ramp = np.linspace(self.base, self.target, s - a0, dtype=np.float32)
                    env[r0 - a : r1 - a] = np.minimum(env[r0 - a : r1 - a], ramp[r0 - a0 : r1 - a0])
            e1 = min(self.total, e + self.release)
            if e1 > e:
                r0, r1 = max(a, e), min(b, e1)
                if r1 > r0:
                    ramp = np.linspace(self.target, self.base, e1 - e, dtype=np.float32)
                    env[r0 - a : r1 - a] = np.minimum(env[r0 - a : r1 - a], ramp[r0 - e : r1 - e])
        return env


class _BgmPlaylist:
    """Same output as build_crossfaded_bgm_pcm, rendered per block: the playlist is laid out once as
    (track, start sample, crossfade length) placements; a block sums the placements it overlaps."""

    def __init__(self, bgm_cfg: BackgroundMusicConfig, total: int, sr: int = SAMPLE_RATE):
        self.total = total
        self.tracks: List[np.ndarray] = []
        self.places: List[Tuple[int, int, int]] = []  # (track index, start, xfade-in length)
        valid = [p for p in (bgm_cfg.file_paths or []) if p and Path(p).exists()]
        if not bgm_cfg.enabled or not valid or total <= 0:
            return
        for fp in valid:
            arr = _decode_to_mono_f32(fp, sr=sr)
            if arr.size > sr * 0.5:
                rms = float(np.sqrt(np.mean(arr**2)) + 1e-6)
                self.tracks.append((arr * min(2.0, 0.14 / rms)).astype(np.float32))
        if not self.tracks:
            return
        req_xfade = int(round(max(0.2, bgm_cfg.crossfade_sec) * sr))
        pos, k = 0, 0
        while pos < total:
            ti = k % len(self.tracks)
            trk = self.tracks[ti]
            if k == 0:
                self.places.append((ti, 0, 0))
                pos = min(total, trk.size)
            else:
                xf = min(req_xfade, trk.size // 3, pos)
                xf = xf if xf > 16 else 0
                st = pos - xf
                self.places.append((ti, st, xf))
                pos = min(total, st + trk.size)
            k += 1
        self.fin = min(int(round(bgm_cfg.fade_in_sec * sr)), total // 4)
        self.fout = min(int(round(bgm_cfg.fade_out_sec * sr)), total // 4)

    @property
    def active(self) -> bool:
        return bool(self.places)

    def block(self, a: int, b: int) -> np.ndarray:
        out = np.zeros(b - a, dtype=np.float32)
        n = len(self.places)
        for i, (ti, st, xf) in enumerate(self.places):
            trk = self.tracks[ti]
            en = min(self.total, st + trk.size)
            if en <= a or st >= b:
                continue
            x0, x1 = max(a, st), min(b, en)
            seg = trk[x0 - st : x1 - st].copy()
            if xf:  # fade in over [st, st + xf)
                f1 = min(x1, st + xf)
                if f1 > x0:
                    theta = np.linspace(0.0, np.pi * 0.5, xf, dtype=np.float32)
                    seg[: f1 - x0] *= np.sin(theta[x0 - st : f1 - st])
            if i + 1 < n:  # fade out under the next placement's crossfade
                _, nst, nxf = self.places[i + 1]
                if nxf:
                    f0, f1 = max(x0, nst), min(x1, nst + nxf)
                    if f1 > f0:
                        theta = np.linspace(0.0, np.pi * 0.5, nxf, dtype=np.float32)
                        seg[f0 - x0 : f1 - x0] *= np.cos(theta[f0 - nst : f1 - nst])
            out[x0 - a : x1 - a] += seg
        if self.fin > 1 and a < self.fin:
            r1 = min(b, self.fin)
            out[: r1 - a] *= np.linspace(0.0, 1.0, self.fin, dtype=np.float32)[a:r1]
        if self.fout > 1 and b > self.total - self.fout:
            f0 = self.total - self.fout
            r0 = max(a, f0)
            out[r0 - a :] *= np.linspace(1.0, 0.0, self.fout, dtype=np.float32)[r0 - f0 : b - f0]
        return out


def _write_mix(
    out_wav: Path,
    total_samples: int,
    clips: List[Tuple[int, np.ndarray]],
    voice_gain: float,
    orig_path: Optional[str],
    orig_vol: float,
    orig_env: Optional[_DuckingEnvelope],
    bgm: Optional[_BgmPlaylist],
    bgm_vol: float,
    bgm_env: Optional[_DuckingEnvelope],
    on_progress: Optional[ProgressCallback] = None,
    pct0: float = 60.0,
    pct1: float = 98.0,
) -> None:
    """Streams the final stereo PCM_16 WAV block by block. clips: (start sample, int16 pcm), sorted, no overlap."""
    block = SAMPLE_RATE * _BLOCK_SEC
    orig = _PcmStream(orig_path) if orig_path else None
    ci = 0
    try:
        with sf.SoundFile(str(out_wav), mode="w", samplerate=SAMPLE_RATE, channels=2, subtype="PCM_16") as wf:
            for a in range(0, total_samples, block):
                b = min(total_samples, a + block)
                mix = np.zeros(b - a, dtype=np.float32)
                # voice clips (sorted by start; a clip may span two blocks)
                while ci < len(clips) and clips[ci][0] + clips[ci][1].size <= a:
                    ci += 1
                j = ci
                while j < len(clips) and clips[j][0] < b:
                    st, pcm = clips[j]
                    x0, x1 = max(a, st), min(b, st + pcm.size)
                    if x1 > x0:
                        mix[x0 - a : x1 - a] += pcm[x0 - st : x1 - st].astype(np.float32) * (voice_gain / 32767.0)
                    j += 1
                if orig is not None:
                    o = orig.read(b - a)
                    mix += o * (orig_env.block(a, b) if orig_env is not None else orig_vol)
                if bgm is not None and bgm.active:
                    mix += bgm.block(a, b) * (bgm_env.block(a, b) if bgm_env is not None else bgm_vol)
                mix = _soft_limit(mix)
                wf.write(np.stack([mix, mix], axis=-1))
                if on_progress and total_samples:
                    on_progress(round(pct0 + (pct1 - pct0) * b / total_samples, 1),
                                f"Đang trộn & ghi audio: {b / SAMPLE_RATE / 60:.0f}/{total_samples / SAMPLE_RATE / 60:.0f} phút...")
    finally:
        if orig is not None:
            orig.close()


def build_standalone_audio_mix_sync(
    project: VideoProject,
    total_dur: float,
) -> Optional[Path]:
    """Builds a standalone mixed WAV (Original Audio + Crossfaded BGM) when rendering WITHOUT Voice AI."""
    bgm_cfg = project.config.bgm
    if not bgm_cfg.enabled or not bgm_cfg.file_paths:
        return None
    total_samples = int(np.ceil(max(0.5, total_dur) * SAMPLE_RATE))
    bgm = _BgmPlaylist(bgm_cfg, total_samples)
    if not bgm.active:
        return None
    audio_cfg = project.config.audio
    orig_vol = 0.0 if audio_cfg.mute_original else max(0.0, float(audio_cfg.original_volume))
    use_orig = project.has_audio and orig_vol > 0.001 and Path(project.video_path).exists()

    out_dir = PROJECTS_DIR / project.id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "bgm_only_mix.wav"
    _write_mix(out_wav, total_samples, [], 0.0, project.video_path if use_orig else None, orig_vol, None,
               bgm, float(bgm_cfg.volume), None)
    return out_wav


def _prep_clip(item: Dict[str, Any], next_start: Optional[float], total_dur: float, cfg: AudioConfig) -> Dict[str, Any]:
    """Decode + trim one TTS clip and time-stretch it if it does not fit its slot (runs in a thread pool;
    the old code stretched clips one after another in the main loop: one ffmpeg run per tight line)."""
    ap = item.get("audio_path", "")
    res: Dict[str, Any] = {"pcm": None, "saved": 0.0, "tempo": 1.0, "tight": False}
    if not ap or not Path(ap).exists():
        return res
    raw = _decode_to_mono_f32(ap, sr=SAMPLE_RATE, timeout=120)
    trimmed = _trim_silence(raw, sr=SAMPLE_RATE)
    res["saved"] = max(0.0, (raw.size - trimmed.size) / SAMPLE_RATE)
    if trimmed.size == 0:
        return res
    start_s, end_s = float(item["start_sec"]), float(item["end_sec"])
    slot_dur = max(0.25, end_s - start_s)
    clip_dur = trimmed.size / SAMPLE_RATE
    if cfg.edit_mode == "narration":
        nxt = (next_start - 0.08) if next_start is not None else total_dur
        allowed = max(slot_dur, min(nxt - start_s, slot_dur + 1.8))
        max_speed = max(1.05, cfg.max_atempo)
    else:
        allowed = slot_dur
        max_speed = max(1.35, cfg.max_atempo)
    pcm = trimmed
    if clip_dur > allowed * 1.03:
        tempo = min(max_speed, clip_dur / max(0.2, allowed))
        res["tempo"] = tempo
        res["tight"] = tempo > 1.35
        stretched = _time_stretch_pcm(ap, tempo, sr=SAMPLE_RATE)
        if stretched.size:
            pcm = stretched
    res["pcm"] = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
    return res


def assemble_project_audio_sync(
    project: VideoProject,
    manifest_path: str,
    on_progress: Optional[ProgressCallback] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Assembles TTS clips + Original Audio + Crossfaded Background Music Playlist into 1 unified stereo WAV
    (streamed in 30 s blocks: constant memory whatever the video length)."""
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise RuntimeError("Không tìm thấy voice_manifest.json. Hãy tạo voice ở Bước 3 trước.")

    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = sorted(manifest.get("items", []), key=lambda it: float(it["start_sec"]))
    if not items:
        raise RuntimeError("Danh sách voice rỗng.")

    cfg: AudioConfig = project.config.audio
    bgm_cfg: BackgroundMusicConfig = project.config.bgm
    total_dur = max(project.duration, max(float(i["end_sec"]) for i in items) + 0.5)
    total_samples = int(np.ceil(total_dur * SAMPLE_RATE))

    if on_progress:
        on_progress(5.0, f"Đang giải mã, cắt khoảng lặng & co giãn {len(items)} file voice (song song)...")

    preps: List[Dict[str, Any]] = [None] * len(items)  # type: ignore[list-item]
    done = [0]

    def _job(i: int) -> Tuple[int, Dict[str, Any]]:
        nxt = float(items[i + 1]["start_sec"]) if i + 1 < len(items) else None
        return i, _prep_clip(items[i], nxt, total_dur, cfg)

    import os as _os

    with ThreadPoolExecutor(max_workers=max(4, min(16, (_os.cpu_count() or 8)))) as pool:
        for i, r in pool.map(_job, range(len(items))):
            preps[i] = r
            done[0] += 1
            if on_progress and (done[0] % 50 == 0 or done[0] == len(items)):
                on_progress(round(5.0 + 50.0 * done[0] / len(items), 1), f"Đã xử lý {done[0]}/{len(items)} file voice...")

    # Placement (sequential): never overlap the previous line
    clips: List[Tuple[int, np.ndarray]] = []
    speech_intervals: List[Tuple[int, int]] = []
    tempos_used: List[float] = []
    tight_cues = 0
    total_saved_sec = 0.0
    cursor_end = 0
    for item, pr in zip(items, preps):
        total_saved_sec += pr["saved"]
        pcm = pr["pcm"]
        if pcm is None or pcm.size == 0:
            continue
        tempos_used.append(pr["tempo"])
        tight_cues += int(pr["tight"])
        st = int(round(float(item["start_sec"]) * SAMPLE_RATE))
        if st < cursor_end:
            st = cursor_end + int(0.02 * SAMPLE_RATE)
        en = min(total_samples, st + pcm.size)
        if en <= st:
            continue
        clips.append((st, pcm[: en - st]))
        speech_intervals.append((st, en))
        cursor_end = en

    orig_vol = 0.0 if cfg.mute_original else max(0.0, float(cfg.original_volume))
    use_orig = project.has_audio and orig_vol > 0.001 and Path(project.video_path).exists()
    orig_env = (
        _DuckingEnvelope(total_samples, speech_intervals, orig_vol, min(orig_vol, float(cfg.duck_level)))
        if use_orig and cfg.edit_mode == "narration" and cfg.auto_ducking
        else None
    )
    bgm = None
    bgm_env = None
    if bgm_cfg.enabled and bgm_cfg.file_paths:
        if on_progress:
            on_progress(57.0, f"Đang chuẩn bị nhạc nền ({len(bgm_cfg.file_paths)} bài, crossfade {bgm_cfg.crossfade_sec}s)...")
        bgm = _BgmPlaylist(bgm_cfg, total_samples)
        if bgm.active and bgm_cfg.auto_ducking and speech_intervals:
            bgm_env = _DuckingEnvelope(total_samples, speech_intervals, float(bgm_cfg.volume),
                                       min(float(bgm_cfg.volume), float(bgm_cfg.duck_level)))

    out_wav = PROJECTS_DIR / project.id / f"dub_mix_{cfg.edit_mode}.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _write_mix(out_wav, total_samples, clips, float(cfg.voice_gain), project.video_path if use_orig else None,
               orig_vol, orig_env, bgm, float(bgm_cfg.volume), bgm_env, on_progress)

    stats = {
        "mode": cfg.edit_mode,
        "cues_assembled": len(speech_intervals),
        "avg_tempo": round(float(np.mean(tempos_used)) if tempos_used else 1.0, 3),
        "tight_cues": tight_cues,
        "silence_trimmed_sec": round(total_saved_sec, 2),
        "bgm_mixed": bool(bgm is not None and bgm.active),
        "wav_path": str(out_wav),
    }
    if on_progress:
        bgm_txt = " + Nhạc nền Crossfade" if stats["bgm_mixed"] else ""
        on_progress(
            100.0,
            f"Ghép audio xong ({stats['cues_assembled']} câu{bgm_txt}, tốc độ TB {stats['avg_tempo']}x)",
        )
    return out_wav, stats
