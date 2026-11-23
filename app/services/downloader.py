from __future__ import annotations
import asyncio
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import DOWNLOADS_DIR, get_aria2c_path, get_ffmpeg_path
from app.models import DownloadVideoRequest, SubtitleCue, TranslateConfig
from app.services.media import parse_srt, write_srt
from app.services.translator import invoke_llm, parse_translated_lines


ProgressCallback = Callable[[float, str], None]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ASS_TAG_RE = re.compile(r"\{\\[^}]*\}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NOISE_LINE_RE = re.compile(
    r"^[\s\[\(\{【（♪♫♬🎵🎶*\-_=~.。,，!！?？:;；/\\|]*(music|applause|laughter|âm nhạc|nhạc|vỗ tay|cười|音乐|掌声|笑声|bgm)?[\s\]\)\}】）♪♫♬🎵🎶*\-_=~.。,，!！?？:;；/\\|]*$",
    re.IGNORECASE,
)
_SPAM_KEYWORDS = (
    "subscribe",
    "like and subscribe",
    "đăng ký kênh",
    "nhớ like",
    "bấm chuông",
    "欢迎订阅",
    "订阅频道",
    "点赞关注",
    "一键三连",
    "字幕由",
    "字幕制作",
    "翻译校对",
    "后期制作",
    "关注不迷路",
)


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\r\n]+', "_", name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:100] or "downloaded_video"


def strip_ansi(text: str) -> str:
    """Strips terminal ANSI color escape sequences and unescaped brackets (e.g. [0;31m)."""
    if not text:
        return ""
    s = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]?", "", str(text))
    s = re.sub(r"\[[0-9;]+m", "", s)
    s = re.sub(r"^\s*ERROR:\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def format_download_error(ex: Exception | str) -> str:
    """Translates raw yt-dlp/aria2c error messages into clean, readable text."""
    raw = strip_ansi(str(ex))
    low = raw.lower()
    if "this video is unavailable" in low or "video unavailable" in low:
        return "Video này không khả dụng hoặc đã bị tác giả xóa/ẩn trên YouTube."
    if "private video" in low:
        return "Video này ở chế độ riêng tư (Private), bạn cần chọn Cookie trình duyệt để tải."
    if "sign in to confirm your age" in low:
        return "Video giới hạn độ tuổi người xem, bạn cần chọn Cookie trình duyệt để tải."
    if "bot" in low or "automated queries" in low:
        return "YouTube tạm thời yêu cầu xác thực hoặc giới hạn IP, vui lòng thử lại sau ít phút hoặc bật Cookie."
    return raw


def clean_srt_rules_sync(srt_path: str, out_path: Optional[str] = None) -> Tuple[Path, Dict[str, Any]]:
    """Fast deterministic SRT garbage cleaner:
    - Strips HTML/VTT/ASS tags and URLs
    - Removes empty, symbol-only, [Music]/[Applause], and common promo/watermark lines
    - Deduplicates rolling/repeated lines common in YouTube/BiliBili auto-captions
    - Fixes overlapping cue timestamps cleanly
    """
    cues = parse_srt(srt_path)
    before = len(cues)
    cleaned: List[SubtitleCue] = []
    prev_norm = ""

    for c in cues:
        txt = c.text
        txt = _ASS_TAG_RE.sub("", txt)
        txt = _HTML_TAG_RE.sub("", txt)
        txt = _URL_RE.sub("", txt)
        txt = txt.replace("\u200b", "").replace("\ufeff", "")
        # Clean per-line duplicates inside a single cue
        sub_lines: List[str] = []
        for ln in txt.splitlines():
            ln = re.sub(r"\s+", " ", ln).strip()
            if not ln:
                continue
            if _NOISE_LINE_RE.match(ln):
                continue
            low = ln.lower()
            if any(k in low for k in _SPAM_KEYWORDS) and len(ln) < 60:
                continue
            if not sub_lines or sub_lines[-1] != ln:
                sub_lines.append(ln)

        merged = " ".join(sub_lines).strip()
        if not merged:
            continue

        # Remove rolling duplicate prefix from previous cue (YouTube auto-sub overlap)
        norm = re.sub(r"\s+", " ", merged).lower()
        if prev_norm and norm == prev_norm and cleaned and (c.start_sec - cleaned[-1].end_sec) < 0.8:
            # Extend previous cue end time instead of duplicating line
            cleaned[-1].end_sec = max(cleaned[-1].end_sec, c.end_sec)
            continue
        if prev_norm and len(prev_norm) > 6 and norm.startswith(prev_norm):
            # Keep only the newly revealed suffix
            suffix = merged[len(cleaned[-1].text) :].strip(" ,.-—")
            if not suffix:
                cleaned[-1].end_sec = max(cleaned[-1].end_sec, c.end_sec)
                continue
            merged = suffix
            norm = re.sub(r"\s+", " ", merged).lower()

        if cleaned and c.start_sec < cleaned[-1].end_sec:
            cleaned[-1].end_sec = max(cleaned[-1].start_sec + 0.1, c.start_sec - 0.01)

        cleaned.append(
            SubtitleCue(
                index=len(cleaned) + 1,
                start_sec=c.start_sec,
                end_sec=max(c.end_sec, c.start_sec + 0.2),
                text=merged,
            )
        )
        prev_norm = norm

    target = Path(out_path) if out_path else Path(srt_path)
    write_srt(cleaned, str(target))
    return target, {
        "before_count": before,
        "after_count": len(cleaned),
        "removed_count": max(0, before - len(cleaned)),
        "path": str(target),
    }


async def clean_srt_with_antigravity(
    srt_path: str,
    model: str = "gemini-3.8-flash",
    thinking_level: str = "low",
    api_key: str = "",
    on_progress: Optional[ProgressCallback] = None,
    llm_caller=invoke_llm,
) -> Tuple[Path, Dict[str, Any]]:
    """Uses selected Antigravity model to remove spam/watermark/ASR garbage lines & fix typos in original SRT
    WITHOUT translating or altering timestamps."""
    cues = parse_srt(srt_path)
    if not cues:
        return Path(srt_path), {"before_count": 0, "after_count": 0, "removed_count": 0}

    cfg = TranslateConfig(
        provider="antigravity",
        model=model,
        thinking_level=thinking_level,  # type: ignore[arg-type]
        api_key=api_key,
    )
    system_prompt = (
        "Bạn là chuyên gia biên tập và làm sạch file phụ đề SRT gốc (KHÔNG DỊCH SANG NGÔN NGỮ KHÁC).\n"
        "Nhiệm vụ:\n"
        "1. GIỮ NGUYÊN ngôn ngữ gốc của phụ đề.\n"
        "2. Với các dòng rác hoàn toàn (quảng cáo kênh, kêu gọi like/sub/đăng ký, credit team sub, tên website, âm thanh [Nhạc]/[Vỗ tay], ký tự nhận dạng giọng nói vô nghĩa): trả về '<số>|||DELETE'.\n"
        "3. Với các dòng lời thoại hợp lệ: giữ nguyên ý, sửa nhẹ lỗi chính tả ASR/OCR hoặc bỏ ký tự thừa nếu có, trả về '<số>|||<nội dung đã làm sạch>'.\n"
        "4. TUYỆT ĐỐI chỉ trả về đúng định dạng '<số>|||<nội dung hoặc DELETE>' cho mỗi dòng, không giải thích, không markdown."
    )

    batch_size = 80
    batches = [cues[i : i + batch_size] for i in range(0, len(cues), batch_size)]
    cleaned_map: Dict[int, str] = {}

    for b_idx, batch in enumerate(batches, start=1):
        if on_progress:
            pct = round(10.0 + ((b_idx - 1) / max(1, len(batches))) * 85.0, 1)
            on_progress(pct, f"Antigravity ({model}) đang dọn rác SRT lô {b_idx}/{len(batches)}...")
        block = "\n".join(f"{c.index}|||{c.text.replace(chr(10), ' ')}" for c in batch)
        user_prompt = f"[TEXT_START]\n{block}\n[TEXT_END]"
        raw = await llm_caller(cfg, system_prompt, user_prompt)
        parsed = parse_translated_lines(raw)
        for c in batch:
            cleaned_map[c.index] = parsed.get(c.index, c.text)

    out_cues: List[SubtitleCue] = []
    removed = 0
    for c in cues:
        val = cleaned_map.get(c.index, c.text).strip()
        if not val or val.upper() == "DELETE":
            removed += 1
            continue
        out_cues.append(
            SubtitleCue(index=len(out_cues) + 1, start_sec=c.start_sec, end_sec=c.end_sec, text=val)
        )

    out_path = Path(srt_path).with_name(f"{Path(srt_path).stem}_clean.srt")
    write_srt(out_cues, str(out_path))
    if on_progress:
        on_progress(100.0, f"AI đã dọn sạch {removed} dòng rác! Còn {len(out_cues)}/{len(cues)} dòng chuẩn.")
    return out_path, {
        "before_count": len(cues),
        "after_count": len(out_cues),
        "removed_count": removed,
        "path": str(out_path),
    }


def _match_sub_lang_candidates(preferred: str, available_keys: List[str]) -> List[str]:
    if not available_keys:
        return []
    pref = preferred.strip().lower()
    if pref in ("", "auto"):
        # Prioritize Chinese / Vietnamese / English / Filipino
        order = ["zh-hans", "zh-cn", "zh", "ai-zh", "zh-hant", "zh-tw", "vi", "en", "fil", "tl"]
        for k in order:
            for ak in available_keys:
                if ak.lower() == k or ak.lower().startswith(k + "-"):
                    return [ak]
        return [available_keys[0]]

    alias_map = {
        "zh": ["zh-hans", "zh-cn", "zh", "ai-zh", "zh-hant", "zh-tw", "zh-hk"],
        "fil": ["fil", "tl", "fil-ph", "tl-ph"],
        "tl": ["fil", "tl", "fil-ph", "tl-ph"],
        "vi": ["vi", "vi-vn"],
        "en": ["en", "en-us", "en-gb", "en-orig"],
        "ja": ["ja", "ja-jp"],
        "ko": ["ko", "ko-kr"],
    }
    targets = alias_map.get(pref, [pref])
    for t in targets:
        for ak in available_keys:
            if ak.lower() == t:
                return [ak]
    for t in targets:
        for ak in available_keys:
            if ak.lower().startswith(t):
                return [ak]
    return [available_keys[0]]


def _download_sync(
    req: DownloadVideoRequest,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """High-speed multi-threaded YouTube/BiliBili downloader using yt-dlp + aria2c + concurrent fragments."""
    import yt_dlp  # type: ignore

    url = req.url.strip()
    if not url:
        raise RuntimeError("Vui lòng nhập link YouTube hoặc BiliBili.")

    ffmpeg_dir = str(Path(get_ffmpeg_path()).parent)
    aria2_path = get_aria2c_path()
    threads = max(1, min(32, int(req.threads)))
    # aria2c strictly limits -x (--max-connection-per-server) to 1..16. Passing > 16 triggers code 28 OPTION_ERROR.
    aria_conn = min(16, max(1, threads))
    aria_split = min(16, max(1, threads))

    if on_progress:
        on_progress(5.0, "Đang phân tích link video & danh sách phụ đề có sẵn...")

    probe_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    if req.browser_cookies != "none":
        probe_opts["cookiesfrombrowser"] = (req.browser_cookies,)

    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise RuntimeError("Không thể đọc thông tin video từ link này.")
    except Exception as ex:
        raise RuntimeError(format_download_error(ex)) from ex

    title = req.custom_name.strip() or info.get("title") or "video"
    safe_stem = _safe_filename(title)
    out_dir = DOWNLOADS_DIR / safe_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_subs = list((info.get("subtitles") or {}).keys())
    auto_subs = list((info.get("automatic_captions") or {}).keys())
    chosen_lang: List[str] = []
    use_auto = False

    if req.sub_lang != "auto":
        m_cand = _match_sub_lang_candidates(req.sub_lang, manual_subs)
        if m_cand and (m_cand[0].lower().startswith(req.sub_lang.lower()) or req.sub_lang.lower().startswith(m_cand[0].lower()[:2])):
            chosen_lang = m_cand
            use_auto = False
        else:
            a_cand = _match_sub_lang_candidates(req.sub_lang, auto_subs)
            if a_cand and (a_cand[0].lower().startswith(req.sub_lang.lower()) or req.sub_lang.lower().startswith(a_cand[0].lower()[:2])):
                chosen_lang = a_cand
                use_auto = True
            elif manual_subs:
                chosen_lang = [manual_subs[0]]
                use_auto = False
            elif auto_subs:
                chosen_lang = [auto_subs[0]]
                use_auto = True
    else:
        if manual_subs:
            chosen_lang = _match_sub_lang_candidates("auto", manual_subs)
            use_auto = False
        elif auto_subs:
            chosen_lang = _match_sub_lang_candidates("auto", auto_subs)
            use_auto = True

    sub_note = (
        f"Phụ đề gốc: {chosen_lang[0]} ({'tự động' if use_auto else 'có sẵn của video'})"
        if chosen_lang
        else "Video không có phụ đề rời đi kèm"
    )
    if on_progress:
        on_progress(12.0, f"Đang tải tốc độ cao ({threads} luồng{' + aria2c' if aria2_path else ''}) | {sub_note}...")

    def _hook(d: Dict[str, Any]):
        if not on_progress:
            return
        st = d.get("status")
        if st == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            down = d.get("downloaded_bytes") or 0
            spd = strip_ansi(d.get("_speed_str", "").strip())
            eta = strip_ansi(d.get("_eta_str", "").strip())
            pct = 12.0 + (down / total) * 72.0 if total > 0 else 40.0
            on_progress(round(min(84.0, pct), 1), f"Đang tải ({threads} luồng): {spd} • Còn {eta}")
        elif st == "finished":
            on_progress(86.0, "Tải xong luồng dữ liệu, đang ghép Video + Audio + Phụ đề...")

    dl_opts: Dict[str, Any] = {
        "outtmpl": str(out_dir / f"{safe_stem}.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": ffmpeg_dir,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "concurrent_fragment_downloads": threads,
        "retries": 10,
        "fragment_retries": 10,
        "progress_hooks": [_hook],
    }

    if aria2_path:
        dl_opts["external_downloader"] = {"default": aria2_path}
        # yt-dlp Aria2cFD defaults to -x16 -j16 -s16. If threads < 16, downscale; otherwise keep standard 16.
        if threads < 16:
            dl_opts["external_downloader_args"] = {
                "aria2c": [
                    f"-x{threads}",
                    f"-s{threads}",
                    f"-j{threads}",
                ]
            }

    if req.browser_cookies != "none":
        dl_opts["cookiesfrombrowser"] = (req.browser_cookies,)

    if chosen_lang:
        dl_opts["subtitleslangs"] = chosen_lang
        dl_opts["writesubtitles"] = not use_auto
        dl_opts["writeautomaticsub"] = use_auto
        dl_opts["subtitlesformat"] = "srt/vtt/best"
        dl_opts["postprocessors"] = [{"key": "FFmpegSubtitlesConvertor", "format": "srt"}]

    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])
    except Exception as ex:
        # If external downloader / aria2c failed for ANY reason, auto-fallback to native yt-dlp multi-fragment downloader
        if aria2_path:
            if on_progress:
                on_progress(45.0, f"Đang tự động chuyển sang tải đa luồng nội bộ ({threads} luồng)...")
            fb_opts = dict(dl_opts)
            fb_opts.pop("external_downloader", None)
            fb_opts.pop("external_downloader_args", None)
            fb_opts["concurrent_fragment_downloads"] = threads
            try:
                with yt_dlp.YoutubeDL(fb_opts) as ydl_fb:
                    ydl_fb.download([url])
            except Exception as fb_ex:
                raise RuntimeError(format_download_error(fb_ex)) from fb_ex
        else:
            raise RuntimeError(format_download_error(ex)) from ex

    # Locate downloaded mp4 & srt files
    video_files = sorted(
        [p for p in out_dir.iterdir() if p.suffix.lower() in (".mp4", ".mkv", ".webm")],
        key=lambda x: x.stat().st_size,
        reverse=True,
    )
    if not video_files:
        raise RuntimeError("Tải hoàn tất nhưng không tìm thấy file video đầu ra.")
    video_path = video_files[0]

    srt_files = sorted(
        [p for p in out_dir.iterdir() if p.suffix.lower() == ".srt"],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    # If only .vtt or .ass downloaded, convert to .srt via ffmpeg
    if not srt_files:
        other_subs = [p for p in out_dir.iterdir() if p.suffix.lower() in (".vtt", ".ass", ".ssa")]
        if other_subs:
            conv_srt = out_dir / f"{safe_stem}.srt"
            subprocess.run(
                [get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(other_subs[0]), str(conv_srt)],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if conv_srt.exists():
                srt_files = [conv_srt]

    srt_path = ""
    if srt_files:
        # Rename cleanly to match video stem
        target_srt = out_dir / f"{safe_stem}.srt"
        if srt_files[0] != target_srt:
            shutil.copy2(srt_files[0], target_srt)
        srt_path = str(target_srt)

    return {
        "video_path": str(video_path),
        "srt_path": srt_path,
        "title": safe_stem,
        "subtitle_source": sub_note,
        "used_aria2c": bool(aria2_path),
        "threads": threads,
    }


async def download_and_clean_video(
    req: DownloadVideoRequest,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    res = await asyncio.to_thread(_download_sync, req, on_progress)
    srt_path = res.get("srt_path", "")
    clean_stats: Dict[str, Any] = {}

    if srt_path and Path(srt_path).exists():
        if req.clean_srt_rules:
            if on_progress:
                on_progress(90.0, "Đang lọc rác SRT cơ bản (thẻ HTML, dòng trùng, [Nhạc])...")
            _, clean_stats = await asyncio.to_thread(clean_srt_rules_sync, srt_path)

        if req.clean_srt_ai:
            if on_progress:
                on_progress(93.0, f"Đang nhờ Antigravity ({req.clean_model}) sửa & xóa dòng rác SRT...")
            cleaned_p, ai_stats = await clean_srt_with_antigravity(
                srt_path,
                model=req.clean_model,
                thinking_level=req.clean_thinking_level,
                api_key=req.api_key,
                on_progress=lambda p, m: on_progress(round(92.0 + p * 0.08, 1), m) if on_progress else None,
            )
            res["srt_path"] = str(cleaned_p)
            clean_stats["ai_clean"] = ai_stats

    res["clean_stats"] = clean_stats
    if on_progress:
        on_progress(100.0, f"Tải hoàn tất: {Path(res['video_path']).name}")
    return res
