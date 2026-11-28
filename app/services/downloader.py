from __future__ import annotations
import asyncio
import os
import re
import shutil
import subprocess
import sys
import threading
import time
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


# Video format: <= 1080p (vertical 1080x1920 included), direct HTTPS (DASH) streams, codecs that decode fast:
#  1. VP9 (YouTube 248: 3.0 GB for an 8 h video, CPU decode+crop 1314 fps)
#  2. H.264 (YouTube 137: 4.7 GB, 1598 fps)
#  AV1 only as a late fallback (399: 2.1 GB but 285 fps CPU decode -> it would throttle AI Clean Sub's ROI
#  reader), and not the HLS "Premium" stream the old "bestvideo[ext=mp4]" picked (616: 7.1 GB, 5 693 fragments).
# Audio: AAC m4a without DRC. Then progressively looser fallbacks (BiliBili / other sites).
_RES = "[height<=1920][width<=1920]"
_AUD = "ba[ext=m4a][format_id!*=drc]/ba[ext=m4a]/ba"
_FORMAT_PREFERENCE = "/".join([
    f"bv{_RES}[vcodec^=vp9][protocol^=http]+({_AUD})",
    f"bv{_RES}[vcodec^=avc1][protocol^=http]+({_AUD})",
    f"bv{_RES}[vcodec!^=av01][protocol^=http]+({_AUD})",
    f"bv{_RES}[vcodec!^=av01]+({_AUD})",
    f"bv{_RES}+({_AUD})",
    f"b{_RES}",
    "bv*+ba/b",
])


def _fmt_dur(sec: Any) -> str:
    try:
        sec = int(round(float(sec)))
    except (TypeError, ValueError):
        return "?"
    h, rem = divmod(max(0, sec), 3600)
    m, s_ = divmod(rem, 60)
    return f"{h}:{m:02d}:{s_:02d}" if h else f"{m:02d}:{s_:02d}"


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


# Lines per request in "automatic" mode: ~1 500 Chinese lines took ~2 min in one Antigravity conversation
# (Gemini 4 Neon); requests run in parallel, so an 8 h movie (10 940 lines) = 8 requests, 2 rounds of 6.
# Size is no longer capped by the 32 K command line: big inputs go to the agent as a file it reads itself
# (translator._call_agentapi), and with a Gemini API key they are a plain HTTPS request.
_AUTO_BATCH_LINES = 1500

_CLEAN_SYSTEM_PROMPT = (
    "Bạn là chuyên gia biên tập và làm sạch file phụ đề SRT gốc (KHÔNG DỊCH SANG NGÔN NGỮ KHÁC).\n"
    "Đầu vào: mỗi dòng có dạng '<số>|||<nội dung>'.\n"
    "Nhiệm vụ:\n"
    "1. GIỮ NGUYÊN ngôn ngữ gốc của phụ đề.\n"
    "2. Dòng rác hoàn toàn (quảng cáo kênh, kêu gọi like/sub/đăng ký, credit team sub, tên website, âm thanh "
    "[Nhạc]/[Vỗ tay], ký tự nhận dạng giọng nói vô nghĩa): trả về '<số>|||DELETE'.\n"
    "3. Dòng thoại có lỗi chính tả ASR/OCR hoặc ký tự thừa: trả về '<số>|||<nội dung đã sửa>' (giữ nguyên ý).\n"
    "4. Dòng thoại đã đúng: KHÔNG trả về gì cho dòng đó (chỉ liệt kê các dòng cần xóa hoặc cần sửa).\n"
    "5. TUYỆT ĐỐI chỉ trả về các dòng đúng định dạng '<số>|||<nội dung đã sửa hoặc DELETE>', không giải thích, "
    "không markdown. Nếu không có dòng nào cần xóa/sửa, trả về đúng chữ: NONE"
)


def _pack_batches(cues: List[SubtitleCue], batch_lines: int, char_budget: int) -> List[List[SubtitleCue]]:
    """Fills each request up to `batch_lines` lines (0 = no line limit) AND the prompt char budget."""
    batches: List[List[SubtitleCue]] = []
    cur: List[SubtitleCue] = []
    size = 0
    for c in cues:
        ln = len(f"{c.index}|||{c.text}") + 1
        if cur and ((batch_lines and len(cur) >= batch_lines) or size + ln > char_budget):
            batches.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += ln
    if cur:
        batches.append(cur)
    return batches


async def clean_srt_with_antigravity(
    srt_path: str,
    model: str = "gemini-3.8-flash",
    thinking_level: str = "low",
    api_key: str = "",
    on_progress: Optional[ProgressCallback] = None,
    llm_caller=invoke_llm,
    concurrency: int = 6,
    cancel: Optional[threading.Event] = None,
    batch_lines: int = 0,
) -> Tuple[Path, Dict[str, Any]]:
    """Uses the selected Antigravity model to remove spam/watermark/ASR garbage lines & fix typos in the original
    SRT WITHOUT translating or altering timestamps.

    Few, big requests: the model answers ONLY the lines to delete or fix (unchanged lines are omitted), so the
    answer stays short however many lines are sent. Lines per request = `batch_lines` (0 = automatic: as many as
    fit in one prompt -> ~1 000 Chinese lines via agentapi, the whole file with a Gemini API key). The old way
    echoed every line back in 80-line batches: 137 sequential requests (~50 min) for an 8 h movie."""
    cues = parse_srt(srt_path)
    if not cues:
        return Path(srt_path), {"before_count": 0, "after_count": 0, "removed_count": 0}

    cfg = TranslateConfig(
        provider="antigravity",
        model=model,
        thinking_level=thinking_level,  # type: ignore[arg-type]
        api_key=api_key,
    )
    for c in cues:
        c.text = c.text.replace("\r", " ").replace("\n", " ")
    per_request = int(batch_lines or 0) if int(batch_lines or 0) > 0 else _AUTO_BATCH_LINES
    batches = _pack_batches(cues, per_request, 10**12)
    split_note = ""
    edits: Dict[int, str] = {}

    sem = asyncio.Semaphore(max(1, concurrency))
    done = [0]
    t0 = time.time()

    async def _one(batch: List[SubtitleCue]) -> None:
        async with sem:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Đã dừng theo yêu cầu.")
            block = "\n".join(f"{c.index}|||{c.text}" for c in batch)
            raw = await llm_caller(cfg, _CLEAN_SYSTEM_PROMPT, f"[TEXT_START]\n{block}\n[TEXT_END]")
            valid = {c.index for c in batch}
            for idx, txt in parse_translated_lines(raw).items():
                if idx in valid:
                    edits[idx] = txt
            done[0] += 1
            if on_progress:
                el = time.time() - t0
                eta = el / done[0] * (len(batches) - done[0])
                on_progress(
                    round(10.0 + done[0] / len(batches) * 85.0, 1),
                    f"Antigravity ({model}) dọn rác SRT: xong {done[0]}/{len(batches)} lần gửi"
                    f" · đã chạy {_fmt_dur(el)}" + (f" · còn ~{_fmt_dur(eta)}" if done[0] < len(batches) else ""),
                )

    if on_progress:
        per = -(-len(cues) // len(batches))
        on_progress(
            10.0,
            f"Antigravity ({model}) dọn rác SRT: {len(cues)} dòng → {len(batches)} lần gửi (~{per} dòng/lần, "
            f"{min(concurrency, len(batches))} lần song song){split_note}...",
        )
    await asyncio.gather(*(_one(b) for b in batches))

    out_cues: List[SubtitleCue] = []
    removed = fixed = 0
    for c in cues:
        val = edits.get(c.index)
        if val is not None and val.strip().upper() == "DELETE":
            removed += 1
            continue
        if val is not None and val.strip() and val.strip() != c.text.strip():
            fixed += 1
            text = val.strip()
        else:
            text = c.text
        out_cues.append(SubtitleCue(index=len(out_cues) + 1, start_sec=c.start_sec, end_sec=c.end_sec, text=text))

    stem = Path(srt_path).stem
    stem = stem[: -len("_clean")] if stem.endswith("_clean") else stem  # no "<name>_clean_clean.srt"
    out_path = Path(srt_path).with_name(f"{stem}_clean.srt")
    write_srt(out_cues, str(out_path))
    if on_progress:
        on_progress(
            100.0,
            f"AI đã dọn {removed} dòng rác, sửa {fixed} dòng ({len(batches)} lần gửi, {_fmt_dur(time.time() - t0)}). "
            f"Còn {len(out_cues)}/{len(cues)} dòng.",
        )
    return out_path, {
        "before_count": len(cues),
        "after_count": len(out_cues),
        "removed_count": removed,
        "fixed_count": fixed,
        "requests": len(batches),
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
    cancel: Optional[threading.Event] = None,
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
        "format": _FORMAT_PREFERENCE,  # so info["requested_formats"] = what will really be downloaded (sizes)
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
    is_youtube = str(info.get("extractor_key") or info.get("extractor") or "").lower().startswith("youtube")
    # aria2c only off YouTube: YouTube throttles every extra connection (measured on a ~100 Mbps line, format
    # 137: yt-dlp's own chunked downloader 12 MB/s = the line's cap; aria2c x16 3.8 MB/s, x1 0.34 MB/s, and the
    # last pieces crawled at ~10 KB/s for minutes). BiliBili's CDN still benefits from parallel connections.
    use_aria2 = bool(aria2_path) and not is_youtube
    t_start = time.time()

    def _elapsed() -> str:
        return _fmt_dur(time.time() - t_start)

    if on_progress:
        how = f"aria2c {aria_conn} kết nối" if use_aria2 else "tải trực tiếp, tối đa băng thông"
        on_progress(8.0, f"Bắt đầu tải ({how}) | {sub_note}")

    # Per-stream byte counters -> one overall percentage for video + audio (+ subtitle).
    streams: Dict[str, Dict[str, Any]] = {}
    planned = {
        str(f.get("format_id")): float(f.get("filesize") or f.get("filesize_approx") or 0)
        for f in (info.get("requested_formats") or [])
    }

    def _check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise yt_dlp.utils.DownloadCancelled("Đã dừng tải theo yêu cầu.")

    def _hook(d: Dict[str, Any]):
        _check_cancel()
        if not on_progress:
            return
        st = d.get("status")
        inf = d.get("info_dict") or {}
        fid = str(inf.get("format_id") or d.get("filename") or "?")
        vc, ac = str(inf.get("vcodec") or ""), str(inf.get("acodec") or "")
        if inf.get("ext") in ("vtt", "srt", "json3", "ass", "srv3") or (vc == "none" and ac == "none"):
            kind = "phụ đề"
        else:
            kind = "hình" if vc not in ("", "none") else "tiếng"
        total = float(d.get("total_bytes") or d.get("total_bytes_estimate") or planned.get(fid) or 0)
        rec = streams.setdefault(fid, {"done": 0.0, "total": total, "kind": kind})
        rec["total"] = max(rec["total"], total)
        if st == "downloading":
            rec["done"] = float(d.get("downloaded_bytes") or 0)
        elif st == "finished":
            rec["done"] = max(rec["done"], rec["total"], float(d.get("downloaded_bytes") or d.get("total_bytes") or 0))
        all_total = sum(max(v["total"], planned.get(k, 0.0)) for k, v in streams.items())
        all_total += sum(sz for k, sz in planned.items() if k not in streams)
        all_done = sum(v["done"] for v in streams.values())
        pct = 8.0 + 77.0 * (all_done / all_total) if all_total > 0 else 30.0
        media = list(planned) or list(streams)
        idx = (media.index(fid) + 1) if fid in media else len(streams)
        n = max(1, len(media))
        if st == "downloading":
            spd = float(d.get("speed") or 0)
            eta = d.get("eta")
            msg = (
                f"Đang tải {kind} ({idx}/{n}): {all_done / 1e6:,.0f} / {all_total / 1e6:,.0f} MB · {spd / 1e6:.1f} MB/s"
                + (f" · còn {_fmt_dur(eta)}" if eta else "")
                + f" · đã chạy {_elapsed()}"
            )
            on_progress(round(min(85.0, pct), 1), msg)
        elif st == "finished":
            nxt = "chuyển sang phần tiếp theo" if idx < n else "chuẩn bị ghép"
            on_progress(round(min(85.0, pct), 1), f"Tải xong {kind} ({idx}/{n}), {nxt}... · đã chạy {_elapsed()}")

    def _pp_hook(d: Dict[str, Any]):
        _check_cancel()
        if not on_progress or d.get("status") != "started":
            return
        pp = str(d.get("postprocessor") or "")
        if pp == "Merger":
            on_progress(86.0, f"Đang ghép hình + tiếng thành 1 file MP4 (không nén lại)... · đã chạy {_elapsed()}")
        elif "Subtitle" in pp:
            on_progress(88.0, f"Đang chuyển phụ đề sang SRT... · đã chạy {_elapsed()}")

    dl_opts: Dict[str, Any] = {
        "outtmpl": str(out_dir / f"{safe_stem}.%(ext)s"),
        "format": _FORMAT_PREFERENCE,
        "merge_output_format": "mp4",
        "ffmpeg_location": ffmpeg_dir,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,  # the console progress bar flooded data/logs/server.log
        "concurrent_fragment_downloads": threads,  # HLS / DASH-fragment sites (BiliBili, m3u8 fallbacks)
        "retries": 10,
        "fragment_retries": 10,
        "progress_hooks": [_hook],
        "postprocessor_hooks": [_pp_hook],
    }

    if use_aria2:
        dl_opts["external_downloader"] = {"default": aria2_path}
        # yt-dlp Aria2cFD defaults to -x16 -j16 -s16. If threads < 16, downscale; otherwise keep standard 16.
        if threads < 16:
            dl_opts["external_downloader_args"] = {"aria2c": [f"-x{threads}", f"-s{threads}", f"-j{threads}"]}

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
    except yt_dlp.utils.DownloadCancelled as ex:
        raise RuntimeError("Đã dừng tải theo yêu cầu.") from ex
    except Exception as ex:
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Đã dừng tải theo yêu cầu.") from ex
        # aria2c failed for any reason -> retry with yt-dlp's own multi-fragment downloader
        if use_aria2:
            if on_progress:
                on_progress(45.0, f"Đang tự động chuyển sang tải đa luồng nội bộ ({threads} luồng)...")
            fb_opts = dict(dl_opts)
            fb_opts.pop("external_downloader", None)
            fb_opts.pop("external_downloader_args", None)
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
        "used_aria2c": use_aria2,
        "threads": threads,
        "download_sec": round(time.time() - t_start, 1),
    }


async def download_and_clean_video(
    req: DownloadVideoRequest,
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    res = await asyncio.to_thread(_download_sync, req, on_progress, cancel)
    srt_path = res.get("srt_path", "")
    clean_stats: Dict[str, Any] = {}

    if srt_path and Path(srt_path).exists():
        if req.clean_srt_rules:
            if on_progress:
                on_progress(90.0, "Đang lọc rác phụ đề gốc (thẻ HTML, dòng trùng, [Nhạc]) — không phải xóa sub trên hình...")
            _, clean_stats = await asyncio.to_thread(clean_srt_rules_sync, srt_path)

        if req.clean_srt_ai:
            if on_progress:
                on_progress(93.0, f"Đang nhờ Antigravity ({req.clean_model}) sửa & xóa dòng rác SRT...")
            cleaned_p, ai_stats = await clean_srt_with_antigravity(
                srt_path,
                model=req.clean_model,
                thinking_level=req.clean_thinking_level,
                api_key=req.api_key,
                on_progress=lambda p, m: on_progress(round(92.0 + p * 0.07, 1), m) if on_progress else None,
                cancel=cancel,
                batch_lines=req.clean_batch_lines,
            )
            res["srt_path"] = str(cleaned_p)
            clean_stats["ai_clean"] = ai_stats

    res["clean_stats"] = clean_stats
    res["total_sec"] = round(time.time() - t0, 1)
    if on_progress:
        on_progress(99.0, f"Tải hoàn tất: {Path(res['video_path']).name} · tổng {_fmt_dur(res['total_sec'])}")
    return res
