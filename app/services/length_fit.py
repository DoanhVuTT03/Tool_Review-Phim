"""Fit translated lines to their time on screen (all languages, all voices).

Why (video 512, Chinese -> Filipino, measured 25/11/2026): a Chinese line of ~6 characters is spoken in ~1.3 s,
its Filipino translation had ~35 characters = 8.6 syllables/s needed, while Edge fil-PH-Angelo reads ~4.8
syllables/s. All the voice together was 31 557 s for 15 556 s of subtitle time (2x): the mixer sped every line
up to its limit and still pushed each line into the next one's time -> a non-stop, rushed voice that lagged the
picture by minutes. The old prompt only said "±15 % of the source rhythm", with no timing at all.

Now:
  1. translation prompt: every line carries its own budget (max syllables, or characters for CJK) computed from
     the time the line may use and the reading speed of the target language (translator.py);
  2. after translating: lines still over budget are rewritten shorter by the LLM (estimate);
  3. after the voice is made: lines whose REAL audio does not fit are rewritten shorter and voiced again.
"""
from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import LANGUAGE_NAME_MAP
from app.models import SubtitleCue, VideoProject

ProgressCallback = Callable[[float, str], None]

# Natural TTS reading speed, units per second at speed 1.0 (syllables; characters for zh/ja/ko/th).
# fil measured on Edge fil-PH-AngeloNeural (10 934 lines of video 512): 4.8 syllables/s.
RATE: Dict[str, float] = {
    "fil": 4.8, "id": 5.0, "ms": 5.0, "es": 5.8, "pt": 5.3, "it": 5.6, "fr": 5.2, "en": 4.2, "de": 4.3,
    "nl": 4.5, "ru": 4.8, "uk": 4.8, "pl": 5.0, "tr": 5.3, "vi": 4.6, "zh": 4.5, "ja": 7.0, "ko": 5.8,
    "th": 11.0, "hi": 5.0, "ar": 4.6,
}
_CHAR_LANGS = {"zh", "ja", "ko", "th"}
_LATIN_VOWELS = re.compile(r"[aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿăâêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+", re.I)
_CYR_VOWELS = re.compile(r"[аеёиоуыэюяіїє]+", re.I)


def lang_key(lang: str) -> str:
    k = (lang or "").lower().split("-")[0]
    return "fil" if k in ("tl", "fil") else k


def unit_name(lang: str) -> str:
    return "ký tự" if lang_key(lang) in _CHAR_LANGS else "âm tiết"


def speech_units(text: str, lang: str) -> float:
    """Spoken length of a line: syllables (vowel groups) for alphabetic languages, one per word for Vietnamese,
    characters for zh/ja/ko/th. Digits count ~1.5 units each (a number is read as words)."""
    t = re.sub(r"<\|[^|]*\|>", " ", text or "")
    k = lang_key(lang)
    digits = len(re.findall(r"\d", t)) * 1.5
    if k in ("zh", "ja"):
        return len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]", t)) + digits
    if k == "ko":
        return len(re.findall(r"[\uac00-\ud7a3]", t)) + digits
    if k == "th":
        return len(re.findall(r"[\u0e00-\u0e7f]", t)) + digits
    words = re.findall(r"[^\W\d_]+", t)
    if k == "vi":
        return len(words) + digits
    n = 0
    for w in words:
        v = _CYR_VOWELS.findall(w) if re.search(r"[\u0400-\u04ff]", w) else _LATIN_VOWELS.findall(w)
        n += max(1, len(v))
    return n + digits


def reading_rate(lang: str, speed: float = 1.0) -> float:
    return RATE.get(lang_key(lang), 5.0) * max(0.5, min(2.0, speed or 1.0))


def allowed_seconds(cues: List[SubtitleCue], i: int, mode: str) -> float:
    """Time a line may use - the same rule as the audio mixer (audio_assembler._prep_clip): narration may run
    into the pause before the next line (up to +1.8 s), dubbing keeps to its own slot."""
    c = cues[i]
    slot = max(0.25, c.end_sec - c.start_sec)
    if mode != "narration":
        return slot
    nxt = cues[i + 1].start_sec - 0.08 if i + 1 < len(cues) else c.end_sec + 1.8
    return max(slot, min(nxt - c.start_sec, slot + 1.8))


def budget_seconds(cues: List[SubtitleCue], i: int, mode: str) -> float:
    """Time the TEXT is written for: like allowed_seconds but leaving a breath (0.3 s) before the next line and
    borrowing at most 1.2 s of the pause -> lines do not run into each other (non-stop voice sounds rushed
    even at normal speed)."""
    c = cues[i]
    slot = max(0.25, c.end_sec - c.start_sec)
    if mode != "narration":
        return slot
    nxt = cues[i + 1].start_sec - 0.3 if i + 1 < len(cues) else c.end_sec + 1.2
    return max(slot, min(nxt - c.start_sec, slot + 1.2))


def line_budgets(cues: List[SubtitleCue], lang: str, mode: str, speed: float = 1.0, speedup: float = 1.0) -> Dict[int, int]:
    """{cue index: max units} - what is read in the line's time at the voice's natural speed."""
    rate = reading_rate(lang, speed)
    return {c.index: max(2, int(math.floor(budget_seconds(cues, i, mode) * rate * speedup))) for i, c in enumerate(cues)}


def project_budgets(project: VideoProject, cues: List[SubtitleCue]) -> Dict[int, int]:
    v = project.config.voice
    speed = v.speed if v.provider == "edge_tts" else 1.0
    return line_budgets(cues, project.config.translate.target_lang, project.config.audio.edit_mode, speed)


# ------------------------------------------------------------------------------------------- LLM shortening
_SHORTEN_SYSTEM = (
    "Bạn là biên tập viên lồng tiếng / thuyết minh phim chuyên nghiệp. Nhiệm vụ: rút gọn câu thoại đã dịch để "
    "giọng đọc vừa khít thời lượng câu trên video, mà vẫn tự nhiên, đúng ngữ pháp, giữ ý chính và tên riêng. "
    "Chỉ trả về đúng định dạng được yêu cầu, không giải thích."
)


def _shorten_prompt(rows: List[Tuple[int, int, str, str]], lang: str, src_name: str, style_guide: str, strict: bool) -> str:
    tgt = LANGUAGE_NAME_MAP.get(lang, lang)
    unit = unit_name(lang)
    body = "\n".join(f"{i}|||{n}|||{src.replace(chr(10), ' ')}|||{cur.replace(chr(10), ' ')}" for i, n, src, cur in rows)
    sg = f"=== STYLE GUIDE (tên riêng, xưng hô - giữ nguyên) ===\n{style_guide[:6000]}\n=====\n\n" if style_guide else ""
    hard = (
        f"- LẦN TRƯỚC câu vẫn còn dài: lần này PHẢI ngắn hơn hẳn, đếm kỹ số {unit}, chấp nhận tóm ý mạnh.\n" if strict else ""
    )
    return (
        f"NGÔN NGỮ CÂU THOẠI: {tgt}\n\n{sg}"
        f"NHIỆM VỤ: RÚT GỌN các câu thuyết minh / lồng tiếng dưới đây. Chúng đang quá dài so với thời lượng câu trên "
        f"video nên giọng đọc bị tua nhanh, đọc dồn sang câu sau.\n"
        f"Mỗi dòng đầu vào: <số>|||<tối đa N {unit}>|||<câu gốc ({src_name})>|||<bản dịch hiện tại>\n\n"
        f"YÊU CẦU:\n"
        f"- Viết lại bản dịch NGẮN HƠN, KHÔNG vượt quá N {unit} (N là số ở cột thứ 2).\n"
        f"- Giữ ý chính; bỏ từ đệm, từ lặp, chủ ngữ hiểu ngầm được; dùng từ ngắn hơn; được phép tóm ý.\n"
        f"- KHÔNG đổi tên riêng và danh xưng nhân vật (vd 'Ikalawang Ginang' không được thành 'Senyora'): giữ y như bản dịch "
        f"hiện tại / Style Guide để cả phim nhất quán; nếu danh xưng quá dài thì bỏ hẳn khi câu vẫn rõ ai.\n"
        f"- Câu phải tự nhiên, đúng ngữ pháp {tgt}, đọc lên trôi chảy, không cụt lủn vô nghĩa.\n"
        f"- KHÔNG dùng '...' hay '…'. Không thêm chú thích.\n{hard}\n"
        f"ĐẦU RA: mỗi dòng đúng '<số>|||<câu đã rút gọn>' (KHÔNG chép lại N hay câu gốc), đủ tất cả các dòng.\n\n"
        f"[TEXT_START]\n{body}\n[TEXT_END]"
    )


async def shorten_lines(
    project: VideoProject,
    targets: Dict[int, int],
    current: Dict[int, str],
    source: Dict[int, str],
    on_progress: Optional[ProgressCallback] = None,
    llm_caller=None,
    batch: int = 150,
) -> Dict[int, str]:
    """Rewrites the given lines to at most targets[idx] units. Returns {idx: new text} only for lines that came
    back shorter than before (a line the model made longer is ignored)."""
    from app.services import translator as tr

    caller = llm_caller or tr.invoke_llm
    cfg = project.config.translate
    lang = cfg.target_lang
    src_name = LANGUAGE_NAME_MAP.get(cfg.source_lang, cfg.source_lang)
    style = ""
    if project.style_guide_path and Path(project.style_guide_path).exists():
        style = Path(project.style_guide_path).read_text(encoding="utf-8", errors="replace").strip()
    out: Dict[int, str] = {}
    todo = sorted(targets)
    # measured 25/11/2026 (Antigravity Neon medium): ~170 lines per request take ~4 min -> small batches, more at once
    sem = asyncio.Semaphore(max(6, min(10, int(cfg.concurrency or 4))))
    done = [0]
    total = len(todo)

    async def one(chunk: List[int], strict: bool) -> None:
        rows = [(i, targets[i], source.get(i, ""), out.get(i, current[i])) for i in chunk]
        async with sem:
            tr.check_cancel()
            try:
                raw = await caller(cfg, _SHORTEN_SYSTEM, _shorten_prompt(rows, lang, src_name, style, strict))
            except (tr.JobCancelled, asyncio.CancelledError):
                raise
            except Exception:  # noqa: BLE001  (a failed batch keeps its current lines)
                raw = ""
        got = tr.parse_translated_lines(raw)
        for i in chunk:
            new = (got.get(i) or "").strip()
            if new and speech_units(new, lang) < speech_units(out.get(i, current[i]), lang):
                out[i] = new
        if not strict:
            done[0] += len(chunk)
            if on_progress:
                on_progress(min(99.0, 100.0 * done[0] / max(1, total)), f"Rút gọn câu tràn khung: {done[0]}/{total} câu")

    await asyncio.gather(*[one(todo[k : k + batch], False) for k in range(0, len(todo), batch)])
    # second pass, stricter, for lines that are still clearly over budget
    still = [i for i in todo if speech_units(out.get(i, current[i]), lang) > targets[i] * 1.15]
    if still:
        if on_progress:
            on_progress(99.0, f"Rút gọn lần 2 (chặt hơn) cho {len(still)} câu vẫn còn dài...")
        await asyncio.gather(*[one(still[k : k + batch], True) for k in range(0, len(still), batch)])
    return out


def _apply(project: VideoProject, changes: Dict[int, str]) -> None:
    """Writes the shortened lines into the translated SRT (same timings) and the translation checkpoint."""
    from app.services import translator as tr
    from app.services.media import parse_srt, write_srt

    if not changes or not project.translated_srt_path:
        return
    cues = parse_srt(project.translated_srt_path)
    out = [SubtitleCue(index=c.index, start_sec=c.start_sec, end_sec=c.end_sec, text=changes.get(c.index, c.text)) for c in cues]
    write_srt(out, project.translated_srt_path)
    cp = tr.load_translate_checkpoint(project)
    if cp.get("translated"):
        for i, t in changes.items():
            cp["translated"][str(i)] = t
        tr.save_translate_checkpoint(project, cp)


async def fit_project_lines(
    project: VideoProject,
    measured: Optional[Dict[int, float]] = None,
    on_progress: Optional[ProgressCallback] = None,
    llm_caller=None,
    tolerance: float = 1.12,
    skip: Optional[set] = None,
) -> Dict[str, Any]:
    """Finds the translated lines that do not fit their time and rewrites them shorter.
    measured = {cue index: real voice length in s (silence trimmed)} after the voice was made; without it the
    length is estimated from the syllable count. Returns {"checked", "over", "changed": {idx: text}}."""
    from app.services.media import parse_srt

    srt = project.translated_srt_path
    if not srt or not Path(srt).exists():
        return {"checked": 0, "over": 0, "changed": {}}
    cues = parse_srt(srt)
    src_cues = parse_srt(project.srt_path) if project.srt_path and Path(project.srt_path).exists() else []
    source = {c.index: c.text for c in src_cues}
    lang = project.config.translate.target_lang
    mode = project.config.audio.edit_mode
    budgets = project_budgets(project, cues)
    max_tempo = max(1.0, min(1.15, float(project.config.audio.max_atempo)))
    targets: Dict[int, int] = {}
    current: Dict[int, str] = {}
    for i, c in enumerate(cues):
        units = speech_units(c.text, lang)
        if units <= 2 or (skip and c.index in skip):
            continue
        if measured is not None:
            d = measured.get(c.index)
            if not d:
                continue
            allow = allowed_seconds(cues, i, mode)
            if d <= allow * max_tempo * 1.03:
                continue  # the mixer fits it with at most a slight speed-up
            # keep the voice's own reading speed for this line, scale its length to its time (with a breath)
            targets[c.index] = max(2, int(units * budget_seconds(cues, i, mode) / d))
        else:
            if units <= budgets[c.index] * tolerance:
                continue
            targets[c.index] = budgets[c.index]
        current[c.index] = c.text
    res: Dict[str, Any] = {"checked": len(cues), "over": len(targets), "changed": {}, "targets": sorted(targets)}
    if not targets:
        return res
    if on_progress:
        how = "voice thật dài hơn khung" if measured is not None else "ước tính đọc không kịp"
        on_progress(1.0, f"{len(targets)}/{len(cues)} câu {how} → nhờ AI rút gọn...")
    changed = await shorten_lines(project, targets, current, source, on_progress, llm_caller)
    _apply(project, changed)
    res["changed"] = changed
    return res


def measure_voice_lengths(manifest_items: List[Dict[str, Any]], workers: int = 12) -> Dict[int, float]:
    """Real spoken length of each voice clip (silence at both ends removed, like the mixer does)."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    def one(it: Dict[str, Any]) -> Tuple[int, float]:
        ap = it.get("audio_path") or ""
        if not ap or not Path(ap).exists():
            return int(it["index"]), 0.0
        try:
            import soundfile as sf

            y, sr = sf.read(ap, dtype="float32")
        except Exception:
            try:
                from app.services.audio_assembler import SAMPLE_RATE, _decode_to_mono_f32

                y, sr = _decode_to_mono_f32(ap, sr=SAMPLE_RATE, timeout=60), SAMPLE_RATE
            except Exception:
                return int(it["index"]), 0.0
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(1)
        if not len(y):
            return int(it["index"]), 0.0
        fr = max(1, int(sr * 0.01))
        n = len(y) // fr
        if n < 3:
            return int(it["index"]), len(y) / sr
        e = np.sqrt((y[: n * fr].reshape(n, fr) ** 2).mean(1))
        on = np.where(e > max(1e-4, float(e.max()) * 0.01))[0]
        if not len(on):
            return int(it["index"]), 0.0
        return int(it["index"]), (on[-1] - on[0] + 1) * fr / sr

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, manifest_items) if v > 0}
