"""'AI soát bảng kiểm tra voice': the suspicious lines of the review table go to the translation LLM
(Antigravity / Gemini) in batches. For each line it reads the intended text, what Whisper heard and the
tool's reasons, and decides whether the difference is only Whisper writing down what it heard its own way
(same sound, spelling, foreign names, numbers, short forms, joined words...) or a real reading error of the
voice (words missing / repeated / replaced by different words, cut last word, gibberish).

  OK   -> line marked OK (ok_by = "ai") unless a length problem is listed (the AI cannot hear the audio:
          loops / silence / too long for its time are kept for the user)
  SAI  -> stays in the table; "Tạo lại câu AI kết luận sai" regenerates them
  NGHI -> stays in the table for the user to listen

The verdict is stored per version (ln["ai"]["t"]): a regenerated line is judged again.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.config import LANGUAGE_NAME_MAP
from app.models import VideoProject

ProgressCallback = Callable[[float, str], None]

# reasons a text-only check cannot clear (they are about the audio itself)
_AUDIO_REASONS = ("Dài bất thường", "Chạm trần", "Dài quá khung")
_VERDICT_RE = re.compile(r"^\s*\[?(\d+)\]?\s*\|\|\|\s*(OK|SAI|NGHI)[^|]*?(?:\|\|\|\s*(.*))?$", re.I)

_SYSTEM = (
    "Bạn là kỹ sư kiểm định giọng đọc AI (TTS) cho phim thuyết minh. Whisper (nhận dạng giọng nói) đã nghe lại "
    "từng câu voice; bạn phân biệt lỗi do WHISPER CHÉP SAI với lỗi do GIỌNG ĐỌC THẬT SỰ ĐỌC SAI. "
    "Chỉ trả về đúng định dạng được yêu cầu, không giải thích thêm."
)


def _prompt(rows: List[Dict[str, Any]], lang_name: str, style: str) -> str:
    body = "\n".join(
        f"{r['index']}|||{_one(r['text'])}|||{_one(r.get('heard') or '')}|||{_one('; '.join(r.get('reasons') or []))}"
        for r in rows
    )
    sg = f"=== TÊN RIÊNG / XƯNG HÔ TRONG PHIM (tham khảo) ===\n{style[:3000]}\n=====\n\n" if style else ""
    return (
        f"NGÔN NGỮ CÂU THOẠI: {lang_name}\n\n{sg}"
        "Mỗi dòng đầu vào: <số>|||<câu voice PHẢI đọc>|||<Whisper nghe được>|||<lý do tool nghi ngờ>\n\n"
        "Giọng đọc TTS gần như luôn đọc đúng chữ; Whisper thì hay chép sai. Coi là DO WHISPER (→ OK) khi khác biệt chỉ là:\n"
        "- từ đồng âm / gần âm, chính tả khác, viết liền / tách / gạch nối, dấu câu, hoa thường;\n"
        "- tên riêng, tên nước ngoài, phiên âm (Whisper viết kiểu khác nhưng đọc lên giống nhau);\n"
        "- số viết bằng chữ ↔ chữ số, viết tắt, dạng khẩu ngữ / rút gọn của cùng một từ;\n"
        "- Whisper bỏ sót / thêm 1 từ đệm ngắn (hư từ, tiểu từ) mà nghĩa câu không đổi;\n"
        "- Whisper dịch / chép sang ngôn ngữ khác nhưng đúng nghĩa câu.\n"
        "Coi là GIỌNG ĐỌC SAI (→ SAI) khi: thiếu cả cụm từ có nghĩa, mất từ cuối câu, đọc lặp lại cụm từ, thêm câu / "
        "từ lạ không có trong câu, đọc thành từ khác hẳn (không gần âm) làm đổi nghĩa, Whisper ra chuỗi vô nghĩa "
        "hoặc không nghe ra lời.\n"
        "Không chắc (vd thiếu 1 từ có nghĩa nhưng có thể Whisper nghe sót) → NGHI.\n"
        "Lý do về ĐỘ DÀI (dài bất thường, chạm trần, dài quá khung): chỉ xét phần chữ, tool sẽ tự giữ lại cho người nghe.\n\n"
        "ĐẦU RA: mỗi câu đúng 1 dòng '<số>|||OK|SAI|NGHI|||<ghi chú ngắn bằng tiếng Việt, tối đa 15 từ>', đủ tất cả các câu.\n"
        "Ví dụ: 12|||OK|||Whisper viết tên riêng khác, đọc giống nhau\n\n"
        f"[TEXT_START]\n{body}\n[TEXT_END]"
    )


def _one(s: str) -> str:
    return (s or "").replace("\n", " ").replace("|||", "/").strip()


def parse_verdicts(raw: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for line in (raw or "").splitlines():
        m = _VERDICT_RE.match(line.strip().strip("`"))
        if m:
            out[int(m.group(1))] = {"verdict": m.group(2).upper(), "note": (m.group(3) or "").strip()[:200]}
    return out


def _has_audio_reason(reasons: List[str]) -> bool:
    return any(r.startswith(_AUDIO_REASONS) for r in reasons or [])


async def ai_check_review(
    project: VideoProject,
    indices: Optional[List[int]] = None,
    on_progress: Optional[ProgressCallback] = None,
    llm_caller=None,
    batch: int = 60,
    recheck: bool = False,
) -> Dict[str, Any]:
    """-> {"checked", "ok", "marked_ok", "bad", "unsure", "missing"}"""
    from app.services import translator as tr
    from app.services.higgs_voice import _cue_info, _lock, _sel, load_state, review_rows, save_state, write_manifest

    caller = llm_caller or tr.invoke_llm
    data = await asyncio.to_thread(review_rows, project, "flagged")
    want = set(int(i) for i in indices) if indices else None
    rows = [
        r for r in data["rows"]
        if not r["stale"] and r["flag"] and r.get("heard") is not None
        and (want is None or r["index"] in want) and (recheck or not r.get("ai"))
    ]
    res: Dict[str, Any] = {"checked": len(rows), "ok": 0, "marked_ok": 0, "bad": 0, "unsure": 0, "missing": 0}
    if not rows:
        return res
    sent_t = {r["index"]: r["sel"] for r in rows}
    cfg = project.config.translate
    lang = cfg.target_lang if project.translated_srt_path else project.config.voice.language
    lang_name = LANGUAGE_NAME_MAP.get(lang, lang)
    style = ""
    if project.style_guide_path and Path(project.style_guide_path).exists():
        style = Path(project.style_guide_path).read_text(encoding="utf-8", errors="replace").strip()

    verdicts: Dict[int, Dict[str, str]] = {}
    sem = asyncio.Semaphore(max(2, min(6, int(cfg.concurrency or 4))))
    done = [0]
    chunks = [rows[k : k + batch] for k in range(0, len(rows), batch)]

    async def one(chunk: List[Dict[str, Any]]) -> None:
        async with sem:
            tr.check_cancel()
            try:
                raw = await caller(cfg, _SYSTEM, _prompt(chunk, lang_name, style))
            except (tr.JobCancelled, asyncio.CancelledError):
                raise
            except Exception as e:  # noqa: BLE001  (a failed batch stays for the user)
                if len(chunks) == 1:
                    raise RuntimeError(f"AI không trả lời: {e}") from e
                raw = ""
        got = parse_verdicts(raw)
        for r in chunk:
            if r["index"] in got:
                verdicts[r["index"]] = got[r["index"]]
        done[0] += len(chunk)
        if on_progress:
            on_progress(min(95.0, 5.0 + 90.0 * done[0] / len(rows)), f"AI soát bảng kiểm tra voice: {done[0]}/{len(rows)} câu")

    if on_progress:
        on_progress(3.0, f"Gửi {len(rows)} câu nghi ngờ cho AI ({cfg.model}) soát lỗi Whisper...")
    await asyncio.gather(*[one(c) for c in chunks])

    def _apply() -> None:
        with _lock(project):
            st = load_state(project)
            reasons = {r["index"]: r.get("reasons") or [] for r in rows}
            for i, v in verdicts.items():
                ln = st["lines"].get(str(i))
                t = _sel(ln) if ln else None
                if not t or t["t"] != sent_t.get(i):
                    continue  # regenerated meanwhile
                ln["ai"] = {"t": t["t"], "verdict": v["verdict"], "note": v["note"]}
                if v["verdict"] == "OK":
                    res["ok"] += 1
                    if not _has_audio_reason(reasons.get(i, [])):
                        ln["ok"] = True
                        ln["ok_by"] = "ai"
                        res["marked_ok"] += 1
                elif v["verdict"] == "SAI":
                    res["bad"] += 1
                else:
                    res["unsure"] += 1
            save_state(project, st)
            cues, _ = _cue_info(project)
            write_manifest(project, st, cues)

    await asyncio.to_thread(_apply)
    res["missing"] = len(rows) - len(verdicts)
    return res
