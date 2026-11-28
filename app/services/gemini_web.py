"""Gemini Web translator: uses gemini.google.com (the user's own Google account, no API key) instead of an API.

Every AI request of the translate step (style guide, translation parts, gap filling, shortening long lines) goes
through `call()`, strictly ONE AT A TIME for the whole tool:

  * manual mode: the request is shown in the "Gemini Web" panel of the UI. The user copies the prompt (or downloads
    it as a file), sends it in Gemini, then pastes the answer back. The tool checks the answer and only accepts it
    when every expected line is there exactly once and translated; otherwise it is refused with the list of
    missing / extra lines and the user asks Gemini again.
  * auto mode: the tool sends the prompt itself in its own Chrome window (gemini_web_browser.py), checks the answer
    the same way and retries in a new chat. If it still fails (not signed in, Gemini page changed, answers keep
    dropping lines) the request falls back to the manual panel so the job never dies silently.
"""
from __future__ import annotations

import asyncio
import contextvars
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import LANGUAGE_NAME_MAP
from app.services.cancel import JobCancelled, check_cancel, current_project

# Set by translator._run_batch so the panel can say "Phần 3/9".
part_label: contextvars.ContextVar[str] = contextvars.ContextVar("gemini_web_part_label", default="")

_LINE_RE = re.compile(r"^\s*(\d+)\s*\|\|\|\s*(.*?)\s*$")
_CJK_RE = re.compile(r"[㐀-鿿]")
AUTO_TRIES = 2  # auto mode: tries (each in a new chat) before handing the request to the user


@dataclass
class WebRequest:
    id: str
    project_id: str
    project_name: str
    kind: str  # "translate" | "lines" (shorten...) | "free" (style guide)
    label: str
    prompt: str
    expected: List[int]
    sources: Dict[int, str]
    target_lang: str
    mode: str
    attachment: str = ""  # a file to attach in Gemini with the prompt (the whole .srt for the style guide)
    state: str = "waiting"  # waiting (for the user) | auto (tool is sending) | done
    status: str = ""
    report: Optional[Dict[str, Any]] = None
    created: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None

    def public(self) -> Dict[str, Any]:
        first = min(self.expected) if self.expected else None
        last = max(self.expected) if self.expected else None
        return {
            "id": self.id,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "kind": self.kind,
            "label": self.label,
            "mode": self.mode,
            "state": self.state,
            "status": self.status,
            "report": self.report,
            "n_lines": len(self.expected),
            "first": first,
            "last": last,
            "chars": len(self.prompt),
            "attachment_name": Path(self.attachment).name if self.attachment else "",
            "created": self.created,
        }


_lock: Optional[asyncio.Lock] = None
_current: Optional[WebRequest] = None
_history: List[Dict[str, Any]] = []


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


# ---------------------------------------------------------------------------------------------------------------
# Prompt building / answer checking
# ---------------------------------------------------------------------------------------------------------------
def _block_lines(user_prompt: str) -> List[str]:
    m = re.search(r"\[TEXT_START\]\n(.*?)\n\[TEXT_END\]", user_prompt, re.S)
    return m.group(1).splitlines() if m else []


def classify(user_prompt: str) -> Tuple[str, List[int], Dict[int, str]]:
    """(kind, expected line numbers, source text per line) of a prompt made by the translator."""
    if user_prompt.startswith("PHÂN TÍCH FILE ĐÍNH KÈM") or "[TEXT_START]" not in user_prompt:
        return "free", [], {}
    expected: List[int] = []
    sources: Dict[int, str] = {}
    for ln in _block_lines(user_prompt):
        parts = ln.split("|||")
        if len(parts) >= 2 and parts[0].strip().isdigit():
            i = int(parts[0].strip())
            expected.append(i)
            sources[i] = parts[-1].strip() if len(parts) <= 3 else parts[2].strip()
    kind = "translate" if user_prompt.startswith("NGÔN NGỮ NGUỒN") else "lines"
    return kind, expected, sources


def build_prompt(system_prompt: str, user_prompt: str, kind: str, expected: List[int]) -> str:
    if kind == "free":
        tail = (
            "\n\n==============================\n"
            "ĐỊNH DẠNG TRẢ LỜI: viết kết quả bằng văn bản (Markdown được). Không hỏi lại, không cần giải thích quy trình."
        )
    else:
        first, last = min(expected), max(expected)
        tail = (
            "\n\n==============================\n"
            "⚠ ĐỊNH DẠNG TRẢ LỜI (BẮT BUỘC - tool sẽ tự kiểm tra, sai là bị loại):\n"
            f"- Trả về ĐÚNG {len(expected)} dòng, từ dòng số {first} đến dòng số {last}, mỗi dòng: <số>|||<kết quả>.\n"
            "- Không bỏ dòng, không gộp dòng, không tách dòng, không thêm dòng, giữ nguyên số thứ tự.\n"
            "- Đặt TOÀN BỘ kết quả trong MỘT khối code ```text ... ```. Không viết gì thêm ngoài khối code."
        )
    return (
        "=== HƯỚNG DẪN (VAI TRÒ & QUY TẮC) ===\n"
        f"{system_prompt.strip()}\n\n"
        "=== YÊU CẦU ===\n"
        f"{user_prompt.strip()}{tail}"
    )


def _looks_untranslated(src: str, out: str, target_lang: str) -> bool:
    if target_lang.startswith(("zh", "ja")):
        return False
    o = out.strip()
    if not o:
        return True
    cjk = len(_CJK_RE.findall(o))
    return o == src.strip() or cjk > max(2, len(o) * 0.3)


def check_answer(req: WebRequest, text: str) -> Dict[str, Any]:
    """ok only when the answer has every expected line exactly once, nothing else, and (translation) translated."""
    text = text or ""
    if req.kind == "free":
        ok = len(text.strip()) >= 80
        return {
            "ok": ok,
            "message": "Đã nhận." if ok else "Câu trả lời quá ngắn / rỗng — hãy dán toàn bộ câu trả lời của Gemini.",
        }
    exp = set(req.expected)
    seen: Dict[int, int] = {}
    got: Dict[int, str] = {}
    for ln in text.splitlines():
        m = _LINE_RE.match(ln)
        if not m:
            continue
        i = int(m.group(1))
        t = re.sub(r"^\d{1,3}\s*\|\|\|\s*", "", m.group(2)).strip()  # a copied-back length budget
        seen[i] = seen.get(i, 0) + 1
        if t:
            got[i] = t
    missing = sorted(exp - set(got))
    extra = sorted(set(seen) - exp)
    dup = sorted(i for i, n in seen.items() if n > 1 and i in exp)
    untr = (
        sorted(i for i in exp & set(got) if _looks_untranslated(req.sources.get(i, ""), got[i], req.target_lang))
        if req.kind == "translate"
        else []
    )
    good = sorted(i for i in exp & set(got) if i not in dup and i not in untr)
    ok = not missing and not extra and not dup and not untr
    parts = []
    if missing:
        parts.append(f"thiếu {len(missing)} dòng ({_fmt_ids(missing)})")
    if extra:
        parts.append(f"thừa {len(extra)} dòng không có trong đề ({_fmt_ids(extra)})")
    if dup:
        parts.append(f"{len(dup)} dòng bị lặp số ({_fmt_ids(dup)})")
    if untr:
        parts.append(f"{len(untr)} dòng chưa dịch / còn chữ gốc ({_fmt_ids(untr)})")
    msg = (
        f"Đủ {len(exp)}/{len(exp)} dòng — đã nhận."
        if ok
        else f"KHÔNG nhận: có {len(good)}/{len(exp)} dòng đúng; " + "; ".join(parts)
        + ". Hãy bảo Gemini dịch lại (hoặc bấm Regenerate) rồi dán lại."
    )
    return {
        "ok": ok,
        "expected": len(exp),
        "good": len(good),
        "missing": missing,
        "extra": extra,
        "dup": dup,
        "untranslated": untr,
        "message": msg,
        "_good_text": "\n".join(f"{i}|||{got[i]}" for i in good),
    }


def _fmt_ids(ids: List[int], n: int = 12) -> str:
    s = ", ".join(str(i) for i in ids[:n])
    return s + (f", … +{len(ids) - n}" if len(ids) > n else "")


def _strip_private(rep: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in rep.items() if not k.startswith("_")}
    for k in ("missing", "extra", "dup", "untranslated"):
        if k in out:
            out[k] = out[k][:200]
    return out


def _label(kind: str, expected: List[int]) -> str:
    if kind == "free":
        return "Phân tích Style Guide (nhân vật, xưng hô, văn phong)"
    rng = f"dòng {min(expected)}–{max(expected)} ({len(expected)} dòng)"
    if kind == "lines":
        return f"Rút gọn câu tràn khung · {len(expected)} câu"
    p = part_label.get()
    return f"{p} · {rng}" if p else f"Dịch {rng}"


# ---------------------------------------------------------------------------------------------------------------
# The LLM caller (plugged into translator.invoke_llm for provider "gemini_web")
# ---------------------------------------------------------------------------------------------------------------
async def call(cfg, system_prompt: str, user_prompt: str, attachment: str = "") -> str:
    global _current
    kind, expected, sources = classify(user_prompt)
    pid = current_project.get() or ""
    pname = pid
    try:
        from app.services.store import store

        p = store.get_project(pid) if pid else None
        pname = p.name if p else pid
    except Exception:  # noqa: BLE001
        pass

    async with _get_lock():  # one request at a time for the whole tool
        check_cancel()
        req = WebRequest(
            id=uuid.uuid4().hex[:12],
            project_id=pid,
            project_name=pname,
            kind=kind,
            label=_label(kind, expected),
            prompt=build_prompt(system_prompt, user_prompt, kind, expected),
            expected=expected,
            sources=sources,
            target_lang=cfg.target_lang,
            mode=cfg.web_mode,
            attachment=attachment or "",
        )
        req.future = asyncio.get_running_loop().create_future()
        _current = req
        try:
            if cfg.web_mode == "auto":
                ans = await _try_auto(req, cfg)
                if ans is not None:
                    return ans
            req.state = "waiting"
            if not req.status:
                req.status = (
                    f"Chờ bạn: tải file {Path(req.attachment).name} → đính kèm vào Gemini + dán câu lệnh → gửi → "
                    "dán câu trả lời vào đây."
                    if req.attachment
                    else "Chờ bạn gửi prompt này trên Gemini và dán câu trả lời vào đây."
                )
            return await _wait_user(req)
        finally:
            _history.insert(0, {"label": req.label, "project_name": req.project_name, "ok": req.state == "done",
                                "at": time.time()})
            del _history[20:]
            _current = None


async def _try_auto(req: WebRequest, cfg) -> Optional[str]:
    from app.services import gemini_web_browser as gb

    req.state = "auto"
    last_err = ""
    for n in range(1, AUTO_TRIES + 1):
        check_cancel()

        def _say(m: str, n=n) -> None:
            req.status = f"[Tự động, lần {n}/{AUTO_TRIES}] {m}"

        try:
            text = await gb.run_prompt(
                req.prompt, new_chat=cfg.web_new_chat or n > 1, on_status=_say, attachment=req.attachment
            )
        except (JobCancelled, asyncio.CancelledError):
            raise
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            req.status = f"[Tự động] lỗi: {last_err}"
            # browser / sign-in problems will not fix themselves by retrying
            if "đăng nhập" in last_err or "ô nhập" in last_err or "Không tìm thấy" in last_err:
                break
            continue
        rep = check_answer(req, text)
        req.report = _strip_private(rep)
        if rep["ok"]:
            req.state = "done"
            return text if req.kind == "free" else rep["_good_text"]
        last_err = rep["message"]
        req.status = f"[Tự động, lần {n}/{AUTO_TRIES}] Gemini trả lời sai → gửi lại trong chat mới. {rep['message']}"
    req.status = (
        f"Tự động không thành công ({last_err[:300]}). → Bạn làm tay phần này: copy prompt, gửi trên Gemini, "
        "dán câu trả lời vào đây. Xong phần này tool lại tự động tiếp."
    )
    return None


async def _wait_user(req: WebRequest) -> str:
    assert req.future is not None
    while True:
        check_cancel()
        try:
            return await asyncio.wait_for(asyncio.shield(req.future), 1.0)
        except asyncio.TimeoutError:
            continue


# ---------------------------------------------------------------------------------------------------------------
# API used by main.py
# ---------------------------------------------------------------------------------------------------------------
def pending() -> Optional[Dict[str, Any]]:
    return _current.public() if _current else None


def history() -> List[Dict[str, Any]]:
    return list(_history)


def prompt_text(req_id: str) -> Optional[str]:
    return _current.prompt if _current and _current.id == req_id else None


def attachment_path(req_id: str) -> Optional[str]:
    if _current and _current.id == req_id and _current.attachment and Path(_current.attachment).exists():
        return _current.attachment
    return None


def submit(req_id: str, text: str, accept_partial: bool = False) -> Dict[str, Any]:
    req = _current
    if not req or req.id != req_id or req.future is None or req.future.done():
        return {"ok": False, "message": "Yêu cầu này không còn chờ (đã xong hoặc đã dừng)."}
    if req.state == "auto":
        return {"ok": False, "message": "Tool đang tự gửi phần này, đợi kết quả hoặc dừng tiến trình."}
    rep = check_answer(req, text)
    if rep["ok"]:
        req.state = "done"
        req.report = _strip_private(rep)
        req.future.set_result(text if req.kind == "free" else rep["_good_text"])
        return req.report
    if accept_partial and req.kind != "free" and rep.get("good"):
        # keep the correct lines; the translator asks again (as a new, smaller request) for the rest
        req.state = "done"
        req.report = _strip_private(rep)
        req.future.set_result(rep["_good_text"])
        out = dict(req.report)
        out["ok"] = True
        out["message"] = (
            f"Đã nhận {rep['good']}/{rep['expected']} dòng đúng. Các dòng còn lại sẽ được tạo thành yêu cầu mới "
            "để bạn gửi lại."
        )
        return out
    req.report = _strip_private(rep)
    req.status = rep["message"]
    return req.report


def chunk_size_for(total_lines: int, max_lines: int) -> int:
    """A file of up to max_lines goes in one part; longer files are split into EQUAL parts of <= max_lines."""
    max_lines = max(20, int(max_lines or 250))
    if total_lines <= max_lines:
        return max(1, total_lines)
    n_parts = -(-total_lines // max_lines)
    return -(-total_lines // n_parts)


def lang_name(code: str) -> str:
    return LANGUAGE_NAME_MAP.get(code, code)
