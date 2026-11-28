"""Higgs TTS 3 voice for a whole video, and the review table behind it.

Order (the Higgs server needs the whole GPU, see gpu_gate):
  1. reference voice: cleaned (music / reverb / noise removed) + cut to whole sentences + its words  (GPU, worker)
  2. server on -> EVERY line generated, 48 at a time, longest first                                  (GPU)
  3. lines whose real audio does not fit their time are rewritten shorter (LLM) and generated again
  4. server off -> Whisper hears every new line -> suspicious lines get reasons                      (GPU)
  5. optional automatic retry of suspicious lines (new seed), best version kept
  6. voice_manifest.json for the mixer + the review table (suspicious lines: text, heard, reasons, play,
     regenerate one / selected / all, choose version, mark OK, edit text)
OCR / sub removal / render only start after this (they wait on the gate; the server is stopped first).

State: data/projects/<id>/voice_cache/higgs/state.json
  {"ref": {...}, "reviewed": bool, "lines": {"<cue index>": {"text", "key", "tries": [try...], "sel", "ok"}}}
  try = {"t", "wav", "dur", "cap", "heard", "err", "reasons", "flag", "tempo"}
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import DATA_DIR, PROJECTS_DIR
from app.models import SubtitleCue, VideoProject, VoiceConfig
from app.services import higgs_tts
from app.services.audio_env import run_worker
from app.services.cancel import check_cancel, is_cancelled
from app.services.gpu_gate import gpu_gate
from app.services.length_fit import allowed_seconds, fit_project_lines, lang_key
from app.services.media import parse_srt
from app.services.voice_review import analyze, score, whisper_lang

ProgressCallback = Callable[[float, str], None]
PARALLEL = 48  # = max_running_requests of higgs3_12gb.yaml
CPU_WHISPER_MAX = 40  # regenerating up to this many lines: Whisper on CPU, the Higgs server stays on
REFS_DIR = DATA_DIR / "voice_refs"
_state_locks: Dict[str, threading.RLock] = {}


# ============================================================================================ reference voice
def _ref_text_for(cfg: VoiceConfig) -> str:
    if cfg.higgs_ref_text.strip():
        return cfg.higgs_ref_text.strip()
    p = Path(cfg.higgs_ref_audio)
    for cand in (p.with_suffix(".txt"), p.with_suffix(".TXT")):
        if cand.exists():
            return cand.read_text(encoding="utf-8-sig", errors="replace").strip()
    return ""


def _ref_cache_dir(cfg: VoiceConfig, text: str) -> Path:
    p = Path(cfg.higgs_ref_audio)
    st = p.stat()
    mode = "raw" if not cfg.higgs_clean_ref else ("strong" if cfg.higgs_clean_strong else "auto")
    h = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|{text}|{mode}|v1".encode("utf-8")).hexdigest()[:12]
    return REFS_DIR / f"{p.stem}_{h}"


def prepare_reference(cfg: VoiceConfig, lang: str, on_progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """Cleans + cuts the reference voice once (cached by file / words / mode). Uses the GPU: call it with the
    gate held and the Higgs server off. -> {"id", "wav", "text", "full_clean", "report"}"""
    src = Path(cfg.higgs_ref_audio or "")
    if not cfg.higgs_ref_audio or not src.exists():
        raise RuntimeError("Higgs TTS 3: chưa chọn file giọng mẫu (hoặc file không còn tồn tại).")
    text = _ref_text_for(cfg)
    out = _ref_cache_dir(cfg, text)
    stem = src.stem
    wav, txt, rep = out / f"{stem}_chuan.wav", out / f"{stem}_chuan.txt", out / f"{stem}_chuan.json"
    if not (wav.exists() and txt.exists()):
        out.mkdir(parents=True, exist_ok=True)
        higgs_tts.stop_if_running()
        args = [str(src), "--lang", whisper_lang(lang), "--min", "8", "--max", "15", "--out", str(out)]
        if text:
            tf = out / "loi_nguoi_dung.txt"
            tf.write_text(text, encoding="utf-8")
            args += ["--txt", str(tf)]
        if not cfg.higgs_clean_ref:
            args.append("--khong-lam-sach")
        elif cfg.higgs_clean_strong:
            args.append("--manh")
        if on_progress:
            on_progress(1.0, "Chuẩn bị giọng mẫu: lọc nhạc nền / vang / tạp âm + cắt câu trọn vẹn...")
        try:
            run_worker(
                "ref_voice_clean.py", args, need_separator=True, timeout=1800,
                on_progress=(lambda a, b: on_progress(min(4.0, 1.0 + float(a) / 33.0), f"Giọng mẫu: {b}")) if on_progress else None,
            )
        except Exception:
            shutil.rmtree(out, ignore_errors=True)
            raise
    report: Dict[str, Any] = {}
    try:
        report = json.loads(rep.read_text(encoding="utf-8"))
    except Exception:
        pass
    ref_words = txt.read_text(encoding="utf-8").strip()
    rid = hashlib.sha1(wav.read_bytes() + ref_words.encode("utf-8")).hexdigest()[:12]
    full = out / f"{stem}_sach_full.wav"
    return {"id": rid, "wav": str(wav), "text": ref_words, "full_clean": str(full) if full.exists() else "",
            "report": report, "source": str(src)}


# ============================================================================================ state
def _dir(project: VideoProject) -> Path:
    d = PROJECTS_DIR / project.id / "voice_cache" / "higgs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lock(project: VideoProject) -> threading.RLock:
    return _state_locks.setdefault(project.id, threading.RLock())


def load_state(project: VideoProject) -> Dict[str, Any]:
    f = _dir(project) / "state.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            bak = f.with_suffix(".bak")
            if bak.exists():
                try:
                    return json.loads(bak.read_text(encoding="utf-8"))
                except Exception:
                    pass
    return {"ref": {}, "reviewed": False, "lines": {}}


def save_state(project: VideoProject, state: Dict[str, Any]) -> None:
    with _lock(project):
        f = _dir(project) / "state.json"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        if f.exists():
            try:
                shutil.copyfile(f, f.with_suffix(".bak"))
            except OSError:
                pass
        tmp.replace(f)


def _key(ref_id: str, text: str) -> str:
    return hashlib.sha1(f"{ref_id}|{text.strip()}".encode("utf-8")).hexdigest()[:14]


def _sel(line: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tries = line.get("tries") or []
    if not tries:
        return None
    s = line.get("sel")
    for t in tries:
        if t["t"] == s:
            return t
    return tries[-1]


def _pick_best(line: Dict[str, Any]) -> None:
    tries = [t for t in line.get("tries") or [] if Path(t["wav"]).exists()]
    if tries:
        line["sel"] = min(reversed(tries), key=score)["t"]  # reversed: newest wins a tie


def _sync_lines(state: Dict[str, Any], cues: List[SubtitleCue], ref_id: str) -> None:
    lines = state.setdefault("lines", {})
    keep = set()
    for c in cues:
        k = str(c.index)
        keep.add(k)
        key = _key(ref_id, c.text)
        ln = lines.get(k)
        if ln is None or ln.get("key") != key:
            lines[k] = {"text": c.text, "key": key, "tries": [], "sel": None, "ok": False}
    for k in list(lines):
        if k not in keep:
            lines.pop(k)


def _cue_info(project: VideoProject) -> Tuple[List[SubtitleCue], Dict[int, float]]:
    srt = project.translated_srt_path or project.srt_path
    if not srt or not Path(srt).exists():
        raise RuntimeError("Chưa có file SRT (gốc hoặc đã dịch) để tạo voice.")
    cues = parse_srt(srt)
    if not cues:
        raise RuntimeError("File SRT không có dòng phụ đề nào.")
    mode = project.config.audio.edit_mode
    return cues, {c.index: allowed_seconds(cues, i, mode) for i, c in enumerate(cues)}


def _max_tempo(project: VideoProject) -> float:
    a = project.config.audio
    return max(1.05, a.max_atempo) if a.edit_mode == "narration" else max(1.35, a.max_atempo)


# ============================================================================================ generation
def _generate(project: VideoProject, state: Dict[str, Any], idxs: List[int], ref: Dict[str, Any], temperature: float,
              on_progress: Optional[ProgressCallback], p0: float, p1: float, label: str) -> int:
    """New version of each line (48 in parallel, longest first). Blocking: run in a thread."""
    if not idxs:
        return 0
    import soundfile as sf

    if not higgs_tts.alive():
        higgs_tts.start(log=lambda m: on_progress(p0, m) if on_progress else None)
    d = _dir(project)
    lines = state["lines"]
    jobs = sorted(idxs, key=lambda i: -len(lines[str(i)]["text"]))
    hdr = {"audio_path": higgs_tts.wsl_path(ref["wav"]), "text": ref["text"]}
    if not ref["text"]:
        hdr.pop("text")
    t0 = time.time()
    done = [0]
    fails: Dict[int, str] = {}
    last_save = [time.time()]

    def one(i: int) -> None:
        if is_cancelled(project.id):
            return
        ln = lines[str(i)]
        t = max([x["t"] for x in ln["tries"]] or [-1]) + 1
        y, sr, cap = higgs_tts.generate(ln["text"], hdr, seed=i * 10 + t, temperature=temperature)
        y = higgs_tts.trim_silence(y, sr)
        wav = d / f"{i:06d}_{ln['key']}_{t}.wav"
        sf.write(str(wav), y, sr, subtype="PCM_16")
        with _lock(project):
            ln["tries"].append({"t": t, "wav": str(wav), "dur": round(len(y) / sr, 3), "cap": bool(cap),
                                "heard": None, "err": None, "reasons": [], "flag": False, "tempo": None})
            ln["sel"] = t
            ln["ok"] = False

    with ThreadPoolExecutor(PARALLEL) as ex:
        futs = {ex.submit(one, i): i for i in jobs}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001  (one line failing must not stop 10 000 others)
                fails[futs[f]] = str(e)[:160]
            done[0] += 1
            if on_progress and (done[0] % 8 == 0 or done[0] == len(jobs)):
                el = max(0.1, time.time() - t0)
                rate = done[0] / el
                on_progress(round(p0 + (p1 - p0) * done[0] / len(jobs), 1),
                            f"{label}: {done[0]}/{len(jobs)} câu ({rate:.1f} câu/s · còn ~{(len(jobs) - done[0]) / max(0.01, rate):.0f}s)"
                            + (f" · {len(fails)} câu lỗi" if fails else ""))
            if time.time() - last_save[0] > 30:
                save_state(project, state)
                last_save[0] = time.time()
            if is_cancelled(project.id):
                for g in futs:
                    g.cancel()
    save_state(project, state)
    check_cancel()
    if fails and len(fails) == len(jobs):
        raise RuntimeError(f"Higgs không tạo được câu nào ({len(fails)} lỗi): {next(iter(fails.values()))}")
    return len(jobs) - len(fails)


def _whisper(project: VideoProject, state: Dict[str, Any], allow: Dict[int, float], device: str,
             on_progress: Optional[ProgressCallback], p0: float, p1: float) -> int:
    """Whisper hears every version not heard yet, then each is judged. Blocking: run in a thread."""
    lines = state["lines"]
    jobs = []
    # in time order: the worker packs ~10 consecutive lines into one 28 s window
    for k in sorted(lines, key=int):
        ln = lines[k]
        for t in ln["tries"]:
            if t.get("heard") is None and Path(t["wav"]).exists():
                jobs.append({"id": f"{k}:{t['t']}", "wav": t["wav"], "text": ln["text"]})
    if not jobs:
        return 0
    lang = project.config.translate.target_lang if project.translated_srt_path else project.config.voice.language
    if device == "cuda":
        higgs_tts.stop_if_running()
    d = _dir(project)
    jf, rf = d / "whisper_jobs.json", d / "whisper_out.jsonl"
    jf.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    if rf.exists():
        rf.unlink()
    if on_progress:
        on_progress(p0, f"Whisper nghe lại {len(jobs)} câu ({'GPU' if device == 'cuda' else 'CPU'})...")
    run_worker(
        "whisper_check.py", [str(jf), str(rf), "--lang", whisper_lang(lang), "--device", device],
        timeout=6 * 3600,
        on_progress=(lambda a, b: on_progress(round(p0 + (p1 - p0) * int(a) / max(1, int(b)), 1),
                                              f"Whisper nghe lại: {a}/{b} câu")) if on_progress else None,
    )
    mt = _max_tempo(project)
    n = 0
    for raw in rf.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(raw)
            k, t = r["id"].split(":")
        except Exception:
            continue
        ln = lines.get(k)
        if not ln:
            continue
        for tr in ln["tries"]:
            if str(tr["t"]) == t:
                tr["heard"] = r.get("heard", "")
                tr.update(analyze(ln["text"], tr["heard"], tr["dur"], tr.get("cap", False), lang, allow.get(int(k), 0.0), mt))
                n += 1
    for ln in lines.values():
        _pick_best(ln)
    save_state(project, state)
    return n


def write_manifest(project: VideoProject, state: Dict[str, Any], cues: List[SubtitleCue]) -> Path:
    items = []
    for c in cues:
        ln = state["lines"].get(str(c.index))
        t = _sel(ln) if ln else None
        items.append({"index": c.index, "start_sec": c.start_sec, "end_sec": c.end_sec, "text": c.text,
                      "audio_path": t["wav"] if t and Path(t["wav"]).exists() else "", "provider_used": "higgs3"})
    srt = project.translated_srt_path or project.srt_path
    manifest = {"project_id": project.id, "srt_file": srt, "voice_config": project.config.voice.model_dump(),
                "workers_used": PARALLEL, "failed": {}, "items": items, "review": review_counts(state)}
    path = PROJECTS_DIR / project.id / "voice_cache" / "voice_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def review_counts(state: Dict[str, Any]) -> Dict[str, int]:
    flagged = ok = unheard = 0
    for ln in state.get("lines", {}).values():
        t = _sel(ln)
        if not t:
            continue
        if t.get("heard") is None:
            unheard += 1
        elif t.get("flag") and not ln.get("ok"):
            flagged += 1
        elif ln.get("ok"):
            ok += 1
    return {"flagged": flagged, "accepted": ok, "unchecked": unheard, "total": len(state.get("lines", {}))}


async def generate_project_voices_higgs(project: VideoProject, on_progress: Optional[ProgressCallback] = None) -> Path:
    cfg = project.config.voice
    lang = project.config.translate.target_lang if project.translated_srt_path else cfg.language
    if on_progress and gpu_gate.busy_text():
        on_progress(1.0, f"Chờ GPU rảnh ({gpu_gate.busy_text()})...")
    async with gpu_gate.exclusive():
        ref = await asyncio.to_thread(prepare_reference, cfg, lang, on_progress)
        state = load_state(project)
        if state.get("ref", {}).get("id") != ref["id"]:
            state["reviewed"] = False
        state["ref"] = {k: ref[k] for k in ("id", "wav", "text", "source")}
        cues, allow = _cue_info(project)
        _sync_lines(state, cues, ref["id"])
        save_state(project, state)
        temp = float(cfg.higgs_temperature)

        todo = [c.index for c in cues if not state["lines"][str(c.index)]["tries"] and c.text.strip()]
        if todo:
            state["reviewed"] = False
            label = ("Higgs TTS 3 tạo voice (48 câu cùng lúc)" if len(todo) >= len(cues) * 0.9 else
                     f"Tạo voice {len(todo)} câu mới / đã đổi chữ ({len(cues) - len(todo)} câu dùng lại)")
            await asyncio.to_thread(_generate, project, state, todo, ref, temp, on_progress, 5.0, 65.0, label)
        elif on_progress:
            on_progress(65.0, f"Cả {len(cues)} câu đã có voice Higgs (dùng lại)")

        # lines whose real audio is longer than their time: shorter text (LLM), voiced again
        if cfg.auto_fit and project.translated_srt_path:
            measured = {}
            for c in cues:
                ln = state["lines"][str(c.index)]
                t = _sel(ln)
                if t and not ln.get("fit_done"):  # once per text (a new text = a new line entry)
                    measured[c.index] = t["dur"]
            if measured:
                fit = await fit_project_lines(
                    project, measured=measured,
                    on_progress=(lambda p, m: on_progress(65.0 + p * 0.05, f"[Độ dài] {m}")) if on_progress else None,
                )
                for i in measured:
                    state["lines"][str(i)]["fit_done"] = True
                if fit["changed"]:
                    cues, allow = _cue_info(project)
                    _sync_lines(state, cues, ref["id"])
                    redo = [int(i) for i in fit["changed"]]
                    await asyncio.to_thread(_generate, project, state, redo, ref, temp, on_progress, 70.0, 74.0,
                                            f"Tạo lại {len(redo)} câu đã rút gọn")
                save_state(project, state)

        if cfg.higgs_review:
            rounds = int(cfg.higgs_auto_retry)
            for rnd in range(rounds + 1):
                await asyncio.to_thread(_whisper, project, state, allow, "cuda", on_progress,
                                        75.0 + rnd * 6, 80.0 + rnd * 6)
                if rnd == rounds:
                    break
                again = [int(k) for k, ln in state["lines"].items()
                         if (_sel(ln) or {}).get("read_bad") and not ln.get("ok") and len(ln["tries"]) <= rnd + 1]
                if not again:
                    break
                await asyncio.to_thread(_generate, project, state, again, ref, temp, on_progress, 80.0 + rnd * 6,
                                        81.0 + rnd * 6, f"Tự tạo lại {len(again)} câu nghi ngờ (lần {rnd + 1})")
        await asyncio.to_thread(higgs_tts.stop_if_running)
        path = write_manifest(project, state, cues)
    rc = review_counts(state)
    if on_progress:
        extra = f" · ⚠ {rc['flagged']} câu nghi ngờ trong Bảng kiểm tra voice" if rc["flagged"] else " · Whisper không thấy câu nào nghi ngờ"
        on_progress(100.0, f"Higgs TTS 3 xong {len(cues)} câu{extra if cfg.higgs_review else ''}")
    return path


def pending_lines(project: VideoProject) -> Dict[str, Any]:
    """Lines whose CURRENT text (SRT) has no Higgs voice yet: text changed after the voice was made (shortened
    by 'Rút gọn câu tràn khung', edited...) or never voiced. -> {"indices", "has_voice", "total"}"""
    state = load_state(project)
    cues, _ = _cue_info(project)
    rid = (state.get("ref") or {}).get("id")
    lines = state.get("lines") or {}
    has_voice = bool(rid and any(ln.get("tries") for ln in lines.values()))
    out = []
    for c in cues:
        if not c.text.strip():
            continue
        ln = lines.get(str(c.index))
        if not has_voice or ln is None or ln.get("key") != _key(rid, c.text) or not ln.get("tries"):
            out.append(c.index)
    return {"indices": out, "has_voice": has_voice, "total": len(cues)}


def needs_review_pause(project: VideoProject) -> int:
    """Number of unresolved suspicious lines when the A-Z run should stop for the user's check (else 0)."""
    cfg = project.config.voice
    if cfg.provider != "higgs3" or not cfg.higgs_review or not cfg.higgs_pause_review:
        return 0
    st = load_state(project)
    if st.get("reviewed"):
        return 0
    return review_counts(st)["flagged"]


# ============================================================================================ review table
def review_rows(project: VideoProject, show: str = "flagged") -> Dict[str, Any]:
    state = load_state(project)
    try:
        cues, allow = _cue_info(project)
    except RuntimeError:
        cues, allow = [], {}
    src: Dict[int, str] = {}
    if project.srt_path and Path(project.srt_path).exists() and project.translated_srt_path:
        src = {c.index: c.text for c in parse_srt(project.srt_path)}
    rid = (state.get("ref") or {}).get("id")
    rows = []
    n_stale = 0
    for c in cues:
        ln = state["lines"].get(str(c.index))
        if not ln:
            continue
        t = _sel(ln)
        # text changed after this voice was made (shortened / edited): the voice still reads the OLD text
        stale = bool(rid and c.text.strip() and ln.get("key") != _key(rid, c.text))
        n_stale += stale
        flagged = bool(t and t.get("flag") and not ln.get("ok"))
        if show == "flagged" and not flagged and not stale:
            continue
        ai = ln.get("ai") or {}
        if not t or ai.get("t") != t.get("t"):
            ai = {}  # verdict of an older version
        rows.append({
            "index": c.index, "start": c.start_sec, "end": c.end_sec, "text": c.text if stale else ln["text"],
            "voiced_text": ln["text"], "stale": stale, "source": src.get(c.index, ""),
            "heard": (t or {}).get("heard"), "reasons": (t or {}).get("reasons") or [], "err": (t or {}).get("err"),
            "dur": (t or {}).get("dur"), "allow": round(allow.get(c.index, 0.0), 2), "flag": flagged,
            "ok": bool(ln.get("ok")), "ok_by": ln.get("ok_by", ""), "sel": ln.get("sel"), "wav": (t or {}).get("wav", ""),
            "ai": {"verdict": ai.get("verdict"), "note": ai.get("note", "")} if ai else None,
            "tries": [{"t": x["t"], "flag": bool(x.get("flag")), "err": x.get("err"), "wav": x["wav"]} for x in ln["tries"]],
        })
    counts = review_counts(state)
    counts["stale"] = n_stale
    return {"rows": rows, "counts": counts, "reviewed": bool(state.get("reviewed")),
            "ref": state.get("ref", {}), "server": higgs_tts.health_json()}


def set_ok(project: VideoProject, idxs: List[int], ok: bool = True) -> Dict[str, int]:
    with _lock(project):
        st = load_state(project)
        for i in idxs:
            ln = st["lines"].get(str(i))
            if ln:
                ln["ok"] = ok
                ln["ok_by"] = "user" if ok else ""
        save_state(project, st)
        cues, _ = _cue_info(project)
        write_manifest(project, st, cues)
    return review_counts(st)


def select_version(project: VideoProject, idx: int, t: int) -> None:
    with _lock(project):
        st = load_state(project)
        ln = st["lines"].get(str(idx))
        if not ln or not any(x["t"] == t for x in ln["tries"]):
            raise RuntimeError("Không có phiên bản này.")
        ln["sel"] = t
        save_state(project, st)
        cues, _ = _cue_info(project)
        write_manifest(project, st, cues)


def mark_reviewed(project: VideoProject, value: bool = True) -> None:
    with _lock(project):
        st = load_state(project)
        st["reviewed"] = value
        save_state(project, st)


def _apply_edits(project: VideoProject, edits: Dict[int, str]) -> None:
    """New text for some lines: written into the SRT the voice reads (translated SRT, else source SRT)."""
    from app.services.media import write_srt

    srt = project.translated_srt_path or project.srt_path
    if not edits or not srt:
        return
    cues = parse_srt(srt)
    out = [SubtitleCue(index=c.index, start_sec=c.start_sec, end_sec=c.end_sec,
                       text=(edits.get(c.index) or c.text).strip()) for c in cues]
    write_srt(out, srt)
    if project.translated_srt_path:
        from app.services import translator as tr

        cp = tr.load_translate_checkpoint(project)
        if cp.get("translated"):
            for i, t in edits.items():
                cp["translated"][str(i)] = t.strip()
            tr.save_translate_checkpoint(project, cp)


async def regenerate_lines(project: VideoProject, idxs: List[int], edits: Optional[Dict[int, str]] = None,
                           on_progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """"Tạo lại" from the review table: a new version for each line (edited text first), then Whisper checks
    it. Few lines -> Whisper on CPU and the server stays on (the next click is instant); many -> server off,
    Whisper on GPU."""
    cfg = project.config.voice
    lang = project.config.translate.target_lang if project.translated_srt_path else cfg.language
    edits = {int(k): v for k, v in (edits or {}).items() if v and v.strip()}
    if edits:
        _apply_edits(project, edits)
    async with gpu_gate.exclusive():
        ref = await asyncio.to_thread(prepare_reference, cfg, lang, on_progress)
        state = load_state(project)
        state["ref"] = {k: ref[k] for k in ("id", "wav", "text", "source")}
        cues, allow = _cue_info(project)
        _sync_lines(state, cues, ref["id"])
        # lines whose text changed since their voice (shortened / edited) lost their old versions in _sync_lines:
        # voice them now too, otherwise they would be silent in the mix
        idxs = list(idxs) + [c.index for c in cues if c.text.strip() and not state["lines"][str(c.index)]["tries"]]
        idxs = [i for i in dict.fromkeys(int(x) for x in idxs) if str(i) in state["lines"]]
        if not idxs:
            return review_counts(state)
        await asyncio.to_thread(_generate, project, state, idxs, ref, float(cfg.higgs_temperature), on_progress,
                                5.0, 70.0, f"Tạo voice {len(idxs)} câu")
        device = "cpu" if len(idxs) <= CPU_WHISPER_MAX else "cuda"
        await asyncio.to_thread(_whisper, project, state, allow, device, on_progress, 70.0, 98.0)
        # the newest version is what the user asked for, unless Whisper finds it clearly worse
        for i in idxs:
            ln = state["lines"][str(i)]
            if ln["tries"]:
                new = ln["tries"][-1]
                best = min(ln["tries"], key=score)
                ln["sel"] = new["t"] if score(new)[:2] <= score(best)[:2] else best["t"]
        save_state(project, state)
        write_manifest(project, state, cues)
    rc = review_counts(state)
    if on_progress:
        on_progress(100.0, f"Tạo lại xong {len(idxs)} câu · còn {rc['flagged']} câu nghi ngờ")
    return rc


async def preview_line(cfg: VoiceConfig, text: str) -> bytes:
    """'Nghe thử giọng' with Higgs: one line with the prepared reference (first time: server start ~1 min)."""
    import io

    import soundfile as sf

    async with gpu_gate.exclusive():
        ref = await asyncio.to_thread(prepare_reference, cfg, cfg.language, None)
        if not await asyncio.to_thread(higgs_tts.alive):
            await asyncio.to_thread(higgs_tts.start)
        hdr = {"audio_path": higgs_tts.wsl_path(ref["wav"]), "text": ref["text"]}
        y, sr, _ = await asyncio.to_thread(higgs_tts.generate, text.strip(), hdr, 7, 180.0, float(cfg.higgs_temperature))
        y = higgs_tts.trim_silence(y, sr)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def lang_of(project: VideoProject) -> str:
    return lang_key(project.config.translate.target_lang if project.translated_srt_path else project.config.voice.language)
