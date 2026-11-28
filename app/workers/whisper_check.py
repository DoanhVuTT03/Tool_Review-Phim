"""Whisper nghe lai cac cau voice vua tao (chay bang Python cua moi truong am thanh: faster-whisper).

Dung: python whisper_check.py <jobs.json> <ket_qua.jsonl> [--lang tl] [--device cuda|cpu] [--model large-v3-turbo]
  jobs.json: [{"id": "...", "wav": "duong_dan.wav", "text": "loi dung cua cau"}, ...]
  ket_qua.jsonl: moi dong {"id", "heard"} (ghi dan -> dung giua chung van giu phan da nghe)
Dong tien do: @@P|<da xong>|<tong>   ket thuc: @@XONG|<so cau>   loi: @@LOI|<thong bao>

Toc do (do 25/11/2026, RTX 3060, 300 cau tieng Philippines ~2,5 s/cau):
  tung cau 1 lan (4 luong)                       :  3,6 cau/s  (Whisper luon ma hoa cua so 30 s du cau chi 2,5 s)
  GOP ~10 cau vao 1 cua so 28 s + 8 cua so / luot : 24,7 cau/s  <- cach nay
Loi nghe duoc cua ca cua so chia ve tung cau bang cach DONG THANG voi loi dung (difflib theo tu): tu khop / sai ->
cau chua tu do; tu thua (doc lap, lan man) -> cau dung truoc no. So sanh voi nghe tung cau: cung ket luan 268/300 cau,
phan lech chu yeu la Whisper nghe tung cau le nghe nham (it ngu canh) -> it bao nham hon.
"""
import argparse
import bisect
import difflib
import json
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")
WHISPER_DIR = os.environ.get("TOOL_WHISPER_DIR", r"D:\Download\Model\faster_whisper")
SR = 16000
GAP = 0.6  # s of silence between two lines inside a window
MAX_WIN = 28.0  # s per window (Whisper sees 30 s)
GROUP_WINDOWS = 48  # windows per transcribe call (results written after each call)
CHAR_LANGS = {"zh", "ja", "ko", "th", "yue"}


def _load(job):
    import librosa
    import numpy as np
    import soundfile as sf

    try:
        y, sr = sf.read(job["wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        return np.asarray(y[: int(MAX_WIN * SR)], np.float32), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _norm(s):
    s = unicodedata.normalize("NFC", s or "").lower().replace("-", " ").replace("’", "'")
    return re.sub(r"[^\w' ]+", " ", s)


def _units(s, char):
    s = _norm(s)
    return [c for c in s if not c.isspace()] if char else [w.strip("'") for w in s.split() if w.strip("'")]


def split_by_alignment(ref_lines, heard, char):
    """Splits the text heard for a whole window back into its lines (see module doc)."""
    raw = [c for c in heard if not c.isspace()] if char else heard.split()
    hyp = [(_units(w, char) or [""])[0] for w in raw]
    ref, owner = [], []
    for i, t in enumerate(ref_lines):
        for u in _units(t, char):
            ref.append(u)
            owner.append(i)
    out = [[] for _ in ref_lines]
    if not ref:
        if raw:
            out[0] = raw
        return ["".join(x) if char else " ".join(x) for x in out]
    hyp_owner = [None] * len(hyp)
    for tag, a, b, c, d in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(d - c):
                ra = min(a + min(k, max(0, b - a - 1)), len(owner) - 1)
                hyp_owner[c + k] = owner[ra]
    last = 0
    for j in range(len(hyp)):
        if hyp_owner[j] is None:
            hyp_owner[j] = last
        last = hyp_owner[j]
    for j, w in enumerate(raw):
        out[hyp_owner[j]].append(w)
    return ["".join(x) if char else " ".join(x) for x in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="large-v3-turbo")
    a = ap.parse_args()
    jobs = json.load(open(a.jobs, encoding="utf-8"))
    if not jobs:
        print("@@XONG|0", flush=True)
        return
    import numpy as np
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    dev = a.device
    if dev == "cuda":
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() < 1:
                dev = "cpu"
        except Exception:
            dev = "cpu"
    wh = WhisperModel(a.model, device=dev, compute_type="float16" if dev == "cuda" else "int8",
                      cpu_threads=0 if dev == "cuda" else max(4, (os.cpu_count() or 8) - 2), download_root=WHISPER_DIR)
    pipe = BatchedInferencePipeline(model=wh)
    batch = 8 if dev == "cuda" else 2
    char = (a.lang or "") in CHAR_LANGS
    out = open(a.out, "a", encoding="utf-8")
    pool = ThreadPoolExecutor(6)
    gap = np.zeros(int(GAP * SR), np.float32)
    done = 0
    step = GROUP_WINDOWS * 10  # ~lines per call
    for g in range(0, len(jobs), step):
        grp = jobs[g : g + step]
        loaded = list(pool.map(_load, grp))
        # windows of whole lines, <= 28 s each
        wins, cur, cur_jobs, pos = [], [], [], 0.0
        for job, (y, err) in zip(grp, loaded):
            if y is None or len(y) < int(0.05 * SR):
                out.write(json.dumps({"id": job["id"], "heard": "", "error": err or "audio rỗng"}, ensure_ascii=False) + "\n")
                continue
            dur = len(y) / SR
            if cur and pos + dur > MAX_WIN:
                wins.append((cur, cur_jobs))
                cur, cur_jobs, pos = [], [], 0.0
            cur += [y, gap]
            cur_jobs.append(job)
            pos += dur + GAP
        if cur:
            wins.append((cur, cur_jobs))
        if wins:
            parts, clips, p = [], [], 0.0
            for wp, _ in wins:
                w = np.concatenate(wp)
                parts.append(w)
                clips.append({"start": p, "end": p + len(w) / SR})
                p += len(w) / SR
            segs, _ = pipe.transcribe(np.concatenate(parts), language=a.lang, beam_size=5, batch_size=batch,
                                      clip_timestamps=clips, without_timestamps=True, condition_on_previous_text=False)
            starts = [c["start"] for c in clips]
            texts = [[] for _ in wins]
            for s in segs:
                texts[max(0, bisect.bisect_right(starts, float(s.start) + 0.01) - 1)].append(s.text.strip())
            for (_, wjobs), tx in zip(wins, texts):
                heard = ("" if char else " ").join(tx)
                for job, h in zip(wjobs, split_by_alignment([j.get("text", "") for j in wjobs], heard, char)):
                    out.write(json.dumps({"id": job["id"], "heard": h.strip()}, ensure_ascii=False) + "\n")
        out.flush()
        done += len(grp)
        print(f"@@P|{done}|{len(jobs)}", flush=True)
    out.close()
    print(f"@@XONG|{len(jobs)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"@@LOI|{type(e).__name__}: {str(e)[:300]}", flush=True)
        sys.exit(1)
