"""CHUAN HOA GIONG MAU de clone giong (Higgs TTS 3) - chay bang Python cua moi truong am thanh (D:\\Download\\Model\\envs\\giong_sach:
audio-separator + faster-whisper + DNSMOS). Ban sao tu gemini_srt_translator/giong_mau_chuan.py, them che do --manh.

1. Do chat luong (DNSMOS P.835: SIG = giong, BAK = nen/on, OVRL = tong).
2. Lam sach CO DIEU KIEN, moi buoc chi ap dung khi phan bi loc ra du lon (giong sach thi de nguyen, khong lam hong):
     nhac nen / tieng nen -> tach giong (Mel-Roformer kim_ft)  |  vang, om -> khu vang (Mel-Roformer dereverb)
     tap am -> khu on (Mel-Roformer denoise). Cuoi cung do lai DNSMOS: khong tot hon thi giu ban goc.
3. Can loi: Whisper large-v3 (moc thoi gian tung tu) + loi co san cua nguoi dung (file .txt cung ten) khop THEO KY TU
   (dung ca cho tieng Nhat / Han / Trung khong co dau cach). Khong co loi -> dung loi Whisper.
4. Chon doan gom CAC CAU TRON VEN dai trong khoang (mac dinh 15-20 s), cat cuoi o KHOANG LANG THAT (>= 0,2 s, nang luong 10 ms
   duoi nguong) sau khi am cuoi cua tu da tat; giu ~0,3 s lang, lam mo 40 ms. Dau doan: 0,15 s truoc tieng dau tien.
5. Kiem tra lai: Whisper nghe doan da cat, so voi loi -> thieu tu cuoi / lech loi thi chon doan khac.
Ra: <ten>_chuan.wav (mono 44,1 kHz), <ten>_chuan.txt (loi DUNG cua doan), <ten>_chuan.json (bao cao).
Dung: python ref_voice_clean.py <file am thanh> [--txt loi.txt] [--lang tl] [--min 8] [--max 15] [--out thu_muc] [--khong-lam-sach] [--manh]
  --manh: LOC MANH - chay du 3 buoc (tach nhac nen, khu vang/om, khu tap am) tru khi buoc do lam mat giong (> 8 dB) hoac
          lam giong te di ro (DNSMOS tong giam > 0,15). Mac dinh (tu dong): chi giu buoc nao DNSMOS do thay tot hon ro.
Dong tien do cho giao dien: @@P|<phan tram>|<viec>   ket thuc: @@XONG|<duong dan wav>   loi: @@LOI|<thong bao>"""
import os, sys, re, json, argparse, difflib, tempfile, shutil, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, librosa

MODEL_DIR = r"D:\Download\Model\giong_sach_models"
WHISPER_DIR = r"D:\Download\Model\faster_whisper"
SR = 44100
M_TACH = "mel_band_roformer_kim_ft_unwa.ckpt"
M_VANG = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
M_ON = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
KET_CAU = "。！？!?….．"          # ky tu ket cau (dau cham '.' xu ly rieng de khong cat '3.5' / 'Mr.')

def P(pct, msg): print(f"@@P|{int(pct)}|{msg}", flush=True)

# ---------------------------------------------------------------- DNSMOS P.835 (Microsoft, dnsmos_local.py)
class Dnsmos:
    def __init__(self):
        import onnxruntime as ort
        self.s = ort.InferenceSession(os.path.join(MODEL_DIR, "sig_bak_ovr.onnx"), providers=["CPUExecutionProvider"])
    def __call__(self, y16):
        n = 144160
        if len(y16) < n: y16 = np.tile(y16, int(np.ceil(n / max(1, len(y16)))))[:max(n, len(y16))]
        out = []
        for a in range(0, max(1, len(y16) - n + 1), 16000):
            seg = y16[a:a + n].astype(np.float32)
            if len(seg) < n: break
            sig, bak, ovr = self.s.run(None, {"input_1": seg[None]})[0][0]
            out.append((np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig), np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak),
                        np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovr)))
        m = np.mean(out, 0)
        return {"SIG": round(float(m[0]), 2), "BAK": round(float(m[1]), 2), "OVRL": round(float(m[2]), 2)}

def rms_db(x): return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))

# ---------------------------------------------------------------- lam sach
def tach(sep_cache, model, y, tmp):
    """-> {ten_stem_thuong: mang} (mono 44,1 kHz)"""
    from audio_separator.separator import Separator
    if model not in sep_cache:
        s = Separator(model_file_dir=MODEL_DIR, output_dir=tmp, output_format="WAV", sample_rate=SR, log_level=40)
        s.load_model(model_filename=model); sep_cache[model] = s
    s = sep_cache[model]; s.output_dir = tmp
    vao = os.path.join(tmp, f"vao_{len(os.listdir(tmp))}.wav"); sf.write(vao, np.stack([y, y], 1), SR)
    outs = s.separate(vao); kq = {}
    for f in outs:
        p = f if os.path.isabs(f) else os.path.join(tmp, f)
        a, r = sf.read(p, dtype="float32"); a = a.mean(1) if a.ndim > 1 else a
        if r != SR: a = librosa.resample(a, orig_sr=r, target_sr=SR)
        ten = re.findall(r"\(([^)]+)\)", os.path.basename(p)); kq[(ten[0] if ten else os.path.basename(p)).lower()] = a[:len(y)]
    return kq

# (buoc, model, ten phan GIONG giu lai). Model khu vang: UVR-DeEcho-DeReverb (thu 23-24/11/2026 tren giong noi: model duy nhat
# khu duoc vang ma khong lam mat loi; cac model Roformer khu vang danh cho giong hat - khong nhan ra vang phong cua giong noi)
BUOC = (("tách giọng khỏi nhạc nền / tiếng nền", M_TACH, "vocals"),
        ("khử vang / ồm", "UVR-DeEcho-DeReverb.pth", "no reverb"),
        ("khử tạp âm", M_ON, "dry"))

def tot_hon(d, c):
    """ban moi d tot hon ro so voi c (DNSMOS): tong diem, hoac giong (SIG), hoac nen sach (BAK) - ma khong kem phan kia"""
    return (d["OVRL"] >= c["OVRL"] + 0.1 or (d["SIG"] >= c["SIG"] + 0.3 and d["BAK"] >= c["BAK"] - 0.15)
            or (d["BAK"] >= c["BAK"] + 0.4 and d["SIG"] >= c["SIG"] - 0.15))

def lam_sach(y, log, manh=False):
    """Moi buoc: chay model, lay phan GIONG; chi giu khi giong khong bi mat (nang luong giam < 8 dB) va DNSMOS tot hon ro."""
    dns = Dnsmos(); y16 = lambda a: librosa.resample(a, orig_sr=SR, target_sr=16000)
    cur = y; dcur = dns(y16(y)); log["dnsmos_goc"] = dcur; buoc, thu = [], []
    tmp = tempfile.mkdtemp(prefix="giong_sach_"); cache = {}
    try:
        for k, (ten, model, stem) in enumerate(BUOC):
            P(10 + 15 * k, f"kiểm tra: {ten}")
            st = tach(cache, model, cur, tmp)
            c = st.get(stem)
            if c is None: thu.append(f"{ten}: model không trả phần '{stem}' ({list(st)})"); continue
            c = c.astype(np.float32)
            mat = rms_db(cur) - rms_db(c)
            if mat > 8: thu.append(f"{ten}: bỏ (phần giữ lại nhỏ hơn {mat:.1f} dB - mất giọng)"); continue
            d = dns(y16(c))
            if tot_hon(d, dcur) or (manh and d["OVRL"] >= dcur["OVRL"] - 0.15 and d["SIG"] >= dcur["SIG"] - 0.2):
                buoc.append(f"{ten}: DNSMOS tổng {dcur['OVRL']} -> {d['OVRL']} (giọng {dcur['SIG']} -> {d['SIG']}, nền {dcur['BAK']} -> {d['BAK']})")
                cur, dcur = c, d
            else:
                thu.append(f"{ten}: không cần (DNSMOS {dcur['OVRL']} -> {d['OVRL']})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log["dnsmos_sau_lam_sach"] = dcur; log["da_thu"] = thu
    log["lam_sach"] = buoc or ["giọng đã sạch, không cần xử lý"]
    return cur

# ---------------------------------------------------------------- can loi
def chu(c): return c.isalnum()
def nl(s): return unicodedata.normalize("NFKC", s).lower()

def can_loi(words, loi):
    """words: [(chu, bat_dau, ket_thuc)] cua Whisper; loi: chuoi loi nguoi dung (hoac None).
    -> danh sach cau: {text, t0, t1, khop} (t0 = bat dau tu dau, t1 = ket thuc tu cuoi; None neu khong can duoc)"""
    wc, wi = [], []                                   # ky tu Whisper + chi so tu chua no
    for i, (w, a, b) in enumerate(words):
        for c in nl(w):
            if chu(c): wc.append(c); wi.append(i)
    if not loi:
        loi = "".join(w for w, a, b in words)
    lc, li = [], []                                   # ky tu cua loi (chi chu/so) + vi tri trong chuoi goc
    for k, c in enumerate(nl(loi)):
        if chu(c): lc.append(c); li.append(k)
    m = {}
    for t, a, b, c, d in difflib.SequenceMatcher(None, lc, wc, autojunk=False).get_opcodes():
        if t == "equal":
            for o in range(b - a): m[a + o] = wi[c + o]
        elif t == "replace" and (b - a) == (d - c):   # sai chinh ta tung ky tu: van khop vi tri
            for o in range(b - a): m[a + o] = wi[c + o]
    # tach cau trong loi goc
    cau, bd = [], 0
    for mt in re.finditer(r"([" + re.escape(KET_CAU) + r"]+|(?<!\d)\.(?!\d)|\n+)[\"'”’»」』)\]]*\s*", loi):
        cau.append((bd, mt.end())); bd = mt.end()
    if bd < len(loi): cau.append((bd, len(loi)))
    out = []
    for a, b in cau:
        txt = loi[a:b].strip()
        idx = [k for k, p in enumerate(li) if a <= p < b]
        if not txt or not idx: continue
        dau = [m[k] for k in idx[:3] if k in m]; cuoi = [m[k] for k in idx[-3:] if k in m]
        khop = sum(k in m for k in idx) / len(idx)
        out.append({"text": txt, "t0": words[dau[0]][1] if dau else None, "t1": words[cuoi[-1]][2] if cuoi else None,
                    "khop": round(khop, 3), "n": len(idx)})
    return out

# ---------------------------------------------------------------- cat chinh xac theo nang luong
def khung_db(y, hop):
    n = len(y) // hop; f = y[:n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)

def diem_cat(db, hop_s, t_cuoi, nguong, t_tiep):
    """Sau tu cuoi: di toi khi het tieng (duoi nguong), tim doan lang >= 0,2 s truoc tieng tiep theo -> (cat, do_dai_lang) hoac None."""
    i = int(t_cuoi / hop_s); n = len(db)
    het = min(n, int((t_cuoi + 0.6) / hop_s))
    while i < het and db[i] > nguong: i += 1          # duoi am cua tu cuoi (Whisper hay bao som)
    if i >= n: return (n * hop_s, 0.5)                  # het file = lang
    j = i
    gioi_han = n if t_tiep is None else min(n, int(t_tiep / hop_s))
    while j < gioi_han and db[j] <= nguong: j += 1
    lang = (j - i) * hop_s
    if lang < 0.2: return None
    return (i * hop_s + min(0.3, lang * 0.6), lang)

def diem_dau(db, hop_s, t_dau, nguong):
    """Dau cau: lui qua tieng dau tien (Whisper hay bao muon) toi KHOANG LANG THAT truoc no (>= 0,1 s, hoac dau file),
    cat ben trong khoang lang do, 0,15 s truoc tieng. Khong co khoang lang (dinh lien cau truoc) -> None (khong dung lam diem dau)."""
    k = min(len(db) - 1, int((t_dau + 0.1) / hop_s))
    while k > 0 and db[k] > nguong: k -= 1                 # lui het tieng cua tu dau
    on = k + 1                                             # khung co tieng dau tien
    while k > 0 and db[k] <= nguong: k -= 1                # do dai khoang lang phia truoc
    lang = (on - k - 1) * hop_s
    if k > 0 and lang < 0.1: return None
    return max(0.0, on * hop_s - min(0.15, lang * 0.7 if k > 0 else 0.15))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vao"); ap.add_argument("--txt"); ap.add_argument("--lang"); ap.add_argument("--out")
    ap.add_argument("--min", type=float, default=8.0); ap.add_argument("--max", type=float, default=15.0)
    ap.add_argument("--khong-lam-sach", action="store_true")
    ap.add_argument("--manh", action="store_true")
    a = ap.parse_args()
    ten = os.path.splitext(os.path.basename(a.vao))[0]
    out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.vao)), "giong_mau_chuan"); os.makedirs(out_dir, exist_ok=True)
    txt_path = a.txt or os.path.splitext(a.vao)[0] + ".txt"
    loi = open(txt_path, encoding="utf-8-sig").read().strip() if os.path.exists(txt_path) else ""
    log = {"file_goc": a.vao, "loi_nguoi_dung": bool(loi), "khoang_muon_s": [a.min, a.max]}

    P(2, "đọc file")
    y, _ = librosa.load(a.vao, sr=SR, mono=True); log["giay_goc"] = round(len(y) / SR, 2)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    sach = y if a.khong_lam_sach else lam_sach(y, log, manh=a.manh)
    log["che_do"] = "khong_lam_sach" if a.khong_lam_sach else ("manh" if a.manh else "tu_dong")
    # ban da lam sach CA FILE (chua cat) de nguoi dung nghe so sanh truoc / sau
    try: sf.write(os.path.join(out_dir, f"{ten}_sach_full.wav"), (sach / (np.abs(sach).max() + 1e-9) * 0.89).astype(np.float32), SR)
    except Exception: pass

    P(60, "Whisper nghe + căn lời")
    from faster_whisper import WhisperModel
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wh = WhisperModel("large-v3", device=dev, compute_type="float16" if dev == "cuda" else "int8", download_root=WHISPER_DIR)
    y16 = librosa.resample(sach, orig_sr=SR, target_sr=16000)
    segs, info = wh.transcribe(y16, language=a.lang, beam_size=5, word_timestamps=True, condition_on_previous_text=False, vad_filter=False)
    words = [(w.word, w.start, w.end) for s in segs for w in s.words]
    lang = a.lang or info.language; log["ngon_ngu"] = lang
    if not words: print("@@LOI|Whisper không nghe ra lời nào trong file", flush=True); sys.exit(2)
    log["loi_whisper"] = "".join(w for w, _, _ in words).strip()
    cau = can_loi(words, loi or None)
    if loi:
        log["khop_loi_%"] = round(100 * sum(c["khop"] * c["n"] for c in cau) / max(1, sum(c["n"] for c in cau)), 1)

    # nang luong tung 10 ms; nguong = nen + 12 dB (nen = phan vi 10 cac khung) nhung khong thap hon dinh - 45 dB
    hop = int(SR * 0.01); db = khung_db(sach, hop); hop_s = hop / SR
    nen = float(np.percentile(db, 10)); nguong = max(nen + 12.0, float(db.max()) - 45.0)
    log["nen_dB"] = round(nen, 1); log["nguong_dB"] = round(nguong, 1)

    # moi cau: diem cat cuoi hop le (co khoang lang that)
    for k, c in enumerate(cau):
        t_tiep = next((x["t0"] for x in cau[k + 1:] if x["t0"] is not None), None)
        c["cat"] = diem_cat(db, hop_s, c["t1"], nguong, t_tiep) if c["t1"] is not None else None
    ung = []
    for i, ci in enumerate(cau):
        if ci["t0"] is None or ci["khop"] < 0.7: continue
        bd = diem_dau(db, hop_s, ci["t0"], nguong)
        if bd is None: continue
        for j in range(i, len(cau)):
            cj = cau[j]
            if cj["khop"] < 0.7 or cj["t1"] is None: break
            if cj["cat"] is None: continue
            dai = cj["cat"][0] - bd
            if dai > a.max + 6: break
            gap = max([(cau[k + 1]["t0"] or 0) - (cau[k]["t1"] or 0) for k in range(i, j)] or [0])
            diem = abs(dai - (a.min + a.max) / 2) + (0 if a.min <= dai <= a.max else 3 + abs(dai - min(max(dai, a.min), a.max)) * 2) \
                   + (2.0 if gap > 1.5 else 0) + 0.15 * i + (1 - min(cj["khop"], ci["khop"])) * 10
            ung.append((diem, i, j, bd, cj["cat"][0], dai))
    if not ung:                                          # file ngan / khong co cau tron ven: lay ca doan noi (bo lang 2 dau)
        co = np.where(db > nguong)[0]
        bd = max(0.0, co[0] * hop_s - 0.15); kt = min(len(sach) / SR, (co[-1] + 1) * hop_s + 0.3)
        ung = [(99, 0, len(cau) - 1, bd, kt, kt - bd)]; log["canh_bao"] = "không có đoạn gồm câu trọn vẹn đúng độ dài -> lấy cả đoạn có tiếng"
    ung.sort()

    P(85, "cắt + Whisper kiểm tra lại")
    chon = None
    for diem, i, j, bd, kt, dai in ung[:6]:
        doan = sach[int(bd * SR):int(kt * SR)].copy()
        fi, fo = int(0.015 * SR), int(0.04 * SR)
        doan[:fi] *= np.linspace(0, 1, fi); doan[-fo:] *= np.linspace(1, 0, fo)
        text = " ".join(c["text"] for c in cau[i:j + 1]) if cau else log["loi_whisper"]
        if lang in ("ja", "zh", "yue"): text = "".join(c["text"] for c in cau[i:j + 1])
        sg, _ = wh.transcribe(librosa.resample(doan, orig_sr=SR, target_sr=16000), language=lang, beam_size=5, condition_on_previous_text=False)
        nghe = "".join(s.text for s in sg)
        a_, b_ = [c for c in nl(text) if chu(c)], [c for c in nl(nghe) if chu(c)]
        sm = difflib.SequenceMatcher(None, a_, b_, autojunk=False)
        cer = 1 - sm.ratio(); duoi = "".join(a_[-4:]); co_duoi = duoi in "".join(b_[-10:]) or sm.find_longest_match(len(a_) - 4, len(a_), max(0, len(b_) - 12), len(b_)).size >= 3
        ops = sm.get_opcodes()
        sach_dau = not (ops and ops[0][0] in ("insert", "replace") and ops[0][4] - ops[0][3] > 2 and ops[0][1] == 0)  # khong dinh chu cau truoc
        tot = co_duoi and sach_dau
        if chon is None or (tot and not chon[1]) or (tot == chon[1] and cer < chon[0]):
            chon = (cer, tot, doan, text, nghe, bd, kt, dai, co_duoi, sach_dau)
        if cer <= 0.08 and tot: break
    cer, _tot, doan, text, nghe, bd, kt, dai, co_duoi, sach_dau = chon
    log["sach_dau"] = bool(sach_dau)
    doan = doan / (np.abs(doan).max() + 1e-9) * 0.89
    wav = os.path.join(out_dir, f"{ten}_chuan.wav"); sf.write(wav, doan, SR)
    open(os.path.join(out_dir, f"{ten}_chuan.txt"), "w", encoding="utf-8").write(text)
    log.update({"doan_s": [round(bd, 2), round(kt, 2)], "giay": round(dai, 2), "loi_doan": text, "whisper_nghe_lai": nghe,
                "lech_ky_tu_%": round(100 * cer, 1), "du_tu_cuoi": bool(co_duoi), "file_ra": wav,
                "dnsmos_ra": Dnsmos()(librosa.resample(doan, orig_sr=SR, target_sr=16000))})
    cb = [log["canh_bao"]] if log.get("canh_bao") else []
    if log["dnsmos_ra"]["OVRL"] < 2.8:
        cb.append(f"giọng mẫu sau khi làm sạch vẫn kém (DNSMOS {log['dnsmos_ra']['OVRL']}/5) — nên dùng bản thu khác sạch hơn")
    if dai < a.min - 0.5 or dai > a.max + 0.5:
        cb.append(f"đoạn dài {dai:.1f} s, ngoài khoảng {a.min:g}-{a.max:g} s (không có đoạn gồm câu trọn vẹn đúng độ dài)")
    if cer > 0.08 or not co_duoi or not sach_dau:
        cb.append(f"Whisper nghe lại lệch {100 * cer:.0f}% so với lời" + ("" if co_duoi else ", có thể thiếu từ cuối") + ("" if sach_dau else ", đầu đoạn dính chữ câu trước"))
    log["canh_bao"] = cb
    json.dump(log, open(os.path.join(out_dir, f"{ten}_chuan.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    P(100, "xong")
    print(json.dumps(log, ensure_ascii=False, indent=1), flush=True)
    print(f"@@XONG|{wav}", flush=True)

if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"@@LOI|{type(e).__name__}: {str(e)[:300]}", flush=True); sys.exit(1)
