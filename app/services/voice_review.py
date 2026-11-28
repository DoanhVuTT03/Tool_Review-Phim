"""Judges one generated voice line from what Whisper heard: wrong / missing / extra / joined words, cut last word,
abnormal length (loop or swallowed words), hit length cap, too long for its time slot.

Whisper itself is imperfect (Filipino ~5 % word errors on clean speech), so near-identical spellings
("ng" / "nang", "po." / "po", hyphens) are not counted, and a line is flagged on what really matters:
any missing word, a missing LAST word, >= 2 extra words, or >= 15 % wrong words.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from app.services.length_fit import lang_key, reading_rate, speech_units

_CHAR_LANGS = {"zh", "ja", "ko", "th"}
WHISPER_LANG = {"fil": "tl"}


def whisper_lang(lang: str) -> str:
    k = lang_key(lang)
    return WHISPER_LANG.get(k, k)


# Spellings Whisper uses for words that SOUND the same (not reading errors). Filipino: "nang" and "ng" are both
# read /naŋ/; colloquial short forms (ganun / yun / meron...) are how the formal words are really pronounced.
_SAME_SOUND = {
    "fil": {"nang": "ng", "ganun": "ganoon", "ganon": "ganoon", "yun": "iyon", "yon": "iyon", "yan": "iyan",
            "dun": "doon", "don": "doon", "meron": "mayroon", "pwede": "puwede", "pano": "paano", "kanya": "kaniya",
            "sya": "siya", "nya": "niya", "nyo": "ninyo", "ninyo": "ninyo", "kayo": "kayo", "talaga": "talaga",
            "ho": "po", "opo": "opo", "oho": "opo", "di": "hindi", "hinde": "hindi", "eto": "ito", "to": "ito"},
}


def _tokens(s: str, lang: str) -> List[str]:
    s = unicodedata.normalize("NFC", s or "").lower().replace("-", " ").replace("’", "'")
    s = re.sub(r"[^\w' ]+", " ", s)
    if lang_key(lang) in _CHAR_LANGS:
        return [c for c in s if not c.isspace()]
    alias = _SAME_SOUND.get(lang_key(lang), {})
    return [alias.get(w, w) for w in (w.strip("'") for w in s.split()) if w]


def _sound_key(w: str) -> str:
    """Rough pronunciation key for Latin-script words: foreign names are spelled many ways by Whisper
    (Markes / Marques, qi / ki, lingzhi / lingji) while the voice read them the same."""
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if not unicodedata.combining(c))
    for a, b in (("qu", "k"), ("q", "k"), ("ck", "k"), ("c", "k"), ("zh", "j"), ("dz", "j"), ("ph", "f"), ("f", "p"),
                 ("v", "b"), ("z", "s"), ("x", "s"), ("y", "i"), ("w", "u"), ("ee", "i"), ("oo", "u"), ("h", "")):
        w = w.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", w)


def _similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if any(ch.isdigit() for ch in a + b):  # "3" vs "tatlo": Whisper writes numbers its own way
        return True
    if difflib.SequenceMatcher(None, a, b).ratio() >= 0.8:
        return True
    ka, kb = _sound_key(a), _sound_key(b)
    return ka == kb or (min(len(ka), len(kb)) >= 4 and difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.8)


def analyze(text: str, heard: str, dur: float, hit_cap: bool, lang: str, allow: float, max_tempo: float) -> Dict[str, Any]:
    """-> {"err": word error ratio, "reasons": [...], "flag": bool, "tempo": speed-up the mixer will need}"""
    ref, hyp = _tokens(text, lang), _tokens(heard, lang)
    char = lang_key(lang) in _CHAR_LANGS
    sep = "" if char else " "
    reasons: List[str] = []
    errors = 0.0
    missing: List[str] = []
    extra: List[str] = []
    wrong: List[Tuple[str, str]] = []
    joined: List[Tuple[str, str]] = []
    last_missing = ""
    if ref and not hyp:
        reasons.append("Whisper không nghe ra lời nào (im lặng / tiếng lạ)")
        errors = len(ref)
    else:
        for tag, a, b, c, d in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            r, h = ref[a:b], hyp[c:d]
            if tag == "delete":
                missing.extend(r)
                errors += len(r)
                if b == len(ref):
                    last_missing = sep.join(r)
            elif tag == "insert":
                extra.extend(h)
                errors += len(h) * (0.5 if len(h) == 1 else 1.0)
            else:
                rj, hj = "".join(r), "".join(h)
                if not char and rj == hj and len(r) != len(h):
                    joined.append((sep.join(r), sep.join(h)))
                    errors += 0.5
                elif _similar(rj, hj):
                    continue
                else:
                    wrong.append((sep.join(r), sep.join(h)))
                    errors += max(len(r), len(h))
                    if b == len(ref) and len(h) < len(r):
                        last_missing = sep.join(r[len(h):])
    n = max(1, len(ref))
    err = errors / n
    if missing:
        reasons.append(f"Thiếu từ: «{sep.join(missing)}»")
    if last_missing:
        reasons.append(f"Mất / cụt từ cuối: «{last_missing}»")
    if wrong:
        reasons.append("Đọc sai: " + "; ".join(f"«{a}» → «{b}»" for a, b in wrong[:4]))
    if joined:
        reasons.append("Dính / tách từ: " + "; ".join(f"«{a}» → «{b}»" for a, b in joined[:4]))
    if extra:
        reasons.append(f"Thừa / lặp từ: «{sep.join(extra)}»")
    # length checks (expected = natural reading time of the text)
    expected = speech_units(text, lang) / reading_rate(lang)
    long_abn = dur > expected * 2.0 + 0.8
    short_abn = expected > 1.2 and dur < expected * 0.5
    if long_abn:
        reasons.append(f"Dài bất thường {dur:.1f}s (đọc tự nhiên ~{expected:.1f}s): có thể đọc lặp / kéo dài / im lặng")
    if short_abn:
        reasons.append(f"Ngắn bất thường {dur:.1f}s (đọc tự nhiên ~{expected:.1f}s): có thể nuốt từ")
    if hit_cap:
        reasons.append("Chạm trần độ dài: có thể bị cắt cụt hoặc đọc lan man")
    tempo = dur / max(0.2, allow) if allow else 1.0
    if tempo > max_tempo * 1.05:
        reasons.append(f"Dài quá khung: {dur:.1f}s / khung {allow:.1f}s (cần nén {tempo:.2f}x, tối đa {max_tempo:.2f}x)")
    # reading problems: a new version (other seed) can fix them -> automatic retry
    read_bad = bool(
        (ref and not hyp)
        or missing
        or last_missing
        or len(extra) >= 2
        or err >= 0.15
        or long_abn
        or short_abn
        or hit_cap
    )
    # too long for its slot: a new version does not help (shorter text does, length_fit) -> only listed when
    # the mixer would have to rush it audibly
    flag = read_bad or tempo > max(1.6, max_tempo * 1.3)
    return {"err": round(err, 3), "reasons": reasons, "flag": flag, "read_bad": read_bad, "tempo": round(tempo, 2)}


def score(t: Dict[str, Any]) -> Tuple:
    """Sort key of a generated version: not flagged, fewer errors, fits its slot, then shorter."""
    return (bool(t.get("flag")), float(t.get("err") or 0.0), max(0.0, float(t.get("tempo") or 1.0) - 1.15), float(t.get("dur") or 0.0))


def expected_seconds(text: str, lang: str) -> float:
    return speech_units(text, lang) / reading_rate(lang)


__all__ = ["analyze", "score", "whisper_lang", "expected_seconds"]
