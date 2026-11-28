from __future__ import annotations
import asyncio
import base64
import hashlib
import io
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import edge_tts
import httpx

from app.config import PROJECTS_DIR
from app.models import SubtitleCue, VideoProject, VoiceConfig
from app.services.media import parse_srt


ProgressCallback = Callable[[float, str], None]

# Language code aliases so 'fil' (Filipino) and 'tl' (Tagalog) always match seamlessly!
_LANG_ALIASES: Dict[str, set[str]] = {
    "fil": {"fil", "tl"},
    "tl": {"fil", "tl"},
    "zh": {"zh"},
}

# Verified live CapCut / TikTok Free TTS voices
CAPCUT_TIKTOK_FREE_VOICES: List[Dict[str, Any]] = [
    # Vietnamese
    {"id": "BV074_streaming", "name": "CapCut VN - Cô Gái Hoạt Bát (Nữ)", "lang": "vi", "gender": "Female", "style": "Review / Sôi nổi", "free": True, "provider": "capcut_tiktok"},
    {"id": "BV075_streaming", "name": "CapCut VN - Thanh Niên Tự Tin (Nam)", "lang": "vi", "gender": "Male", "style": "Review / Thuyết minh", "free": True, "provider": "capcut_tiktok"},
    # Volcengine (CapCut's voice library) multilingual voice 天才少女: cn/en/ja/pt/es/id/vi/th - verified via relay.
    # (Volcengine has NO Filipino/Tagalog voice: for Filipino use Edge-TTS fil-PH-AngeloNeural / BlessicaNeural.)
    {"id": "BV421_streaming", "name": "CapCut Đa ngữ - Thiên Tài Thiếu Nữ (Nữ)", "lang": "vi", "gender": "Female", "style": "Đa ngôn ngữ / Trẻ trung", "free": True, "provider": "capcut_tiktok"},
    # English US / UK / AU
    {"id": "en_us_001", "name": "TikTok US - Jessie (Female)", "lang": "en", "gender": "Female", "style": "Viral / Clear", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_us_002", "name": "TikTok US - Female 2", "lang": "en", "gender": "Female", "style": "Natural", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_us_006", "name": "TikTok US - Joey (Male)", "lang": "en", "gender": "Male", "style": "Storyteller", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_us_007", "name": "TikTok US - Professor (Male)", "lang": "en", "gender": "Male", "style": "Calm Narrator", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_us_009", "name": "TikTok US - Scientist (Male)", "lang": "en", "gender": "Male", "style": "Deep", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_us_010", "name": "TikTok US - Confidence (Male)", "lang": "en", "gender": "Male", "style": "Energetic", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_male_narration", "name": "CapCut EN - Storyteller Narrator (Male)", "lang": "en", "gender": "Male", "style": "Movie Recap", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_male_cody", "name": "CapCut EN - Serious Cody (Male)", "lang": "en", "gender": "Male", "style": "Serious / Review", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_female_emotional", "name": "CapCut EN - Peaceful Emotional (Female)", "lang": "en", "gender": "Female", "style": "Emotional", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_female_samc", "name": "CapCut EN - Empathetic (Female)", "lang": "en", "gender": "Female", "style": "Warm", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_male_funny", "name": "CapCut EN - Wacky Funny (Male)", "lang": "en", "gender": "Male", "style": "Comedy", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_uk_001", "name": "TikTok UK - British Narrator (Male)", "lang": "en", "gender": "Male", "style": "Documentary", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_uk_003", "name": "TikTok UK - Male 2", "lang": "en", "gender": "Male", "style": "Classic", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_au_001", "name": "TikTok AU - Metro (Female)", "lang": "en", "gender": "Female", "style": "Casual", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_au_002", "name": "TikTok AU - Smooth (Male)", "lang": "en", "gender": "Male", "style": "Smooth", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_us_ghostface", "name": "TikTok Character - Ghostface", "lang": "en", "gender": "Male", "style": "Horror / Dramatic", "free": True, "provider": "capcut_tiktok"},
    {"id": "en_male_pirate", "name": "TikTok Character - Pirate", "lang": "en", "gender": "Male", "style": "Character", "free": True, "provider": "capcut_tiktok"},
    # Chinese
    {"id": "BV001_streaming", "name": "CapCut ZH - 通用女声 (Nữ Chuẩn)", "lang": "zh", "gender": "Female", "style": "Standard", "free": True, "provider": "capcut_tiktok"},
    {"id": "BV002_streaming", "name": "CapCut ZH - 通用男声 (Nam Chuẩn)", "lang": "zh", "gender": "Male", "style": "Standard", "free": True, "provider": "capcut_tiktok"},
    {"id": "BV700_streaming", "name": "CapCut ZH - 灿灿 (Nữ Giải Thích Phim)", "lang": "zh", "gender": "Female", "style": "Movie Recap", "free": True, "provider": "capcut_tiktok"},
    {"id": "BV701_streaming", "name": "CapCut ZH - 擎苍 (Nam Thuyết Minh)", "lang": "zh", "gender": "Male", "style": "Movie Recap", "free": True, "provider": "capcut_tiktok"},
    {"id": "BV411_streaming", "name": "CapCut ZH - 影视解说小帅 (Nam Thuyết Minh Phim)", "lang": "zh", "gender": "Male", "style": "Movie Recap", "free": True, "provider": "capcut_tiktok"},
    {"id": "BV421_streaming", "name": "CapCut ZH - 天才少女 (Nữ Đa ngữ)", "lang": "zh", "gender": "Female", "style": "Đa ngôn ngữ", "free": True, "provider": "capcut_tiktok"},
    {"id": "BV005_streaming", "name": "CapCut ZH - 活泼女声 (Nữ Hoạt Bát)", "lang": "zh", "gender": "Female", "style": "Lively", "free": True, "provider": "capcut_tiktok"},
    # Japanese
    {"id": "jp_001", "name": "TikTok JP - Miho (Female)", "lang": "ja", "gender": "Female", "style": "Standard", "free": True, "provider": "capcut_tiktok"},
    {"id": "jp_003", "name": "TikTok JP - Keiko (Female)", "lang": "ja", "gender": "Female", "style": "Soft", "free": True, "provider": "capcut_tiktok"},
    {"id": "jp_005", "name": "TikTok JP - Sakura (Female)", "lang": "ja", "gender": "Female", "style": "Bright", "free": True, "provider": "capcut_tiktok"},
    {"id": "jp_006", "name": "TikTok JP - Naoki (Male)", "lang": "ja", "gender": "Male", "style": "Narrator", "free": True, "provider": "capcut_tiktok"},
    {"id": "jp_male_osamu", "name": "CapCut JP - Osamu (Male)", "lang": "ja", "gender": "Male", "style": "Calm", "free": True, "provider": "capcut_tiktok"},
    # Korean
    {"id": "kr_002", "name": "TikTok KR - Male 1", "lang": "ko", "gender": "Male", "style": "Narrator", "free": True, "provider": "capcut_tiktok"},
    {"id": "kr_003", "name": "TikTok KR - Female", "lang": "ko", "gender": "Female", "style": "Clear", "free": True, "provider": "capcut_tiktok"},
    {"id": "kr_004", "name": "TikTok KR - Male 2", "lang": "ko", "gender": "Male", "style": "Energetic", "free": True, "provider": "capcut_tiktok"},
    # Indonesian
    {"id": "id_001", "name": "TikTok ID - Female", "lang": "id", "gender": "Female", "style": "Natural", "free": True, "provider": "capcut_tiktok"},
    # Spanish
    {"id": "es_002", "name": "TikTok ES - Spain Male", "lang": "es", "gender": "Male", "style": "Narrator", "free": True, "provider": "capcut_tiktok"},
    {"id": "es_mx_002", "name": "TikTok ES - Mexico Male", "lang": "es", "gender": "Male", "style": "Review", "free": True, "provider": "capcut_tiktok"},
    # Portuguese
    {"id": "br_001", "name": "TikTok PT-BR - Female 1", "lang": "pt", "gender": "Female", "style": "Standard", "free": True, "provider": "capcut_tiktok"},
    {"id": "br_003", "name": "TikTok PT-BR - Female 2", "lang": "pt", "gender": "Female", "style": "Warm", "free": True, "provider": "capcut_tiktok"},
    {"id": "br_005", "name": "TikTok PT-BR - Male", "lang": "pt", "gender": "Male", "style": "Narrator", "free": True, "provider": "capcut_tiktok"},
    # French
    {"id": "fr_001", "name": "TikTok FR - Male 1", "lang": "fr", "gender": "Male", "style": "Narrator", "free": True, "provider": "capcut_tiktok"},
    {"id": "fr_002", "name": "TikTok FR - Male 2", "lang": "fr", "gender": "Male", "style": "Casual", "free": True, "provider": "capcut_tiktok"},
    # German
    {"id": "de_001", "name": "TikTok DE - Female", "lang": "de", "gender": "Female", "style": "Clear", "free": True, "provider": "capcut_tiktok"},
    {"id": "de_002", "name": "TikTok DE - Male", "lang": "de", "gender": "Male", "style": "Narrator", "free": True, "provider": "capcut_tiktok"},
]

# Free Google Translate TTS voices (supports Tagalog/Filipino & 50+ languages, zero API key!)
GTTS_FREE_VOICES: List[Dict[str, Any]] = [
    {"id": "gtts:tl", "name": "Google TTS Free - Filipino / Tagalog", "lang": "fil", "gender": "Female", "style": "Free chuẩn Tagalog", "free": True, "provider": "gtts"},
    {"id": "gtts:vi", "name": "Google TTS Free - Tiếng Việt", "lang": "vi", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:en", "name": "Google TTS Free - English (US)", "lang": "en", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:en-ph", "name": "Google TTS Free - English (Philippines)", "lang": "fil", "gender": "Female", "style": "English-PH accent", "free": True, "provider": "gtts"},
    {"id": "gtts:zh-CN", "name": "Google TTS Free - 中文 (Giản thể)", "lang": "zh", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:ja", "name": "Google TTS Free - 日本語", "lang": "ja", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:ko", "name": "Google TTS Free - 한국어", "lang": "ko", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:th", "name": "Google TTS Free - ไทย", "lang": "th", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:id", "name": "Google TTS Free - Bahasa Indonesia", "lang": "id", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:es", "name": "Google TTS Free - Español", "lang": "es", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:pt", "name": "Google TTS Free - Português", "lang": "pt", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:fr", "name": "Google TTS Free - Français", "lang": "fr", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:de", "name": "Google TTS Free - Deutsch", "lang": "de", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:ru", "name": "Google TTS Free - Русский", "lang": "ru", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:ar", "name": "Google TTS Free - العربية", "lang": "ar", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
    {"id": "gtts:hi", "name": "Google TTS Free - हिन्दी", "lang": "hi", "gender": "Female", "style": "Free chuẩn", "free": True, "provider": "gtts"},
]

_EDGE_VOICES_CACHE: Optional[List[Dict[str, Any]]] = None

_EDGE_FALLBACK_MAP: Dict[Tuple[str, str], str] = {
    ("vi", "Female"): "vi-VN-HoaiMyNeural",
    ("vi", "Male"): "vi-VN-NamMinhNeural",
    ("fil", "Female"): "fil-PH-BlessicaNeural",
    ("fil", "Male"): "fil-PH-AngeloNeural",
    ("tl", "Female"): "fil-PH-BlessicaNeural",
    ("tl", "Male"): "fil-PH-AngeloNeural",
    ("en", "Female"): "en-US-AvaMultilingualNeural",
    ("en", "Male"): "en-US-AndrewMultilingualNeural",
    ("zh", "Female"): "zh-CN-XiaoxiaoNeural",
    ("zh", "Male"): "zh-CN-YunxiNeural",
    ("ja", "Female"): "ja-JP-NanamiNeural",
    ("ja", "Male"): "ja-JP-KeitaNeural",
    ("ko", "Female"): "ko-KR-SunHiNeural",
    ("ko", "Male"): "ko-KR-InJoonNeural",
    ("th", "Female"): "th-TH-PremwadeeNeural",
    ("th", "Male"): "th-TH-NiwatNeural",
    ("id", "Female"): "id-ID-GadisNeural",
    ("id", "Male"): "id-ID-ArdiNeural",
    ("es", "Female"): "es-ES-ElviraNeural",
    ("es", "Male"): "es-MX-JorgeNeural",
    ("pt", "Female"): "pt-BR-FranciscaNeural",
    ("pt", "Male"): "pt-BR-AntonioNeural",
    ("fr", "Female"): "fr-FR-VivienneMultilingualNeural",
    ("fr", "Male"): "fr-FR-RemyMultilingualNeural",
    ("de", "Female"): "de-DE-KatjaNeural",
    ("de", "Male"): "de-DE-ConradNeural",
    ("ru", "Female"): "ru-RU-SvetlanaNeural",
    ("ru", "Male"): "ru-RU-DmitryNeural",
    ("ar", "Female"): "ar-SA-ZariyahNeural",
    ("ar", "Male"): "ar-SA-HamedNeural",
    ("hi", "Female"): "hi-IN-SwaraNeural",
    ("hi", "Male"): "hi-IN-MadhurNeural",
}


async def list_edge_voices(force_refresh: bool = False) -> List[Dict[str, Any]]:
    global _EDGE_VOICES_CACHE
    if _EDGE_VOICES_CACHE is not None and not force_refresh:
        return _EDGE_VOICES_CACHE
    try:
        raw = await edge_tts.list_voices()
        voices = []
        for v in raw:
            locale = v.get("Locale", "")
            lang_code = locale.split("-")[0].lower() if locale else "en"
            short = v.get("ShortName", "")
            gender = v.get("Gender", "Female")
            is_multi = "Multilingual" in short
            is_ph = "-PH-" in short
            style_label = (
                "⭐ Nâng cấp Đa ngôn ngữ (Clipchamp / Multilingual)"
                if is_multi
                else "Bản địa Philippines"
                if is_ph
                else "Neural Free"
            )
            voices.append(
                {
                    "id": short,
                    "name": f"{'⭐ ' if is_multi else ''}{short} ({gender})",
                    "lang": lang_code,
                    "locale": locale,
                    "gender": gender,
                    "style": style_label,
                    "multilingual": is_multi,
                    "region_ph": is_ph,
                    "free": True,
                    "provider": "edge_tts",
                }
            )
        _EDGE_VOICES_CACHE = voices
        return voices
    except Exception:
        return [
            {
                "id": vid,
                "name": f"{vid} ({g})",
                "lang": l,
                "gender": g,
                "style": "Neural Free",
                "multilingual": "Multilingual" in vid,
                "free": True,
                "provider": "edge_tts",
            }
            for (l, g), vid in _EDGE_FALLBACK_MAP.items()
        ]


async def list_all_voices(
    provider: str = "all",
    language: str = "",
    include_multilingual: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if provider in ("all", "capcut_tiktok"):
        out.extend(CAPCUT_TIKTOK_FREE_VOICES)
    if provider in ("all", "edge_tts"):
        out.extend(await list_edge_voices())
    if provider in ("all", "gtts"):
        out.extend(GTTS_FREE_VOICES)

    if not language:
        return out

    lang_l = language.lower().split("-")[0]
    match_codes = _LANG_ALIASES.get(lang_l, {lang_l})

    native: List[Dict[str, Any]] = []
    multilingual_extras: List[Dict[str, Any]] = []

    for v in out:
        v_lang = v["lang"].lower()
        is_ph_english = lang_l in ("fil", "tl") and v.get("region_ph", False)
        if v_lang in match_codes or is_ph_english:
            native.append(v)
        elif include_multilingual and v.get("multilingual", False):
            multilingual_extras.append(v)

    # Always put native voices first, followed by upgraded MultilingualNeural voices (like Clipchamp!)
    return native + multilingual_extras


def _split_text_chunks(text: str, max_len: int = 210) -> List[str]:
    clean = re.sub(r"\s+", " ", text.strip())
    if len(clean) <= max_len:
        return [clean] if clean else []
    chunks: List[str] = []
    parts = re.split(r"(?<=[,.!?;:。！？])\s*", clean)
    cur = ""
    for p in parts:
        if not p:
            continue
        if len(cur) + len(p) + 1 <= max_len:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_len:
                cur = p
            else:
                words = p.split(" ")
                cur = ""
                for w in words:
                    if len(cur) + len(w) + 1 <= max_len:
                        cur = f"{cur} {w}".strip()
                    else:
                        if cur:
                            chunks.append(cur)
                        cur = w
    if cur:
        chunks.append(cur)
    return chunks


async def _call_tiktok_tts_chunk(
    client: httpx.AsyncClient,
    text: str,
    voice_id: str,
    session_id: str = "",
) -> Optional[bytes]:
    if session_id.strip():
        for host in (
            "https://api16-normal-c-useast1a.tiktokv.com/media/api/text/speech/invoke/",
            "https://api16-normal-useast5.tiktokv.com/media/api/text/speech/invoke/",
        ):
            try:
                r = await client.post(
                    host,
                    params={
                        "text_speaker": voice_id,
                        "req_text": text,
                        "speaker_map_type": "0",
                        "aid": "1233",
                    },
                    headers={
                        "User-Agent": "com.zhiliaoapp.musically/2022600030 (Linux; U; Android 12; en_US; Pixel 6; Build/SP2A.220505.002)",
                        "Cookie": f"sessionid={session_id.strip()}",
                    },
                    timeout=12.0,
                )
                if r.status_code == 200:
                    data = r.json()
                    v_str = (data.get("data") or {}).get("v_str")
                    if v_str:
                        return base64.b64decode(v_str)
            except Exception:
                continue

    relays = [
        ("https://ottsy.weilbyte.dev/api/generation", {"text": text, "voice": voice_id}),
        ("https://tiktok-tts.weilnet.workers.dev/api/generation", {"text": text, "voice": voice_id}),
    ]
    for url, payload in relays:
        for attempt in range(2):
            try:
                r = await client.post(url, json=payload, timeout=12.0, follow_redirects=True)
                if r.status_code == 200:
                    data = r.json()
                    b64 = data.get("data") or data.get("base64") or data.get("audio")
                    if isinstance(b64, str) and len(b64) > 80:
                        if "," in b64[:40]:
                            b64 = b64.split(",", 1)[1]
                        return base64.b64decode(b64)
            except Exception:
                await asyncio.sleep(0.35 * (attempt + 1))
                continue
    return None


def _resolve_edge_fallback_voice(cfg: VoiceConfig) -> str:
    if "Neural" in cfg.voice_id:
        return cfg.voice_id
    meta = next((v for v in CAPCUT_TIKTOK_FREE_VOICES if v["id"] == cfg.voice_id), None)
    lang = (cfg.language or (meta["lang"] if meta else "vi")).lower().split("-")[0]
    gender = meta["gender"] if meta else "Female"
    return (
        _EDGE_FALLBACK_MAP.get((lang, gender))
        or _EDGE_FALLBACK_MAP.get((lang, "Female"))
        or "en-US-AvaMultilingualNeural"
    )


async def _call_edge_tts(text: str, voice_id: str, speed: float = 1.0) -> bytes:
    rate_pct = int(round((speed - 1.0) * 100))
    rate_str = f"{rate_pct:+d}%"
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            comm = edge_tts.Communicate(text=text, voice=voice_id, rate=rate_str)
            buf = bytearray()
            async for chunk in comm.stream():
                if chunk.get("type") == "audio":
                    buf.extend(chunk["data"])
            if buf:
                return bytes(buf)
        except Exception as e:
            last_err = e
        await asyncio.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Edge-TTS không trả về audio cho giọng {voice_id}: {last_err}")


async def _call_gtts(text: str, voice_id: str) -> bytes:
    """Free Google TTS synthesizer (supports Tagalog/Filipino 'tl' and 50+ languages)."""
    from gtts import gTTS  # type: ignore

    code = voice_id.replace("gtts:", "").strip() or "tl"
    tld = "com"
    if code == "en-ph":
        code = "en"
        tld = "com.ph"
    elif code == "fil":
        code = "tl"

    def _run() -> bytes:
        buf = io.BytesIO()
        gTTS(text=text, lang=code, tld=tld, slow=False).write_to_fp(buf)
        return buf.getvalue()

    return await asyncio.to_thread(_run)


async def synthesize_line(
    text: str,
    cfg: VoiceConfig,
    client: Optional[httpx.AsyncClient] = None,
) -> Tuple[bytes, str]:
    clean_text = text.strip()
    if not clean_text:
        return b"", "empty"

    if cfg.provider == "gtts" or cfg.voice_id.startswith("gtts:"):
        audio = await _call_gtts(clean_text, cfg.voice_id or f"gtts:{cfg.language or 'tl'}")
        return audio, f"gtts:{cfg.voice_id}"

    if cfg.provider == "capcut_tiktok":
        chunks = _split_text_chunks(clean_text, max_len=210)
        close_client = client is None
        http_client = client or httpx.AsyncClient(follow_redirects=True)
        try:
            parts: List[bytes] = []
            all_ok = True
            for ch in chunks:
                audio = await _call_tiktok_tts_chunk(
                    http_client, ch, cfg.voice_id, cfg.tiktok_session_id
                )
                if audio:
                    parts.append(audio)
                else:
                    all_ok = False
                    break
            if all_ok and parts:
                return b"".join(parts), f"capcut_tiktok:{cfg.voice_id}"
        finally:
            if close_client:
                await http_client.aclose()

        if not cfg.fallback_edge_tts:
            raise RuntimeError(
                f"Không gọi được endpoint CapCut/TikTok cho giọng '{cfg.voice_id}'. "
                "Hãy nhập TikTok sessionid hoặc bật tự động dự phòng Edge-TTS."
            )

    # _resolve_edge_fallback_voice keeps a real Edge voice ("...Neural") and maps anything else (a CapCut id left
    # over after switching the voice source to Edge-TTS) to a matching Edge voice. Sending "BV074_streaming" to
    # Edge made "Nghe thử giọng" fail with HTTP 500.
    edge_vid = _resolve_edge_fallback_voice(cfg)
    audio_bytes = await _call_edge_tts(clean_text, edge_vid, cfg.speed)
    return audio_bytes, f"edge_tts:{edge_vid}"


def _cache_key(text: str, cfg: VoiceConfig) -> str:
    raw = f"{cfg.provider}|{cfg.voice_id}|{cfg.speed:.2f}|{text.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


# Max parallel requests per voice source. Edge-TTS (Microsoft) handles hundreds of parallel requests, so the
# user's number is used as is. The free CapCut/TikTok relays and Google gTTS rate-limit / block quickly.
_MAX_WORKERS = {"edge_tts": 500, "capcut_tiktok": 32, "gtts": 16}


def effective_tts_workers(cfg: VoiceConfig, requested: Optional[int] = None) -> int:
    n = int(requested or cfg.concurrency or 12)
    return max(1, min(_MAX_WORKERS.get(cfg.provider, 32), n))


async def generate_project_voices(
    project: VideoProject,
    concurrency: Optional[int] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Voice for every line. Higgs TTS 3 has its own flow (GPU server + Whisper review, higgs_voice.py).
    Other voices: parallel requests, then (auto_fit) the lines whose real audio is longer than their time on
    screen are rewritten shorter by the LLM and voiced again (only those lines: the rest comes from the cache)."""
    if project.config.voice.provider == "higgs3":
        from app.services.higgs_voice import generate_project_voices_higgs

        return await generate_project_voices_higgs(project, on_progress)
    path = await _generate_project_voices_core(project, concurrency, on_progress)
    cfg = project.config.voice
    if not (cfg.auto_fit and project.translated_srt_path and Path(project.translated_srt_path).exists()):
        return path
    from app.services.length_fit import fit_project_lines, measure_voice_lengths

    items = json.loads(path.read_text(encoding="utf-8")).get("items", [])
    measured = await asyncio.to_thread(measure_voice_lengths, items)
    fit = await fit_project_lines(
        project, measured=measured,
        on_progress=(lambda p, m: on_progress(97.0, f"[Độ dài] {m}")) if on_progress else None,
    )
    if not fit["changed"]:
        return path
    if on_progress:
        on_progress(97.5, f"Tạo lại voice cho {len(fit['changed'])} câu đã rút gọn...")
    return await _generate_project_voices_core(project, concurrency, on_progress)


async def _generate_project_voices_core(
    project: VideoProject,
    concurrency: Optional[int] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """High-throughput parallel TTS synthesis (Edge-TTS: up to 500 parallel requests), live speed/ETA
    telemetry, persistent disk cache. A failing line no longer aborts the whole job: failed lines are retried
    at the end with fewer parallel requests, and only lines that still fail are reported."""
    srt_file = project.translated_srt_path or project.srt_path
    if not srt_file or not Path(srt_file).exists():
        raise RuntimeError("Chưa có file SRT (gốc hoặc đã dịch) để tạo voice.")

    cues: List[SubtitleCue] = parse_srt(srt_file)
    if not cues:
        raise RuntimeError("File SRT không có dòng phụ đề nào.")

    cfg = project.config.voice
    workers = effective_tts_workers(cfg, concurrency)
    voice_dir = PROJECTS_DIR / project.id / "voice_cache"
    voice_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(workers)
    done_count = 0
    total = len(cues)
    results: Dict[int, Dict[str, Any]] = {}
    t_start = time.perf_counter()

    if on_progress:
        on_progress(
            2.0,
            f"Đang tạo voice song song ({workers} luồng) cho {total} câu ({cfg.voice_name or cfg.voice_id})...",
        )

    failed: Dict[int, str] = {}
    limits = httpx.Limits(max_connections=workers + 4, max_keepalive_connections=workers)
    async with httpx.AsyncClient(follow_redirects=True, limits=limits) as http_client:
        async def _worker(cue: SubtitleCue, gate: asyncio.Semaphore, count: bool = True):
            nonlocal done_count
            key = _cache_key(cue.text, cfg)
            out_mp3 = voice_dir / f"{key}.mp3"
            provider_used = "cache"
            if not out_mp3.exists() or out_mp3.stat().st_size < 200:
                async with gate:
                    try:
                        audio_bytes, provider_used = await synthesize_line(cue.text, cfg, client=http_client)
                        if audio_bytes:
                            tmp = out_mp3.with_suffix(".part")
                            tmp.write_bytes(audio_bytes)
                            tmp.replace(out_mp3)  # never leave a half-written mp3 in the cache
                        failed.pop(cue.index, None)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001  (one bad line must not abort 10 000 others)
                        failed[cue.index] = str(e)[:160]
                        provider_used = "failed"
            results[cue.index] = {
                "index": cue.index,
                "start_sec": cue.start_sec,
                "end_sec": cue.end_sec,
                "text": cue.text,
                "audio_path": str(out_mp3) if out_mp3.exists() else "",
                "provider_used": provider_used,
            }
            if not count:
                return
            done_count += 1
            if on_progress and (done_count % 3 == 0 or done_count == total):
                elapsed = max(0.1, time.perf_counter() - t_start)
                rate = done_count / elapsed
                eta = (total - done_count) / max(0.1, rate)
                pct = round(5.0 + (done_count / total) * 90.0, 1)
                err = f" • {len(failed)} câu lỗi (sẽ thử lại)" if failed else ""
                on_progress(
                    pct,
                    f"Tạo voice ({workers} luồng): {done_count}/{total} câu ({rate:.1f} câu/s • Còn ~{eta:.0f}s){err}",
                )

        await asyncio.gather(*[_worker(c, sem) for c in cues])

        # Retry the failed lines (usually rate limiting under many threads) with fewer parallel requests.
        for attempt in range(2):
            if not failed:
                break
            retry_cues = [c for c in cues if c.index in failed]
            if on_progress:
                on_progress(96.0, f"Thử lại {len(retry_cues)} câu lỗi (lần {attempt + 1}/2, ít luồng hơn)...")
            await asyncio.sleep(2.0 * (attempt + 1))
            gate = asyncio.Semaphore(max(1, min(8, workers // 4 or 1)))
            await asyncio.gather(*[_worker(c, gate, count=False) for c in retry_cues])

    manifest = {
        "project_id": project.id,
        "srt_file": srt_file,
        "voice_config": cfg.model_dump(),
        "workers_used": workers,
        "failed": {str(k): v for k, v in failed.items()},
        "items": [results[c.index] for c in cues if c.index in results],
    }
    manifest_path = voice_dir / "voice_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_lines = sum(1 for it in manifest["items"] if it.get("audio_path"))
    if failed and ok_lines == 0:
        sample = next(iter(failed.values()))
        raise RuntimeError(f"Không tạo được câu voice nào ({len(failed)} câu lỗi). Lỗi đầu tiên: {sample}")
    if on_progress:
        total_sec = time.perf_counter() - t_start
        warn = f" ⚠ {len(failed)} câu vẫn lỗi sau 2 lần thử lại (sẽ im lặng ở đoạn đó)" if failed else ""
        on_progress(100.0, f"Xong {ok_lines}/{total} câu voice trong {total_sec:.1f}s ({workers} luồng)!{warn}")
    return manifest_path
