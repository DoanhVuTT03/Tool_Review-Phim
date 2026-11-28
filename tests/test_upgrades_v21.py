import asyncio
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.config import TEMP_DIR, get_aria2c_path
from app.main import app
from app.models import BackgroundMusicConfig, VoiceConfig
from app.services.audio_assembler import SAMPLE_RATE, build_crossfaded_bgm_pcm
from app.services.downloader import clean_srt_rules_sync, clean_srt_with_antigravity
from app.services.media import parse_srt
from app.services.tts_engine import synthesize_line


async def fake_cleaner_llm(cfg, system_prompt: str, user_prompt: str) -> str:
    assert "KHÔNG DỊCH" in system_prompt
    return "1|||DELETE\n2|||男主角推开了大门。\n3|||他发现屋里空无一人。"


async def main():
    client = TestClient(app)

    print("=== 1. ANTIGRAVITY LATEST MODELS CHECK ===")
    m_res = client.get("/api/config/models").json()
    ids = [m["id"] for m in m_res["models"]]
    print("  Models:", ids)
    # With the Antigravity app running the list is live and uses Antigravity's own ids (the old tool aliases
    # "orchestrator" / "claude-opus-5.5" were never valid there); without it, the static fallback list.
    aliases = {"orchestrator": ("orchestrator", "gemini-pro-orchestrator-neon"),
               "claude-opus-5.5": ("claude-opus-5.5", "claude-opus-5-5"),
               "claude-sonnet-5.5": ("claude-sonnet-5.5", "claude-sonnet-5-5")}
    for expected in ("gemini-4-neon", "gemini-4-hydrogen", "orchestrator", "claude-opus-5.5", "claude-sonnet-5.5"):
        assert any(a in ids for a in aliases.get(expected, (expected,))), f"Missing model {expected}"

    print("\n=== 2. FILIPINO VOICES + EDGE MULTILINGUAL + GTTS CHECK ===")
    edge_fil = client.get("/api/voices?provider=edge_tts&language=fil&include_multilingual=true").json()
    edge_fil_ids = [v["id"] for v in edge_fil["voices"]]
    print(f"  Edge-TTS Filipino + Multilingual ({edge_fil['count']}): {edge_fil_ids[:6]} ...")
    assert "fil-PH-BlessicaNeural" in edge_fil_ids
    assert "fil-PH-AngeloNeural" in edge_fil_ids
    assert "en-US-AvaMultilingualNeural" in edge_fil_ids
    assert "en-US-AndrewMultilingualNeural" in edge_fil_ids
    assert edge_fil["count"] == 14  # 4 PH native + 10 MultilingualNeural

    capcut_fil = client.get("/api/voices?provider=capcut_tiktok&language=fil").json()
    print(f"  CapCut/TikTok Filipino (should be 0, NOT dumping all languages): {capcut_fil['count']}")
    assert capcut_fil["count"] == 0

    gtts_fil = client.get("/api/voices?provider=gtts&language=fil").json()
    print(f"  Google TTS Free Filipino ({gtts_fil['count']}): {[v['id'] for v in gtts_fil['voices']]}")
    assert gtts_fil["count"] >= 2

    # Live TTS synthesis tests for Filipino (gTTS + Edge Multilingual)
    b_gtts, prov_g = await synthesize_line(
        "Magandang araw sa inyong lahat!",
        VoiceConfig(provider="gtts", language="fil", voice_id="gtts:tl"),
    )
    print(f"  Live gTTS Tagalog bytes: {len(b_gtts)} ({prov_g})")
    assert len(b_gtts) > 2000

    b_multi, prov_m = await synthesize_line(
        "Magandang araw sa inyong lahat!",
        VoiceConfig(provider="edge_tts", language="fil", voice_id="en-US-AvaMultilingualNeural"),
    )
    print(f"  Live Edge Multilingual Tagalog bytes: {len(b_multi)} ({prov_m})")
    assert len(b_multi) > 4000

    print("\n=== 3. BGM OVERLAPPING CROSSFADE PLAYLIST TEST ===")
    tmp = TEMP_DIR / "test_v21"
    tmp.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 4.0, int(4.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
    t1 = tmp / "bgm1.wav"
    t2 = tmp / "bgm2.wav"
    sf.write(str(t1), 0.3 * np.sin(2 * np.pi * 220 * t), SAMPLE_RATE)
    sf.write(str(t2), 0.3 * np.sin(2 * np.pi * 330 * t), SAMPLE_RATE)

    # Multi-track crossfade loop covering 11 seconds (Track1 -> xfade -> Track2 -> xfade -> Track1 -> ...)
    bgm_multi = build_crossfaded_bgm_pcm(
        BackgroundMusicConfig(enabled=True, file_paths=[str(t1), str(t2)], crossfade_sec=1.2, volume=0.3),
        total_samples=int(11.0 * SAMPLE_RATE),
    )
    assert bgm_multi.size == int(11.0 * SAMPLE_RATE)
    assert not np.isnan(bgm_multi).any()
    assert float(np.max(np.abs(bgm_multi))) > 0.05
    # Verify smooth fade-in at t=0 and fade-out at end
    assert abs(float(bgm_multi[0])) < 1e-4 and abs(float(bgm_multi[-1])) < 1e-4

    # Single-track crossfade loop covering 9 seconds
    bgm_single = build_crossfaded_bgm_pcm(
        BackgroundMusicConfig(enabled=True, file_paths=[str(t1)], crossfade_sec=1.0, volume=0.3),
        total_samples=int(9.0 * SAMPLE_RATE),
    )
    assert bgm_single.size == int(9.0 * SAMPLE_RATE)
    print(f"  BGM Crossfade multi-track ({bgm_multi.size} samples) & single-track loop OK!")

    print("\n=== 4. ARIA2C + 2-STAGE SRT GARBAGE CLEANER TEST ===")
    aria2 = get_aria2c_path()
    print(f"  aria2c binary: {aria2}")
    assert aria2 and Path(aria2).exists()

    dirty_srt = tmp / "dirty.srt"
    dirty_srt.write_text(
        "1\n00:00:00,100 --> 00:00:01,500\n[Music]\n\n"
        "2\n00:00:01,600 --> 00:00:03,000\n<i>欢迎订阅频道</i>\n\n"
        "3\n00:00:03,100 --> 00:00:04,500\n<font color=\"#fff\">广告赞助商链接</font>\n\n"
        "4\n00:00:04,600 --> 00:00:06,000\n男主角推开了大门。\n\n"
        "5\n00:00:06,100 --> 00:00:07,200\n男主角推开了大门。\n\n"
        "6\n00:00:07,500 --> 00:00:09,000\n{\\an8}他发现屋里空无一人。\n",
        encoding="utf-8",
    )
    _, r_stats = clean_srt_rules_sync(str(dirty_srt))
    print(f"  Stage 1 Rule Cleaner: {r_stats}")
    assert r_stats["before_count"] == 6 and r_stats["after_count"] == 3

    cleaned_p, ai_stats = await clean_srt_with_antigravity(
        str(dirty_srt), model="gemini-3.8-flash", thinking_level="low", llm_caller=fake_cleaner_llm
    )
    final_cues = parse_srt(str(cleaned_p))
    print(f"  Stage 2 AI Cleaner: {ai_stats} -> {[c.text for c in final_cues]}")
    assert len(final_cues) == 2 and final_cues[0].text == "男主角推开了大门。"

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL V2.1 UPGRADE TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
