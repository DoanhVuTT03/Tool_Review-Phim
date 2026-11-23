import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.config import PROJECTS_DIR, TEMP_DIR, get_ffmpeg_path
from app.main import app
from app.models import BoxRect, EditConfig, VideoProject
from app.services.audio_assembler import assemble_project_audio_sync
from app.services.media import parse_srt, probe_video
from app.services.render_engine import render_project_sync
from app.services.sub_remover import clean_hardsub_video_sync
from app.services.translator import (
    detect_translation_backends,
    load_translate_checkpoint,
    translate_project_srt,
)
from app.services.tts_engine import generate_project_voices, list_all_voices


def create_hardsub_movie(asset_dir: Path):
    asset_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_path()

    # Create a 4s 960x540 video with a burned-in hardsub text from t=1.0..2.8s so OCR can detect & clean it!
    movie = asset_dir / "hardsub_movie.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=0x1e3a5f:s=960x540:r=24:d=4",
            "-f", "lavfi", "-i", "sine=frequency=330:duration=4",
            "-vf", (
                "drawbox=x=0:y=0:w=960:h=270:color=0x2c5282@1.0:t=fill,"
                "drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':"
                "text='ORIGINAL HARDSUB TEXT':fontcolor=white:fontsize=38:"
                "x=(w-text_w)/2:y=h-75:enable='between(t,1.0,2.8)'"
            ),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(movie),
        ],
        check=True,
    )

    srt = asset_dir / "hardsub_movie.srt"
    srt.write_text(
        "1\n00:00:00,200 --> 00:00:01,800\n大家好，欢迎来到今天的电影解说。\n\n"
        "2\n00:00:02,000 --> 00:00:03,800\n男主角终于发现了背后的惊天秘密。\n",
        encoding="utf-8",
    )

    logo = asset_dir / "logo.png"
    img = Image.new("RGBA", (180, 90), (220, 40, 60, 235))
    draw = ImageDraw.Draw(img)
    draw.text((28, 34), "PHIM REVIEW", fill=(255, 255, 255, 255))
    img.save(logo)

    return movie, srt, logo


async def fake_llm_caller(cfg, system_prompt: str, user_prompt: str) -> str:
    """Deterministic test stub verifying gem.txt rules, 2-phase Style Guide, and <id>|||<text> parsing."""
    assert "dịch giả phụ đề" in system_prompt.lower()
    if "PHÂN TÍCH FILE ĐÍNH KÈM" in user_prompt:
        return (
            "1. CHỦ ĐỀ: Review phim hành động kịch tính\n"
            "2. NGỮ CẢNH: Mở đầu giới thiệu bí mật nhân vật chính\n"
            "3. NHÂN VẬT & XƯNG HÔ: Người kể chuyện (tôi/các bạn), Nam chính (anh ta)\n"
            "4. VĂN PHONG: Cuốn hút, súc tích chuẩn review phim YouTube"
        )
    assert "STYLE GUIDE ĐÃ CHỐT" in user_prompt
    assert "±" in user_prompt
    return (
        "1|||Xin chào mọi người, chào mừng đến với tập review phim hôm nay.\n"
        "2|||Cuối cùng nam chính cũng phát hiện ra bí mật động trời phía sau."
    )


async def run_all():
    test_id = "test_full_az"
    asset_dir = TEMP_DIR / test_id
    proj_work_dir = PROJECTS_DIR / test_id
    if proj_work_dir.exists():
        shutil.rmtree(proj_work_dir, ignore_errors=True)

    print("=== 1. API & BACKENDS CHECK ===")
    backends = detect_translation_backends()
    print(f"  Translation backends: {backends}")
    assert backends["antigravity_sdk"]["available"] is True, "google-antigravity SDK must be installed!"
    assert backends["gem_txt_exists"] is True

    client = TestClient(app)
    assert client.get("/api/translate/backends").status_code == 200
    v_res = client.get("/api/voices?provider=capcut_tiktok&language=vi").json()
    print(f"  CapCut/TikTok Free Vietnamese voices: {v_res['count']}")
    assert v_res["count"] >= 2
    all_edge = await list_all_voices(provider="edge_tts")
    print(f"  Total Edge-TTS free world voices: {len(all_edge)}")
    assert len(all_edge) >= 50

    movie, srt, logo = create_hardsub_movie(asset_dir)
    meta = probe_video(str(movie))

    cfg = EditConfig()
    cfg.subtitle.enabled = True
    cfg.subtitle.box = BoxRect(x=0.08, y=0.76, w=0.84, h=0.16)
    cfg.subtitle.border_radius = 18
    cfg.logo.enabled = True
    cfg.logo.file_path = str(logo)
    cfg.logo.box = BoxRect(x=0.80, y=0.04, w=0.16, h=0.11)
    cfg.sub_removal.mode = "ai_clean"
    cfg.sub_removal.box = BoxRect(x=0.10, y=0.80, w=0.80, h=0.18)
    cfg.sub_removal.auto_detect_ocr = True
    cfg.audio.edit_mode = "narration"
    cfg.audio.original_volume = 0.18
    cfg.audio.auto_ducking = True
    cfg.voice.provider = "capcut_tiktok"
    cfg.voice.language = "vi"
    cfg.voice.voice_id = "BV074_streaming"
    cfg.voice.fallback_edge_tts = True

    proj = VideoProject(
        id=test_id,
        name="hardsub_movie.mp4",
        video_path=str(movie),
        srt_path=str(srt),
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        duration=meta["duration"],
        has_audio=meta["has_audio"],
        config=cfg,
    )

    print("\n=== 2. PHASE 2: STYLE GUIDE + TRANSLATE + CHECKPOINT ===")
    out_srt = await translate_project_srt(proj, reset_checkpoint=True, llm_caller=fake_llm_caller)
    proj.translated_srt_path = str(out_srt)
    assert Path(proj.style_guide_path).exists(), "Style Guide file missing!"
    cues = parse_srt(str(out_srt))
    cp = load_translate_checkpoint(proj)
    print(f"  Style Guide saved: {proj.style_guide_path}")
    print(f"  Translated SRT ({len(cues)} cues): {[c.text for c in cues]}")
    print(f"  Checkpoint entries: {len(cp.get('translated', {}))}")
    assert len(cues) == 2 and len(cp["translated"]) == 2

    print("\n=== 3. PHASE 3: LIVE TTS VOICE GENERATION & DISK CACHE ===")
    t0 = time.perf_counter()
    manifest_path = await generate_project_voices(proj, concurrency=4)
    print(f"  Generated real TTS voices in {time.perf_counter() - t0:.2f}s -> {manifest_path}")
    assert manifest_path.exists()
    # Second call should hit cache near-instantly
    t1 = time.perf_counter()
    await generate_project_voices(proj, concurrency=4)
    cache_elapsed = time.perf_counter() - t1
    print(f"  Cached TTS rerun time: {cache_elapsed:.3f}s")
    assert cache_elapsed < 1.0

    print("\n=== 4A. PHASE 4: AUDIO ASSEMBLER (NARRATION & DUBBING) ===")
    nar_wav, nar_stats = assemble_project_audio_sync(proj, str(manifest_path))
    print(f"  Narration stats: {nar_stats}")
    assert nar_wav.exists() and nar_stats["cues_assembled"] == 2

    proj.config.audio.edit_mode = "dubbing"
    proj.config.audio.mute_original = True
    dub_wav, dub_stats = assemble_project_audio_sync(proj, str(manifest_path))
    print(f"  Dubbing stats: {dub_stats}")
    assert dub_wav.exists() and dub_stats["cues_assembled"] == 2
    proj.config.audio.edit_mode = "narration"
    proj.config.audio.mute_original = False

    print("\n=== 4B. PHASE 4: AI CLEAN SUB REMOVAL (OCR + ROI + TEMPORAL + QSV) ===")
    t2 = time.perf_counter()
    clean_vid, clean_stats = clean_hardsub_video_sync(proj)
    print(f"  AI Clean Sub finished in {time.perf_counter() - t2:.2f}s: {clean_stats}")
    assert clean_vid.exists()
    assert clean_stats["cleaned_frames"] > 0 and clean_stats["untouched_frames"] > 0

    print("\n=== 4C. PHASE 4: FINAL SINGLE-PASS RENDER (CLEAN VIDEO + ROUNDED SUB + LOGO + NARRATION WAV) ===")
    t3 = time.perf_counter()
    final_mp4 = render_project_sync(
        proj,
        voice_wav_path=str(nar_wav),
        source_video_override=str(clean_vid),
        voice_already_mixed=True,
    )
    out_meta = probe_video(str(final_mp4))
    print(f"  Final A-Z MP4 rendered in {time.perf_counter() - t3:.2f}s -> {final_mp4}")
    print(f"  Output metadata: {out_meta}")
    assert final_mp4.exists() and out_meta["width"] == 960 and out_meta["height"] == 540 and out_meta["has_audio"]

    # Cleanup synthetic test files
    final_mp4.unlink(missing_ok=True)
    shutil.rmtree(asset_dir, ignore_errors=True)
    shutil.rmtree(proj_work_dir, ignore_errors=True)
    print("\nALL FULL A-Z (PHASES 1-4) TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_all())
