import subprocess
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.config import TEMP_DIR, get_ffmpeg_path
from app.main import app
from app.models import BoxRect, EditConfig, VideoProject
from app.services.hardware import detect_hardware
from app.services.media import probe_video, scan_system_fonts
from app.services.render_engine import render_project_sync


def create_test_assets(asset_dir: Path):
    asset_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_path()

    # 1. Synthetic 4s 1280x720 test movie with audio
    movie = asset_dir / "test_movie.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(movie),
        ],
        check=True,
    )

    # 2. Synthetic Vietnamese SRT
    srt = asset_dir / "test_movie.srt"
    srt.write_text(
        "1\n00:00:00,200 --> 00:00:01,900\nXin chào! Đây là phụ đề tiếng Việt bo góc siêu nét.\n\n"
        "2\n00:00:02,100 --> 00:00:03,800\nKhung phụ đề kéo rộng và cao độc lập không bị méo!\n",
        encoding="utf-8",
    )

    # 3. Synthetic Logo PNG
    logo = asset_dir / "test_logo.png"
    img = Image.new("RGBA", (220, 120), (20, 90, 220, 230))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([4, 4, 215, 115], radius=18, fill=(20, 90, 220, 240), outline=(255, 255, 255, 255), width=4)
    draw.text((40, 48), "REVIEW TV", fill=(255, 255, 255, 255))
    img.save(logo)

    # 4. Synthetic 3s Green-screen CTA video (#00FF00 bg + red Subscribe button)
    gs = asset_dir / "test_greenscreen.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=0x00FF00:s=480x270:r=30:d=3",
            "-vf", "drawbox=x=90:y=85:w=300:h=100:color=red@1.0:t=fill",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(gs),
        ],
        check=True,
    )
    return movie, srt, logo, gs


def main():
    print("=== 1. HARDWARE PROBE ===")
    hw = detect_hardware(force_refresh=True)
    for k in ("cpu_cores", "cpu_threads", "ram_gb", "gpus", "encoders", "recommended_encoder", "ai_device", "dual_gpu_capable", "tier", "summary"):
        print(f"  {k}: {hw[k]}")
    assert hw["ffmpeg_available"], "FFmpeg not found!"

    print("\n=== 2. FONTS SCAN ===")
    fonts = scan_system_fonts()
    print(f"  Found {len(fonts)} system fonts. Top 3: {[f['name'] for f in fonts[:3]]}")
    assert len(fonts) > 0

    print("\n=== 3. FASTAPI ENDPOINTS CHECK ===")
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/hardware").status_code == 200
    assert len(client.get("/api/config/languages").json()["languages"]) >= 30
    assert len(client.get("/api/presets").json()["presets"]) >= 1
    assert "dịch giả phụ đề" in client.get("/api/config/gem").json()["content"].lower()
    print("  All FastAPI endpoints returned 200 OK!")

    print("\n=== 4. FULL SINGLE-PASS RENDER TEST (BLUR + ROUNDED SUB + LOGO + GREEN SCREEN + AUDIO) ===")
    asset_dir = TEMP_DIR / "phase1_test"
    movie, srt, logo, gs = create_test_assets(asset_dir)
    meta = probe_video(str(movie))
    print(f"  Probed synthetic movie: {meta}")

    cfg = EditConfig()
    cfg.subtitle.enabled = True
    cfg.subtitle.box = BoxRect(x=0.10, y=0.72, w=0.80, h=0.18)
    cfg.subtitle.border_radius = 22
    cfg.subtitle.bg_opacity = 0.78
    cfg.logo.enabled = True
    cfg.logo.file_path = str(logo)
    cfg.logo.box = BoxRect(x=0.80, y=0.05, w=0.15, h=0.10)
    cfg.green_screen.enabled = True
    cfg.green_screen.file_path = str(gs)
    cfg.green_screen.box = BoxRect(x=0.05, y=0.10, w=0.28, h=0.20)
    cfg.green_screen.start_sec = 0.5
    cfg.sub_removal.mode = "blur"
    cfg.sub_removal.box = BoxRect(x=0.15, y=0.84, w=0.70, h=0.12)
    cfg.sub_removal.blur_strength = 20
    cfg.audio.original_volume = 0.35

    proj = VideoProject(
        id="test_p1",
        name="test_movie.mp4",
        video_path=str(movie),
        srt_path=str(srt),
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        duration=meta["duration"],
        has_audio=meta["has_audio"],
        config=cfg,
    )

    t0 = time.perf_counter()
    logs = []
    out_file = render_project_sync(proj, on_progress=lambda p, m: logs.append((p, m)))
    elapsed = time.perf_counter() - t0
    out_meta = probe_video(str(out_file))
    print(f"  Rendered in {elapsed:.2f}s -> {out_file}")
    print(f"  Output metadata: {out_meta}")
    print(f"  Last status: {logs[-1] if logs else None}")
    assert out_file.exists() and out_file.stat().st_size > 10_000
    assert out_meta["width"] == 1280 and out_meta["height"] == 720
    assert out_meta["has_audio"] is True

    # Cleanup synthetic test output
    out_file.unlink(missing_ok=True)
    print("\nALL PHASE 1 TESTS PASSED!")


if __name__ == "__main__":
    main()
