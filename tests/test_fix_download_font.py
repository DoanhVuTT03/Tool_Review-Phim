import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import DownloadVideoRequest, SubtitleConfig
from app.services.downloader import _download_sync, format_download_error, strip_ansi
from app.services.media import parse_srt, scan_system_fonts
from app.services.subtitle_renderer import _load_font, render_single_subtitle_png


class TestFixDownloadAndFont(unittest.TestCase):
    def test_strip_ansi_and_format_error(self):
        raw_error = "\x1b[0;31mERROR:\x1b[0m aria2c exited with code 28"
        clean = strip_ansi(raw_error)
        self.assertNotIn("\x1b", clean)
        self.assertNotIn("[0;31m", clean)
        self.assertIn("aria2c exited with code 28", clean)

        unavailable_err = "ERROR: [youtube] TWPtJk2YEis: This video is unavailable"
        formatted = format_download_error(unavailable_err)
        self.assertIn("không khả dụng", formatted)

        bracket_err = "[0;31mERROR:[0m Private video"
        formatted_priv = format_download_error(bracket_err)
        self.assertIn("chế độ riêng tư", formatted_priv)

    def test_download_with_32_threads_aria2c(self):
        """Verify downloading a test video with threads=32 does NOT trigger aria2c code 28."""
        req = DownloadVideoRequest(
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
            custom_name="test_fix_32_threads",
            threads=32,
            sub_lang="auto",
            browser_cookies="none",
            clean_srt_rules=False,
            clean_srt_ai=False,
        )
        res = _download_sync(req)
        self.assertTrue(Path(res["video_path"]).exists())
        self.assertEqual(res["threads"], 32)
        print("Downloaded file successfully:", res["video_path"])

    def test_font_scanning_and_vietnamese_rendering(self):
        """Verify system fonts are detected and Vietnamese subtitle PNG renders cleanly."""
        fonts = scan_system_fonts()
        self.assertGreater(len(fonts), 5)

        # Test rendering subtitle PNG with Vietnamese accents
        cfg = SubtitleConfig(
            text_color="#FFFFFF",
            font_size=36,
            bold=True,
            outline_width=2,
            outline_color="#000000",
            bg_enabled=True,
            bg_color="#000000",
            bg_opacity=0.6,
            border_radius=8,
        )
        font = _load_font(cfg, 36)
        self.assertIsNotNone(font)

        out_png = Path("downloads") / "test_vi_sub.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        render_single_subtitle_png(
            text="Hôm nay chúng ta sẽ xem review phim cực kỳ hấp dẫn!",
            cfg=cfg,
            box_w=800,
            box_h=120,
            out_path=out_png,
        )
        self.assertTrue(out_png.exists())
        self.assertGreater(out_png.stat().st_size, 500)
        print("Rendered Vietnamese subtitle PNG size:", out_png.stat().st_size)


if __name__ == "__main__":
    unittest.main()
