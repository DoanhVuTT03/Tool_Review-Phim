from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Tuple
import os
from PIL import Image, ImageDraw, ImageFont

from app.models import SubtitleConfig, SubtitleCue
from app.services.effects import (
    channel_loop_frames,
    channel_margins,
    subtitle_frame_counts,
    subtitle_intro_frames,
    subtitle_outro_frames,
)


def _hex_to_rgba(hex_color: str, alpha_float: float = 1.0) -> Tuple[int, int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (255, 255, 255, int(round(alpha_float * 255)))
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    a = max(0, min(255, int(round(alpha_float * 255))))
    return (r, g, b, a)


def _load_font(cfg: SubtitleConfig, size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if cfg.font_path and Path(cfg.font_path).exists():
        candidates.append(cfg.font_path)
    win_fonts = Path(r"C:\Windows\Fonts")
    if cfg.bold:
        candidates.extend(
            [
                str(win_fonts / "arialbd.ttf"),
                str(win_fonts / "segoeuib.ttf"),
                str(win_fonts / "tahomabd.ttf"),
            ]
        )
    candidates.extend(
        [
            str(win_fonts / "arial.ttf"),
            str(win_fonts / "segoeui.ttf"),
            str(win_fonts / "tahoma.ttf"),
        ]
    )
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=max(10, size_px))
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    stroke_width: int,
) -> List[str]:
    wrapped: List[str] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        words = raw_line.split(" ")
        # Handle CJK/no-space strings gracefully by character fallback
        if len(words) == 1 and draw.textlength(raw_line, font=font) > max_width:
            cur = ""
            for ch in raw_line:
                test = cur + ch
                if cur and draw.textlength(test, font=font) + stroke_width * 2 > max_width:
                    wrapped.append(cur)
                    cur = ch
                else:
                    cur = test
            if cur:
                wrapped.append(cur)
            continue

        cur = ""
        for w in words:
            test = f"{cur} {w}".strip()
            if cur and draw.textlength(test, font=font) + stroke_width * 2 > max_width:
                wrapped.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            wrapped.append(cur)
    return wrapped or [""]


def render_single_subtitle_png(
    text: str,
    cfg: Any,
    box_w: int,
    box_h: int,
    out_path: Path,
    scale_factor: float = 1.0,
) -> None:
    render_subtitle_image(text, cfg, box_w, box_h, scale_factor).save(out_path, format="PNG", compress_level=1)


def render_subtitle_image(
    text: str,
    cfg: Any,
    box_w: int,
    box_h: int,
    scale_factor: float = 1.0,
) -> Image.Image:
    """Renders 1 subtitle cue (or the channel name: same style fields) onto an exact (box_w x box_h)
    transparent RGBA canvas. Supports true rounded-corner background box, padding, stroke, alignment, and
    auto-fit if text overflows height.
    """
    box_w = max(16, int(box_w))
    box_h = max(16, int(box_h))
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad_x = max(2, int(round(cfg.padding_x * scale_factor)))
    pad_y = max(2, int(round(cfg.padding_y * scale_factor)))
    stroke_w = max(0, int(round(cfg.outline_width * scale_factor)))
    radius = max(0, int(round(cfg.border_radius * scale_factor)))
    avail_w = max(20, box_w - pad_x * 2)
    avail_h = max(16, box_h - pad_y * 2)

    font_size = max(10, int(round(cfg.font_size * scale_factor)))
    font = _load_font(cfg, font_size)
    lines = _wrap_text(draw, text, font, avail_w, stroke_w)

    # Measure total height and gently scale down font if overflowing box height
    def measure(fnt, lns):
        heights = []
        widths = []
        for ln in lns:
            bb = draw.textbbox((0, 0), ln, font=fnt, stroke_width=stroke_w)
            widths.append(bb[2] - bb[0])
            heights.append(max(fnt.size if hasattr(fnt, "size") else 16, bb[3] - bb[1]))
        line_step = [int(h * cfg.line_spacing) for h in heights]
        total_h = sum(line_step[:-1]) + (heights[-1] if heights else 0)
        return widths, heights, line_step, total_h

    widths, heights, line_step, total_h = measure(font, lines)
    while total_h > avail_h and font_size > 12:
        font_size = max(12, int(font_size * 0.9))
        font = _load_font(cfg, font_size)
        lines = _wrap_text(draw, text, font, avail_w, stroke_w)
        widths, heights, line_step, total_h = measure(font, lines)

    # Draw full-box rounded rectangle background (matches Preview box 1:1)
    if cfg.bg_enabled and cfg.bg_opacity > 0:
        bg_rgba = _hex_to_rgba(cfg.bg_color, cfg.bg_opacity)
        max_r = min(radius, box_w // 2, box_h // 2)
        draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=max_r, fill=bg_rgba)

    # Vertical positioning inside box
    if cfg.valign == "top":
        cur_y = pad_y
    elif cfg.valign == "bottom":
        cur_y = max(pad_y, box_h - pad_y - total_h)
    else:
        cur_y = max(pad_y, (box_h - total_h) // 2)

    text_rgba = _hex_to_rgba(cfg.text_color, 1.0)
    outline_rgba = _hex_to_rgba(cfg.outline_color, 1.0)

    for ln, lw, step in zip(lines, widths, line_step):
        if cfg.align == "left":
            x = pad_x
        elif cfg.align == "right":
            x = max(pad_x, box_w - pad_x - lw)
        else:
            x = max(pad_x, (box_w - lw) // 2)
        draw.text(
            (x, cur_y),
            ln,
            font=font,
            fill=text_rgba,
            stroke_width=stroke_w,
            stroke_fill=outline_rgba if stroke_w > 0 else None,
        )
        cur_y += step

    return img


def build_subtitle_concat_sequence(
    cues: List[SubtitleCue],
    cfg: SubtitleConfig,
    video_w: int,
    video_h: int,
    video_duration: float,
    work_dir: Path,
    fps: float = 30.0,
) -> Tuple[Path, int, int, int, int]:
    """Generates RGBA PNGs for all cues in parallel and builds an FFmpeg concat manifest.
    Returns: (concat_txt_path, box_x, box_y, box_w, box_h)
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    box_x = max(0, int(round(cfg.box.x * video_w)))
    box_y = max(0, int(round(cfg.box.y * video_h)))
    box_w = max(16, min(video_w - box_x, int(round(cfg.box.w * video_w))))
    box_h = max(16, min(video_h - box_y, int(round(cfg.box.h * video_h))))
    # Keep even dimensions for clean FFmpeg overlay alignment
    box_w = max(16, box_w - (box_w % 2))
    box_h = max(16, box_h - (box_h % 2))

    # Reference scale relative to 1080p height so font_size matches preview consistently
    scale_factor = video_h / 1080.0

    blank_path = work_dir / "blank.png"
    Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0)).save(blank_path, format="PNG", compress_level=1)

    valid_cues = [
        c for c in sorted(cues, key=lambda x: x.start_sec)
        if c.end_sec > c.start_sec and c.start_sec < video_duration
    ]

    effect = getattr(cfg, "effect", "none") or "none"
    effect_sec = float(getattr(cfg, "effect_duration", 0.3) or 0.3)

    # Parallel PNG generation (+ the animation frames of each cue when an effect is on)
    def _render_task(item: Tuple[int, SubtitleCue]) -> Tuple[int, Path, List[Path], List[Path], float]:
        idx, cue = item
        out_p = work_dir / f"cue_{idx:05d}.png"
        img = render_subtitle_image(cue.text, cfg, box_w, box_h, scale_factor)
        img.save(out_p, format="PNG", compress_level=1)
        n_in, n_out, step = subtitle_frame_counts(effect, effect_sec, max(0.0, cue.end_sec - cue.start_sec), fps)
        intro: List[Path] = []
        outro: List[Path] = []
        for k, fr in enumerate(subtitle_intro_frames(img, effect, n_in)):
            fp = work_dir / f"cue_{idx:05d}_i{k:02d}.png"
            fr.save(fp, format="PNG", compress_level=1)
            intro.append(fp)
        for k, fr in enumerate(subtitle_outro_frames(img, effect, n_out)):
            fp = work_dir / f"cue_{idx:05d}_o{k:02d}.png"
            fr.save(fp, format="PNG", compress_level=1)
            outro.append(fp)
        return idx, out_p, intro, outro, step

    cue_paths: dict[int, Path] = {}
    cue_fx: dict[int, Tuple[List[Path], List[Path], float]] = {}
    with ThreadPoolExecutor(max_workers=max(6, min(16, os.cpu_count() or 6))) as pool:
        for idx, p, intro, outro, step in pool.map(_render_task, enumerate(valid_cues)):
            cue_paths[idx] = p
            cue_fx[idx] = (intro, outro, step)

    def _fmt_file(p: Path) -> str:
        # FFmpeg concat demuxer safe forward-slash path escaping
        s = str(p.resolve()).replace("\\", "/").replace("'", "'\\''")
        return f"file '{s}'"

    manifest_lines: List[str] = ["ffconcat version 1.0"]
    cur_t = 0.0
    for idx, cue in enumerate(valid_cues):
        start = max(cur_t, cue.start_sec)
        end = min(video_duration, cue.end_sec)
        if end <= start + 0.01:
            continue
        if start > cur_t + 0.005:
            manifest_lines.append(_fmt_file(blank_path))
            manifest_lines.append(f"duration {start - cur_t:.4f}")
        intro, outro, step = cue_fx.get(idx, ([], [], 0.0))
        span = end - start
        # never let the animation frames overrun a (clipped / shortened) cue: drop outro first, then intro
        while outro and (len(intro) + len(outro)) * step > span * 0.9:
            outro = outro[:-1]
        while intro and len(intro) * step > span * 0.9:
            intro = intro[1:]
        for fp in intro:
            manifest_lines.append(_fmt_file(fp))
            manifest_lines.append(f"duration {step:.4f}")
        hold = span - (len(intro) + len(outro)) * step
        manifest_lines.append(_fmt_file(cue_paths[idx]))
        manifest_lines.append(f"duration {hold:.4f}")
        for fp in outro:
            manifest_lines.append(_fmt_file(fp))
            manifest_lines.append(f"duration {step:.4f}")
        cur_t = end

    if video_duration > cur_t + 0.01:
        manifest_lines.append(_fmt_file(blank_path))
        manifest_lines.append(f"duration {video_duration - cur_t:.4f}")
    # Repeat final frame per FFmpeg concat demuxer spec
    manifest_lines.append(_fmt_file(blank_path))

    concat_path = work_dir / "subs_concat.txt"
    concat_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    return concat_path, box_x, box_y, box_w, box_h


def build_channel_text_sequence(
    cfg: Any,
    video_w: int,
    video_h: int,
    fps: float,
    work_dir: Path,
) -> Tuple[List[str], int, int]:
    """Channel name overlay input. Returns (ffmpeg input args, overlay x, overlay y).
    Static -> one looped PNG; effect -> one period of animation frames in a concat file looped forever
    (-stream_loop -1). Both are endless inputs: overlay them with shortest=1."""
    work_dir.mkdir(parents=True, exist_ok=True)
    bx = max(0, int(round(cfg.box.x * video_w)))
    by = max(0, int(round(cfg.box.y * video_h)))
    bw = max(16, min(video_w - bx, int(round(cfg.box.w * video_w))))
    bh = max(16, min(video_h - by, int(round(cfg.box.h * video_h))))
    bw, bh = bw - bw % 2, bh - bh % 2
    img = render_subtitle_image(cfg.text or " ", cfg, bw, bh, video_h / 1080.0)
    effect = getattr(cfg, "effect", "none") or "none"
    mx, my = channel_margins(effect, bw, bh)
    # keep the enlarged canvas inside the frame
    mx = min(mx, bx, max(0, video_w - bx - bw))
    my = min(my, by, max(0, video_h - by - bh))
    mx, my = mx - mx % 2, my - my % 2
    if mx or my:
        canvas = Image.new("RGBA", (bw + 2 * mx, bh + 2 * my), (0, 0, 0, 0))
        canvas.paste(img, (mx, my))
        img = canvas
    if effect == "none":
        png = work_dir / "channel.png"
        img.save(png, format="PNG", compress_level=1)
        return ["-loop", "1", "-framerate", "5", "-i", str(png)], bx - mx, by - my

    frames = channel_loop_frames(img, effect, fps, float(getattr(cfg, "effect_period_sec", 8.0) or 8.0))
    lines = ["ffconcat version 1.0"]
    for k, (fr, dur) in enumerate(frames):
        fp = work_dir / f"channel_{k:04d}.png"
        fr.save(fp, format="PNG", compress_level=1)
        lines.append("file '" + str(fp.resolve()).replace("\\", "/").replace("'", "'\\''") + "'")
        lines.append(f"duration {dur:.4f}")
    txt = work_dir / "channel_loop.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")
    return ["-stream_loop", "-1", "-f", "concat", "-safe", "0", "-i", str(txt)], bx - mx, by - my
