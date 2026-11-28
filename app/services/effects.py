"""Motion effects for the subtitle box and the channel-name text.

Every effect is computed from the exact RGBA image the tool already renders (rounded box, font, outline...), so the
animated result looks exactly like the static one. The frames go into the same ffmpeg concat sequence as the
static images (short entries of 1/fps each), so they work unchanged with the CPU and the CUDA overlay graphs.

Subtitles: an entrance animation per cue (+ a short fade-out).
Channel name: a looping animation (continuous, or repeated every `period` seconds).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter

SUBTITLE_EFFECTS = {
    "none": "Không hiệu ứng",
    "fade": "Mờ dần vào / ra",
    "slide_up": "Trượt lên",
    "pop": "Bật nảy (Pop)",
    "zoom_in": "Phóng to vào",
    "typewriter": "Gõ chữ (lộ dần trái → phải)",
    "blur_in": "Nhòe → nét",
    "bounce": "Rơi nảy",
    "slide_left": "Trượt từ phải sang",
}

CHANNEL_EFFECTS = {
    "none": "Không hiệu ứng",
    "breathe": "Nhịp thở (phóng to / thu nhỏ)",
    "float": "Bồng bềnh (lên xuống)",
    "pulse": "Nhấp nháy mờ",
    "shimmer": "Ánh sáng lướt qua (định kỳ)",
    "slide_in": "Trượt vào (định kỳ)",
    "typewriter": "Gõ chữ (định kỳ)",
}

ANIM_FPS_MAX = 30.0


# ---------------------------------------------------------------------------------------------------------------
# easing
# ---------------------------------------------------------------------------------------------------------------


def _ease_out_cubic(p: float) -> float:
    return 1.0 - (1.0 - p) ** 3


def _ease_out_back(p: float, s: float = 1.9) -> float:
    q = p - 1.0
    return 1.0 + (s + 1.0) * q**3 + s * q**2


def _bounce_out(p: float) -> float:
    n1, d1 = 7.5625, 2.75
    if p < 1 / d1:
        return n1 * p * p
    if p < 2 / d1:
        p -= 1.5 / d1
        return n1 * p * p + 0.75
    if p < 2.5 / d1:
        p -= 2.25 / d1
        return n1 * p * p + 0.9375
    p -= 2.625 / d1
    return n1 * p * p + 0.984375


# ---------------------------------------------------------------------------------------------------------------
# one transformed frame
# ---------------------------------------------------------------------------------------------------------------


def transform(
    base: Image.Image,
    alpha: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    scale: float = 1.0,
    blur: float = 0.0,
    wipe: float | None = None,
    highlight: float | None = None,
) -> Image.Image:
    """base: RGBA canvas. Returns a same-size RGBA frame: scaled about the centre, shifted, blurred, faded,
    revealed left->right up to `wipe` (0..1), with an optional diagonal light band at `highlight` (0..1)."""
    w, h = base.size
    img = base
    if abs(scale - 1.0) > 1e-3:
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((sw, sh), Image.BILINEAR)
    if blur > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ox = (w - img.width) // 2 + int(round(dx))
    oy = (h - img.height) // 2 + int(round(dy))
    canvas.paste(img, (ox, oy))
    if alpha >= 0.999 and wipe is None and highlight is None:
        return canvas
    a = np.asarray(canvas).copy()
    if highlight is not None:
        # diagonal soft white band sweeping left -> right, only on visible pixels
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        pos = (xx + 0.6 * yy) / (w + 0.6 * h)
        band = np.exp(-(((pos - (highlight * 1.3 - 0.15)) / 0.07) ** 2)) * 0.65
        rgb = a[..., :3].astype(np.float32)
        a[..., :3] = np.clip(rgb + (255.0 - rgb) * band[..., None], 0, 255).astype(np.uint8)
    if wipe is not None:
        edge = max(2.0, 0.05 * w)
        xs = np.arange(w, dtype=np.float32)
        ramp = np.clip((wipe * (w + edge) - xs) / edge, 0.0, 1.0)
        a[..., 3] = (a[..., 3].astype(np.float32) * ramp[None, :]).astype(np.uint8)
    if alpha < 0.999:
        a[..., 3] = (a[..., 3].astype(np.float32) * max(0.0, alpha)).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


# ---------------------------------------------------------------------------------------------------------------
# subtitles: entrance (+ fade-out) per cue
# ---------------------------------------------------------------------------------------------------------------


def subtitle_intro_frames(base: Image.Image, effect: str, n: int) -> List[Image.Image]:
    """n frames BEFORE the static image (frame k shows progress k/n; the static image is progress 1)."""
    if effect == "none" or n <= 0:
        return []
    w, h = base.size
    out: List[Image.Image] = []
    for k in range(n):
        p = k / n
        e = _ease_out_cubic(p)
        if effect == "fade":
            f = transform(base, alpha=e)
        elif effect == "slide_up":
            f = transform(base, alpha=min(1.0, p * 1.8), dy=(1.0 - e) * 0.45 * h)
        elif effect == "pop":
            f = transform(base, alpha=min(1.0, p * 2.5), scale=0.55 + 0.45 * _ease_out_back(p))
        elif effect == "zoom_in":
            f = transform(base, alpha=e, scale=1.35 - 0.35 * e)
        elif effect == "typewriter":
            f = transform(base, wipe=p)
        elif effect == "blur_in":
            f = transform(base, alpha=min(1.0, 0.2 + e), blur=(1.0 - e) * 9.0 * max(0.6, h / 150.0))
        elif effect == "bounce":
            f = transform(base, alpha=min(1.0, p * 3.0), dy=-(1.0 - _bounce_out(p)) * 0.6 * h)
        elif effect == "slide_left":
            f = transform(base, alpha=e, dx=(1.0 - e) * 0.3 * w)
        else:
            return []
        out.append(f)
    return out


def subtitle_outro_frames(base: Image.Image, effect: str, n: int) -> List[Image.Image]:
    """n frames AFTER the static image: a quick fade-out (the same for every effect)."""
    if effect == "none" or n <= 0:
        return []
    return [transform(base, alpha=1.0 - _ease_out_cubic((k + 1) / (n + 1))) for k in range(n)]


def subtitle_frame_counts(effect: str, effect_sec: float, cue_sec: float, fps: float) -> Tuple[int, int, float]:
    """(intro frames, outro frames, frame duration) for one cue; short cues get shorter animations."""
    if effect == "none":
        return 0, 0, 0.0
    afps = min(ANIM_FPS_MAX, max(10.0, fps))
    step = 1.0 / afps
    intro_sec = min(effect_sec if effect != "typewriter" else max(effect_sec, 0.5), cue_sec * 0.45)
    outro_sec = min(0.15, cue_sec * 0.15)
    return max(0, int(intro_sec / step)), max(0, int(outro_sec / step)), step


# ---------------------------------------------------------------------------------------------------------------
# channel name: looping animation
# ---------------------------------------------------------------------------------------------------------------


def channel_margins(effect: str, w: int, h: int) -> Tuple[int, int]:
    """Transparent margin around the channel box so scaling / floating is not clipped at the box edge."""
    if effect in ("breathe", "float", "slide_in"):
        return int(round(w * 0.06)) // 2 * 2, int(round(h * 0.18)) // 2 * 2
    return 0, 0


def channel_loop_frames(base: Image.Image, effect: str, fps: float, period: float) -> List[Tuple[Image.Image, float]]:
    """One period of the loop as (frame, duration seconds). The caller loops it for the whole video."""
    if effect == "none":
        return [(base, max(1.0, period))]
    afps = min(ANIM_FPS_MAX, max(10.0, fps))
    step = 1.0 / afps
    w, h = base.size
    frames: List[Tuple[Image.Image, float]] = []

    def cyc(seconds: float):
        n = max(2, int(round(seconds * afps)))
        return n, [k / n for k in range(n)]

    if effect == "breathe":
        n, ps = cyc(2.2)
        frames = [(transform(base, scale=1.0 + 0.05 * (0.5 - 0.5 * math.cos(2 * math.pi * p))), step) for p in ps]
    elif effect == "float":
        n, ps = cyc(2.6)
        frames = [(transform(base, dy=math.sin(2 * math.pi * p) * 0.07 * h), step) for p in ps]
    elif effect == "pulse":
        n, ps = cyc(1.8)
        frames = [(transform(base, alpha=0.62 + 0.38 * (0.5 + 0.5 * math.cos(2 * math.pi * p))), step) for p in ps]
    elif effect == "shimmer":
        sweep = min(1.0, period * 0.5)
        n, ps = cyc(sweep)
        frames = [(transform(base, highlight=p), step) for p in ps]
        frames.append((base, max(step, period - sweep)))
    elif effect == "slide_in":
        t_in, t_out = 0.5, 0.4
        n_in, ps = cyc(t_in)
        frames = [(transform(base, alpha=_ease_out_cubic(p), dx=-(1.0 - _ease_out_cubic(p)) * 0.5 * w), step) for p in ps]
        frames.append((base, max(step, period - t_in - t_out)))
        n_out, ps = cyc(t_out)
        frames += [(transform(base, alpha=1.0 - _ease_out_cubic(p)), step) for p in ps]
    elif effect == "typewriter":
        t_in, t_out = 1.2, 0.4
        n_in, ps = cyc(t_in)
        frames = [(transform(base, wipe=p), step) for p in ps]
        frames.append((base, max(step, period - t_in - t_out)))
        n_out, ps = cyc(t_out)
        frames += [(transform(base, alpha=1.0 - _ease_out_cubic(p)), step) for p in ps]
    else:
        frames = [(base, max(1.0, period))]
    return frames
