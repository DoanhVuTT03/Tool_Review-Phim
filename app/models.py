from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class BoxRect(BaseModel):
    """Normalized rectangle in [0.0, 1.0] relative to video frame width/height.
    Allows independent width (w) and height (h) resizing without aspect-ratio lock.
    """
    x: float = Field(0.1, ge=0.0, le=1.0)
    y: float = Field(0.78, ge=0.0, le=1.0)
    w: float = Field(0.8, gt=0.01, le=1.0)
    h: float = Field(0.14, gt=0.01, le=1.0)


class SubtitleConfig(BaseModel):
    enabled: bool = True
    box: BoxRect = Field(default_factory=lambda: BoxRect(x=0.08, y=0.78, w=0.84, h=0.14))
    font_family: str = "Arial"
    font_path: str = ""
    font_size: int = Field(42, ge=10, le=200)
    bold: bool = True
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = Field(2, ge=0, le=12)
    bg_enabled: bool = True
    bg_color: str = "#000000"
    bg_opacity: float = Field(0.72, ge=0.0, le=1.0)
    border_radius: int = Field(16, ge=0, le=80)
    padding_x: int = Field(20, ge=0, le=80)
    padding_y: int = Field(12, ge=0, le=80)
    align: Literal["left", "center", "right"] = "center"
    valign: Literal["top", "middle", "bottom"] = "middle"
    line_spacing: float = Field(1.2, ge=0.8, le=2.5)
    sample_text: str = "Đây là dòng phụ đề xem trước (kéo khung để chỉnh dài/rộng)"


class LogoConfig(BaseModel):
    enabled: bool = False
    file_path: str = ""
    folder_path: str = ""
    box: BoxRect = Field(default_factory=lambda: BoxRect(x=0.82, y=0.04, w=0.14, h=0.12))
    opacity: float = Field(0.95, ge=0.0, le=1.0)


class GreenScreenConfig(BaseModel):
    enabled: bool = False
    file_path: str = ""
    box: BoxRect = Field(default_factory=lambda: BoxRect(x=0.65, y=0.68, w=0.30, h=0.22))
    chroma_color: str = "#00FF00"
    similarity: float = Field(0.28, ge=0.01, le=0.8)
    blend: float = Field(0.08, ge=0.0, le=0.5)
    start_sec: float = Field(0.0, ge=0.0)
    loop_enabled: bool = False
    mute_cta_audio: bool = True


class SubRemovalConfig(BaseModel):
    mode: Literal["none", "blur", "ai_clean"] = "none"
    box: BoxRect = Field(default_factory=lambda: BoxRect(x=0.12, y=0.84, w=0.76, h=0.12))
    blur_strength: int = Field(24, ge=4, le=80)
    ai_engine: Literal["sttn", "lama"] = "sttn"
    auto_detect_ocr: bool = True
    ocr_sample_fps: float = Field(2.0, ge=1.0, le=15.0)  # frames between samples use the union of both samples
    mask_dilate_px: int = Field(5, ge=1, le=20)


class BackgroundMusicConfig(BaseModel):
    """Background music playlist with seamless equal-power overlapping crossfade (fade-out tail + fade-in head)."""
    enabled: bool = False
    file_paths: List[str] = Field(default_factory=list)
    volume: float = Field(0.22, ge=0.0, le=1.5)
    crossfade_sec: float = Field(3.5, ge=0.5, le=15.0)
    auto_ducking: bool = True
    duck_level: float = Field(0.08, ge=0.0, le=1.0)
    fade_in_sec: float = Field(1.5, ge=0.0, le=10.0)
    fade_out_sec: float = Field(2.5, ge=0.0, le=10.0)


class AudioConfig(BaseModel):
    edit_mode: Literal["narration", "dubbing"] = "narration"
    original_volume: float = Field(0.15, ge=0.0, le=2.0)
    mute_original: bool = False
    auto_ducking: bool = True
    duck_level: float = Field(0.12, ge=0.0, le=1.0)
    max_atempo: float = Field(1.25, ge=1.0, le=2.0)
    voice_gain: float = Field(1.15, ge=0.2, le=3.0)


class TranslateConfig(BaseModel):
    source_lang: str = "zh"
    target_lang: str = "vi"
    provider: Literal["antigravity", "gemini_api"] = "antigravity"
    model: str = "gemini-4-neon"
    thinking_level: Literal["minimal", "low", "medium", "high", "xhigh", "max"] = "high"
    batch_size: int = Field(45, ge=10, le=120)
    length_tolerance_pct: int = Field(15, ge=5, le=50)
    concurrency: int = Field(4, ge=1, le=16)  # number of batches translated in parallel
    agy_path: str = ""
    api_key: str = ""


class VoiceConfig(BaseModel):
    provider: Literal["capcut_tiktok", "edge_tts", "gtts"] = "capcut_tiktok"
    language: str = "vi"
    voice_id: str = "BV074_streaming"
    voice_name: str = "CapCut VN - Cô Gái Hoạt Bát (Nữ)"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    concurrency: int = Field(12, ge=1, le=32)
    include_multilingual: bool = True
    tiktok_session_id: str = ""
    fallback_edge_tts: bool = True


class EditConfig(BaseModel):
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    logo: LogoConfig = Field(default_factory=LogoConfig)
    green_screen: GreenScreenConfig = Field(default_factory=GreenScreenConfig)
    sub_removal: SubRemovalConfig = Field(default_factory=SubRemovalConfig)
    bgm: BackgroundMusicConfig = Field(default_factory=BackgroundMusicConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    performance_profile: Literal["auto", "balanced", "dual_gpu_max", "workstation"] = "auto"
    encoder_override: Literal["auto", "h264_qsv", "h264_nvenc", "libx264"] = "auto"


class ChannelPreset(BaseModel):
    id: str
    name: str
    config: EditConfig = Field(default_factory=EditConfig)


class VideoProject(BaseModel):
    id: str
    name: str
    video_path: str
    srt_path: str = ""
    translated_srt_path: str = ""
    style_guide_path: str = ""
    voice_dir: str = ""
    dub_audio_path: str = ""
    clean_video_path: str = ""
    output_path: str = ""
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    duration: float = 0.0
    has_audio: bool = True
    status: Literal["ready", "downloading", "translating", "voicing", "cleaning", "rendering", "done", "error"] = "ready"
    progress: float = 0.0
    status_message: str = ""
    preset_id: Optional[str] = None
    config: EditConfig = Field(default_factory=EditConfig)


class SubtitleCue(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    text: str


class RenderRequest(BaseModel):
    project_id: str
    preview_seconds: Optional[float] = None
    use_voice: bool = True
    run_clean_sub: bool = True


class DownloadVideoRequest(BaseModel):
    url: str
    custom_name: str = ""
    sub_lang: str = "zh"  # Preferred subtitle language (e.g. zh, vi, en, auto)
    threads: int = Field(16, ge=1, le=32)
    browser_cookies: Literal["none", "edge", "chrome"] = "none"
    clean_srt_rules: bool = True
    clean_srt_ai: bool = False
    clean_model: str = "gemini-3.8-flash"
    clean_thinking_level: Literal["minimal", "low", "medium", "high", "xhigh"] = "low"
    api_key: str = ""
    preset_id: Optional[str] = None
