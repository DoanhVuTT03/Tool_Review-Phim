from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
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
    # entrance animation of every cue (+ short fade-out), see app/services/effects.py
    effect: Literal["none", "fade", "slide_up", "pop", "zoom_in", "typewriter", "blur_in", "bounce", "slide_left"] = "none"
    effect_duration: float = Field(0.3, ge=0.1, le=1.5)


class ChannelTextConfig(BaseModel):
    """Channel name text (e.g. "Review Phim Hay"): same look options as the subtitle box + a looping effect."""
    enabled: bool = False
    text: str = "Tên Kênh Của Bạn"
    box: BoxRect = Field(default_factory=lambda: BoxRect(x=0.03, y=0.03, w=0.3, h=0.08))
    font_path: str = ""
    font_size: int = Field(40, ge=10, le=200)
    bold: bool = True
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = Field(2, ge=0, le=12)
    bg_enabled: bool = True
    bg_color: str = "#E11D48"
    bg_opacity: float = Field(0.85, ge=0.0, le=1.0)
    border_radius: int = Field(14, ge=0, le=80)
    padding_x: int = Field(16, ge=0, le=80)
    padding_y: int = Field(8, ge=0, le=80)
    align: Literal["left", "center", "right"] = "center"
    valign: Literal["top", "middle", "bottom"] = "middle"
    line_spacing: float = Field(1.15, ge=0.8, le=2.5)
    effect: Literal["none", "breathe", "float", "pulse", "shimmer", "slide_in", "typewriter"] = "none"
    effect_period_sec: float = Field(8.0, ge=2.0, le=120.0)  # slide_in / typewriter / shimmer repeat every N s


class WatermarkRemovalConfig(BaseModel):
    """Burned-in text logos / watermarks (e.g. "本作品含AI生成内容") in one or more user-drawn regions.
    ai_clean: OCR scans every region and removes the text it finds (same engine as AI Clean Sub);
    blur: plain box blur of each region."""
    enabled: bool = False
    mode: Literal["ai_clean", "blur"] = "ai_clean"
    boxes: List[BoxRect] = Field(default_factory=lambda: [BoxRect(x=0.70, y=0.02, w=0.28, h=0.07)])
    blur_strength: int = Field(24, ge=4, le=80)


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
    provider: Literal["antigravity", "gemini_api", "gemini_web"] = "antigravity"
    model: str = "gemini-4-neon"
    thinking_level: Literal["minimal", "low", "medium", "high", "xhigh", "max"] = "high"
    # lines per translation request. Big batches are fine: prompts over the 32 K command line go to the
    # Antigravity agent as a file it reads itself (translator._call_agentapi)
    batch_size: int = Field(45, ge=10, le=2000)
    length_tolerance_pct: int = Field(15, ge=5, le=50)
    concurrency: int = Field(4, ge=1, le=16)  # number of batches translated in parallel
    # Translate the WHOLE SRT in ONE Antigravity conversation (the agent reads the file and writes the result
    # files itself): one context for the whole film -> most consistent names / pronouns. Lines it misses are
    # translated afterwards in normal batches, so the result is always complete.
    whole_file: bool = False
    # every line gets a length budget (max syllables) from its time on screen; lines still too long after
    # translating are rewritten shorter -> the voice is not sped up / rushed (app/services/length_fit.py)
    fit_timing: bool = True
    agy_path: str = ""
    api_key: str = ""
    # ---- Gemini Web (gemini.google.com, no API key - app/services/gemini_web.py) ----
    # manual: the tool shows each prompt, the user sends it in Gemini and pastes the answer back;
    # auto: the tool drives its own Chrome window (logged in once) and sends every part by itself.
    # Either way ONE request at a time, and every answer is checked (missing / extra / untranslated lines).
    web_mode: Literal["manual", "auto"] = "manual"
    # a file with up to this many lines goes in one part; longer files are split into equal parts of at
    # most this size (web chats answer badly / drop lines on very long inputs)
    web_chunk_lines: int = Field(250, ge=20, le=3000)
    web_new_chat: bool = True  # auto mode: a fresh Gemini chat for every part


class VoiceConfig(BaseModel):
    provider: Literal["capcut_tiktok", "edge_tts", "gtts", "higgs3"] = "capcut_tiktok"
    language: str = "vi"
    voice_id: str = "BV074_streaming"
    voice_name: str = "CapCut VN - Cô Gái Hoạt Bát (Nữ)"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    # parallel TTS requests. Edge-TTS handles hundreds; CapCut/TikTok relays and gTTS are capped lower in
    # tts_engine (they rate-limit hard)
    concurrency: int = Field(12, ge=1, le=500)
    include_multilingual: bool = True
    tiktok_session_id: str = ""
    fallback_edge_tts: bool = True
    # after the voice is made: lines whose real audio is longer than their time on screen are rewritten
    # shorter (LLM) and voiced again -> no rushed / sped-up voice (app/services/length_fit.py)
    auto_fit: bool = True
    # ---- Higgs TTS 3 (voice clone, WSL server, see app/services/higgs_voice.py) ----
    higgs_ref_audio: str = ""  # reference voice file (its words: higgs_ref_text, else the .txt next to it)
    higgs_ref_text: str = ""
    higgs_clean_ref: bool = True  # remove music / reverb / noise from the reference first
    higgs_clean_strong: bool = False  # strong: run all 3 cleaning models (default: only those that measurably help)
    higgs_review: bool = True  # Whisper hears every line -> suspicious lines go to the review table
    higgs_auto_retry: int = Field(1, ge=0, le=3)  # automatic new versions of suspicious lines before listing
    higgs_pause_review: bool = True  # A-Z run stops after the voice when suspicious lines are left
    higgs_temperature: float = Field(0.8, ge=0.1, le=1.5)


class EditConfig(BaseModel):
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    logo: LogoConfig = Field(default_factory=LogoConfig)
    green_screen: GreenScreenConfig = Field(default_factory=GreenScreenConfig)
    sub_removal: SubRemovalConfig = Field(default_factory=SubRemovalConfig)
    wm_removal: WatermarkRemovalConfig = Field(default_factory=WatermarkRemovalConfig)
    channel_text: ChannelTextConfig = Field(default_factory=ChannelTextConfig)
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
    timing: Dict[str, Any] = Field(default_factory=dict)  # stopwatch of the last job (app/services/timing.py)


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
    clean_batch_lines: int = Field(0, ge=0, le=100000)  # lines per AI request, 0 = automatic (as many as fit)
    api_key: str = ""
    preset_id: Optional[str] = None
