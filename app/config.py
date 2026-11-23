from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = BASE_DIR / "bin"
DATA_DIR = BASE_DIR / "data"
PRESETS_DIR = DATA_DIR / "presets"
PROJECTS_DIR = DATA_DIR / "projects"
DOWNLOADS_DIR = DATA_DIR / "downloads"
OUTPUTS_DIR = DATA_DIR / "outputs"
TEMP_DIR = DATA_DIR / "temp"
MODELS_DIR = DATA_DIR / "models"
GEM_TXT_PATH = BASE_DIR / "gem.txt"
WEB_DIR = BASE_DIR / "app" / "web"

for _d in (BIN_DIR, DATA_DIR, PRESETS_DIR, PROJECTS_DIR, DOWNLOADS_DIR, OUTPUTS_DIR, TEMP_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_ffmpeg_path() -> str:
    local_bin = BIN_DIR / "ffmpeg.exe"
    if local_bin.exists():
        return str(local_bin)
    found = shutil.which("ffmpeg")
    return found or str(local_bin)


def get_ffprobe_path() -> str:
    local_bin = BIN_DIR / "ffprobe.exe"
    if local_bin.exists():
        return str(local_bin)
    found = shutil.which("ffprobe")
    return found or str(local_bin)


def get_aria2c_path() -> str:
    local_bin = BIN_DIR / "aria2c.exe"
    if local_bin.exists():
        return str(local_bin)
    return shutil.which("aria2c") or ""


ANTIGRAVITY_MODELS = [
    {"id": "gemini-4-neon", "label": "Gemini 4 Neon (New)", "default_thinking": "high"},
    {"id": "gemini-4-argon", "label": "Gemini 4 Argon (New)", "default_thinking": "medium"},
    {"id": "gemini-4-hydrogen", "label": "Gemini 4 Hydrogen (New)", "default_thinking": "medium"},
    {"id": "orchestrator", "label": "Orchestrator — Gemini 4 Neon Extra High (New)", "default_thinking": "xhigh"},
    {"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash (Nhanh)", "default_thinking": "medium"},
    {"id": "claude-opus-5.5", "label": "Claude Opus 5.5 (New)", "default_thinking": "medium"},
    {"id": "claude-sonnet-5.5", "label": "Claude Sonnet 5.5 (New)", "default_thinking": "medium"},
    {"id": "gpt-oss-120b", "label": "GPT-OSS 120B (Medium)", "default_thinking": "medium"},
]


WORLD_LANGUAGES = [
    {"code": "zh", "name": "Tiếng Trung (Giản thể / Phồn thể)", "native": "中文"},
    {"code": "vi", "name": "Tiếng Việt", "native": "Tiếng Việt"},
    {"code": "en", "name": "Tiếng Anh", "native": "English"},
    {"code": "fil", "name": "Tiếng Philippines (Filipino / Tagalog)", "native": "Filipino"},
    {"code": "ja", "name": "Tiếng Nhật", "native": "日本語"},
    {"code": "ko", "name": "Tiếng Hàn", "native": "한국어"},
    {"code": "th", "name": "Tiếng Thái", "native": "ไทย"},
    {"code": "id", "name": "Tiếng Indonesia", "native": "Bahasa Indonesia"},
    {"code": "ms", "name": "Tiếng Malaysia", "native": "Bahasa Melayu"},
    {"code": "km", "name": "Tiếng Khmer", "native": "ភាសាខ្មែរ"},
    {"code": "lo", "name": "Tiếng Lào", "native": "ລາວ"},
    {"code": "my", "name": "Tiếng Myanmar", "native": "မြန်မာ"},
    {"code": "es", "name": "Tiếng Tây Ban Nha", "native": "Español"},
    {"code": "pt", "name": "Tiếng Bồ Đào Nha", "native": "Português"},
    {"code": "fr", "name": "Tiếng Pháp", "native": "Français"},
    {"code": "de", "name": "Tiếng Đức", "native": "Deutsch"},
    {"code": "it", "name": "Tiếng Ý", "native": "Italiano"},
    {"code": "ru", "name": "Tiếng Nga", "native": "Русский"},
    {"code": "ar", "name": "Tiếng Ả Rập", "native": "العربية"},
    {"code": "hi", "name": "Tiếng Hindi", "native": "हिन्दी"},
    {"code": "bn", "name": "Tiếng Bengal", "native": "বাংলা"},
    {"code": "ur", "name": "Tiếng Urdu", "native": "اردو"},
    {"code": "tr", "name": "Tiếng Thổ Nhĩ Kỳ", "native": "Türkçe"},
    {"code": "pl", "name": "Tiếng Ba Lan", "native": "Polski"},
    {"code": "nl", "name": "Tiếng Hà Lan", "native": "Nederlands"},
    {"code": "sv", "name": "Tiếng Thụy Điển", "native": "Svenska"},
    {"code": "uk", "name": "Tiếng Ukraine", "native": "Українська"},
    {"code": "cs", "name": "Tiếng Séc", "native": "Čeština"},
    {"code": "ro", "name": "Tiếng Romania", "native": "Română"},
    {"code": "el", "name": "Tiếng Hy Lạp", "native": "Ελληνικά"},
    {"code": "he", "name": "Tiếng Do Thái", "native": "עברית"},
    {"code": "fa", "name": "Tiếng Ba Tư", "native": "فارسی"},
    {"code": "hu", "name": "Tiếng Hungary", "native": "Magyar"},
    {"code": "fi", "name": "Tiếng Phần Lan", "native": "Suomi"},
    {"code": "da", "name": "Tiếng Đan Mạch", "native": "Dansk"},
    {"code": "no", "name": "Tiếng Na Uy", "native": "Norsk"},
    {"code": "sw", "name": "Tiếng Swahili", "native": "Kiswahili"},
]

LANGUAGE_NAME_MAP = {item["code"]: f"{item['name']} ({item['native']})" for item in WORLD_LANGUAGES}
LANGUAGE_NAME_MAP["tl"] = LANGUAGE_NAME_MAP["fil"]
