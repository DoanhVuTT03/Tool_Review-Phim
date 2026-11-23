from __future__ import annotations
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from app.config import GEM_TXT_PATH, LANGUAGE_NAME_MAP, PROJECTS_DIR
from app.models import SubtitleCue, TranslateConfig, VideoProject
from app.services.cancel import JobCancelled, check_cancel, is_cancelled, run_tracked
from app.services.media import parse_srt, write_srt


ProgressCallback = Callable[[float, str], None]

_THINKING_BUDGET_MAP = {
    "low": 1024,
    "medium": 4096,
    "high": 12288,
    "max": 24576,
}

_LINE_RE = re.compile(r"^\s*(\d+)\s*\|\|\|\s*(.+?)\s*$")


def get_antigravity_agentapi_path(custom_path: str = "") -> Optional[Tuple[str, List[str]]]:
    """Returns (executable_path, prefix_args) for Antigravity agentapi."""
    if custom_path and Path(custom_path).exists():
        p = Path(custom_path)
        if p.name.lower() == "language_server.exe":
            return (str(p), ["agentapi"])
        return (str(p), [])

    # Prefer direct binary to avoid .bat newline stripping issues on Windows
    ls_exe = Path.home() / "AppData" / "Local" / "Programs" / "antigravity" / "resources" / "bin" / "language_server.exe"
    if ls_exe.exists():
        return (str(ls_exe), ["agentapi"])

    candidates = [
        Path.home() / ".gemini" / "antigravity" / "bin" / "agentapi.bat",
        Path.home() / ".gemini" / "antigravity" / "bin" / "agentapi",
        Path.home() / "AppData" / "Roaming" / "Antigravity" / "bin" / "agy-node.cmd",
    ]
    for p in candidates:
        if p.exists():
            return (str(p), [])

    for name in ("agentapi", "agy"):
        w = shutil.which(name)
        if w:
            return (w, [])

    return None


def _port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_antigravity_ls_env() -> Optional[Dict[str, str]]:
    """`agentapi` only works when ANTIGRAVITY_LS_ADDRESS / ANTIGRAVITY_CSRF_TOKEN are set
    (Antigravity sets them only inside its own terminal). Outside Antigravity we rebuild them
    from the running language server's hub file: ~/.gemini/antigravity/daemon/hub_<port>.json.
    Returns None when the Antigravity app is not running.
    """
    if os.environ.get("ANTIGRAVITY_LS_ADDRESS") and os.environ.get("ANTIGRAVITY_CSRF_TOKEN"):
        return {}
    daemon_dir = Path.home() / ".gemini" / "antigravity" / "daemon"
    if not daemon_dir.is_dir():
        return None
    hubs = sorted(daemon_dir.glob("hub_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for hub in hubs:
        try:
            info = json.loads(hub.read_text(encoding="utf-8"))
            port = int(info.get("httpPort") or 0)
            token = str(info.get("csrfToken") or "")
        except Exception:
            continue
        if port and token and _port_open("127.0.0.1", port):
            env = {
                "ANTIGRAVITY_LS_ADDRESS": f"127.0.0.1:{port}",
                "ANTIGRAVITY_CSRF_TOKEN": token,
            }
            if info.get("lsVersion"):
                env["ANTIGRAVITY_LS_VERSION"] = str(info["lsVersion"])
            return env
    return None


def detect_translation_backends(custom_agy_path: str = "") -> Dict[str, Any]:
    agentapi_info = get_antigravity_agentapi_path(custom_agy_path)
    agy_bin = ""
    if custom_agy_path and Path(custom_agy_path).exists():
        agy_bin = custom_agy_path
    else:
        agy_bin = shutil.which("agy") or ""

    sdk_ok = False
    try:
        import google.antigravity  # type: ignore # noqa: F401
        sdk_ok = True
    except Exception:
        sdk_ok = False

    env_gemini_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    return {
        "agentapi": {"available": bool(agentapi_info), "path": agentapi_info[0] if agentapi_info else ""},
        "agy_cli": {"available": bool(agy_bin), "path": agy_bin},
        "antigravity_sdk": {"available": sdk_ok},
        "gemini_api_env": {"available": env_gemini_key},
        "gem_txt_exists": GEM_TXT_PATH.exists(),
    }


def _load_gem_system_prompt() -> str:
    if GEM_TXT_PATH.exists():
        return GEM_TXT_PATH.read_text(encoding="utf-8").strip()
    return (
        "Bạn là dịch giả phụ đề chuyên nghiệp. Luôn trả về đúng định dạng '<số>|||<bản dịch>' "
        "cho từng dòng trong khung [TEXT_START] ... [TEXT_END], không thêm giải thích hay markdown."
    )


async def _call_agentapi(
    exec_info: Tuple[str, List[str]],
    model: str,
    thinking_level: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float = 240.0,
) -> str:
    """Uses local Antigravity Language Server / agentapi to run in a fresh session
    using the user's currently signed-in Google Antigravity account without requiring an API key.
    """
    exe_path, prefix_args = exec_info
    brain_dir = Path.home() / ".gemini" / "antigravity" / "brain"
    combined = (
        f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n[/SYSTEM INSTRUCTIONS]\n\n"
        f"{user_prompt}"
    )

    # Protect Windows 32KB CreateProcess command-line limit
    if len(combined) > 28000:
        combined = combined[:28000] + "\n[TEXT_TRUNCATED_FOR_LIMIT]\n[TEXT_END]"

    # Map model name
    m_low = (model or "").lower()
    if "flash" in m_low:
        model_arg = "flash"
    elif "pro" in m_low or "opus" in m_low or "sonnet" in m_low:
        model_arg = "pro"
    else:
        model_arg = model.strip() or "flash"

    ls_env = get_antigravity_ls_env()
    if ls_env is None:
        raise RuntimeError(
            "Không kết nối được Antigravity: app Antigravity chưa mở. "
            "Hãy mở app Antigravity (đã đăng nhập Google) rồi chạy lại, "
            "hoặc nhập Gemini API Key ở Tab Dịch SRT."
        )
    env = os.environ.copy()
    env.update(ls_env)

    def _exec() -> str:
        cmd = [exe_path] + prefix_args + ["new-conversation", f"--model={model_arg}", combined]
        res = run_tracked(cmd, timeout=timeout, env=env)
        if res.returncode != 0:
            detail = (res.stderr or "").strip()
            try:
                detail = json.loads(res.stdout).get("error") or detail
            except Exception:
                detail = detail or (res.stdout or "").strip()
            raise RuntimeError(f"Antigravity agentapi lỗi (code {res.returncode}): {str(detail)[:300]}")
        try:
            data = json.loads(res.stdout)
            conv_id = data["response"]["newConversation"]["conversationId"]
        except Exception as e:
            raise RuntimeError(f"agentapi trả về dữ liệu không hợp lệ: {res.stdout[:300]}") from e

        log_dir = brain_dir / conv_id / ".system_generated" / "logs"
        compact_path = log_dir / "transcript.jsonl"
        full_path = log_dir / "transcript_full.jsonl"

        t_end = time.time() + timeout
        while time.time() < t_end:
            check_cancel()
            if compact_path.exists():
                try:
                    lines = compact_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                    for idx, ln in enumerate(reversed(lines)):
                        if not ln.strip():
                            continue
                        item = json.loads(ln)
                        if item.get("type") == "PLANNER_RESPONSE" and item.get("status") == "DONE":
                            if "content" in item.get("truncated_fields", []):
                                line_idx = len(lines) - 1 - idx
                                full_lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
                                if line_idx < len(full_lines):
                                    full_item = json.loads(full_lines[line_idx])
                                    return full_item.get("content", "").strip()
                            return item.get("content", "").strip()
                except Exception:
                    pass
            time.sleep(0.8)
        raise TimeoutError("Hết thời gian chờ phản hồi từ Antigravity session (quá 240 giây).")

    return await asyncio.to_thread(_exec)


async def _call_agy_cli(
    agy_path: str,
    model: str,
    thinking_level: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Invokes Antigravity CLI (`agy`) in headless single-shot fresh session mode."""
    combined_prompt = (
        f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n[/SYSTEM INSTRUCTIONS]\n\n"
        f"{user_prompt}"
    )

    # Pass prompt via stdin to avoid Windows 32KB cmdline limit on large SRT batches
    cmd = [agy_path, "-p", "-", "--model", model]
    # Support effort/thinking flag if CLI accepts it; fall back cleanly if older version
    env = os.environ.copy()
    env["AGY_EFFORT"] = thinking_level

    def _run():
        res = run_tracked(cmd, input=combined_prompt, timeout=300, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"Antigravity CLI (`agy`) lỗi (code {res.returncode}): {res.stderr.strip()[:400]}")
        return res.stdout.strip()

    return await asyncio.to_thread(_run)


async def _call_antigravity_sdk(
    model: str,
    thinking_level: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str = "",
) -> str:
    """Invokes google-antigravity Python SDK with a brand-new Agent session & exact ThinkingLevel."""
    from google.antigravity import (  # type: ignore
        Agent,
        GeminiAPIEndpoint,
        GeminiModelOptions,
        LocalAgentConfig,
        ModelTarget,
        ThinkingLevel,
    )

    lvl_map = {
        "minimal": ThinkingLevel.MINIMAL,
        "low": ThinkingLevel.LOW,
        "medium": ThinkingLevel.MEDIUM,
        "high": ThinkingLevel.HIGH,
        "xhigh": ThinkingLevel.EXTRA_HIGH,
        "max": ThinkingLevel.EXTRA_HIGH,
    }
    model_name = (model or "gemini-4-neon").strip()
    eff_thinking = thinking_level
    if model_name == "orchestrator":
        model_name = "gemini-4-neon"
        eff_thinking = "xhigh"

    target = ModelTarget(
        name=model_name,
        endpoint=GeminiAPIEndpoint(
            api_key=api_key or None,
            options=GeminiModelOptions(
                thinking_level=lvl_map.get(eff_thinking, ThinkingLevel.MEDIUM)
            ),
        ),
    )
    config = LocalAgentConfig(
        system_instructions=system_prompt,
        model=target,
        api_key=api_key or None,
    )

    chunks: List[str] = []
    async with Agent(config) as agent:
        response = await agent.chat(user_prompt)
        async for token in response:
            chunks.append(str(token))
    return "".join(chunks).strip()


async def _call_gemini_rest_api(
    api_key: str,
    model: str,
    thinking_level: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Direct Gemini API call with explicit thinkingConfig & safety settings disabled for fictional movie scripts."""
    model_id = model.strip() or "gemini-2.5-pro"
    if not model_id.startswith("models/"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_id}:generateContent?key={api_key}"

    budget = _THINKING_BUDGET_MAP.get(thinking_level, 4096)
    payload: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.35,
            "thinkingConfig": {"thinkingBudget": budget},
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code == 400 and "thinkingConfig" in resp.text:
            # Retry without thinkingConfig for non-thinking models
            payload["generationConfig"].pop("thinkingConfig", None)
            resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API lỗi HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Model không trả về nội dung: {json.dumps(data, ensure_ascii=False)[:300]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if not p.get("thought") and p.get("text")]
    return "\n".join(text_parts).strip()


async def invoke_llm(
    cfg: TranslateConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Dispatches to Antigravity (Local agentapi with signed-in user account / CLI) or Gemini API key."""
    check_cancel()
    backends = detect_translation_backends(cfg.agy_path)
    api_key = (cfg.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

    if cfg.provider == "antigravity":
        # 1. If API key is provided, prefer direct API / SDK call
        if api_key:
            if backends["antigravity_sdk"]["available"]:
                try:
                    return await _call_antigravity_sdk(
                        cfg.model,
                        cfg.thinking_level,
                        system_prompt,
                        user_prompt,
                        api_key=api_key,
                    )
                except Exception:
                    pass
            return await _call_gemini_rest_api(
                api_key,
                cfg.model,
                cfg.thinking_level,
                system_prompt,
                user_prompt,
            )

        # 2. If no API key provided, use local Antigravity Language Server / agentapi (uses signed-in Google account!)
        agentapi_path = get_antigravity_agentapi_path(cfg.agy_path)
        if agentapi_path:
            return await _call_agentapi(
                agentapi_path,
                cfg.model,
                cfg.thinking_level,
                system_prompt,
                user_prompt,
            )

        # 3. If agy CLI binary is found
        if backends["agy_cli"]["available"]:
            return await _call_agy_cli(
                backends["agy_cli"]["path"],
                cfg.model,
                cfg.thinking_level,
                system_prompt,
                user_prompt,
            )

        raise RuntimeError(
            "Chưa có Gemini API Key và không phát hiện phiên Antigravity đã đăng nhập. "
            "Vui lòng nhập API Key tại Tab Dịch SRT hoặc đăng nhập vào Google Antigravity."
        )

    if not api_key:
        raise RuntimeError("Vui lòng nhập Gemini API Key (hoặc đặt biến môi trường GEMINI_API_KEY).")
    return await _call_gemini_rest_api(
        api_key,
        cfg.model,
        cfg.thinking_level,
        system_prompt,
        user_prompt,
    )


def _project_work_dir(project: VideoProject) -> Path:
    d = PROJECTS_DIR / project.id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sample_cues_for_analysis(cues: List[SubtitleCue], max_lines: int = 250) -> List[SubtitleCue]:
    if len(cues) <= max_lines:
        return cues
    head = cues[:120]
    mid_start = max(120, (len(cues) - 80) // 2)
    mid = cues[mid_start : mid_start + 80]
    tail = cues[-50:]
    return head + mid + tail


def parse_translated_lines(raw_output: str) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    for line in raw_output.splitlines():
        m = _LINE_RE.match(line)
        if m:
            idx = int(m.group(1))
            txt = m.group(2).strip()
            if txt:
                parsed[idx] = txt
    return parsed


def _srt_hash(srt_path: str) -> str:
    return hashlib.sha1(Path(srt_path).read_bytes()).hexdigest()[:16]


def load_translate_checkpoint(project: VideoProject) -> Dict[str, Any]:
    cp_file = _project_work_dir(project) / "translate_state.json"
    if not cp_file.exists():
        return {}
    try:
        return json.loads(cp_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_translate_checkpoint(project: VideoProject, state_data: Dict[str, Any]) -> None:
    cp_file = _project_work_dir(project) / "translate_state.json"
    cp_file.write_text(json.dumps(state_data, ensure_ascii=False, indent=2), encoding="utf-8")


async def analyze_style_guide(
    project: VideoProject,
    on_progress: Optional[ProgressCallback] = None,
    llm_caller=invoke_llm,
) -> Path:
    """Phase 2A: Analyze full/sampled SRT using gem.txt rules -> save <stem>_style_guide.md."""
    if not project.srt_path or not Path(project.srt_path).exists():
        raise RuntimeError("Video chưa có file phụ đề .srt gốc.")

    cues = parse_srt(project.srt_path)
    if not cues:
        raise RuntimeError("File SRT rỗng hoặc không đúng định dạng.")

    cfg = project.config.translate
    src_name = LANGUAGE_NAME_MAP.get(cfg.source_lang, cfg.source_lang)
    tgt_name = LANGUAGE_NAME_MAP.get(cfg.target_lang, cfg.target_lang)
    system_prompt = _load_gem_system_prompt()

    if on_progress:
        on_progress(5.0, f"Pha 1/2: Đang phân tích Topic, Context & Style Guide ({len(cues)} dòng)...")

    sampled = _sample_cues_for_analysis(cues)
    text_block = "\n".join(f"{c.index}|||{c.text.replace(chr(10), ' ')}" for c in sampled)

    user_prompt = (
        f"PHÂN TÍCH FILE ĐÍNH KÈM (KHÔNG DỊCH).\n"
        f"- Ngôn ngữ nguồn: {src_name}\n"
        f"- Ngôn ngữ đích: {tgt_name}\n"
        f"- Tên phim/tệp: {project.name}\n\n"
        f"Hãy đọc và hiểu toàn bộ nội dung phụ đề dưới đây, chốt ra STYLE GUIDE ngắn gọn, chính xác theo đúng 4 mục:\n"
        f"1. CHỦ ĐỀ (Topic): Thể loại phim, đề tài chính.\n"
        f"2. NGỮ CẢNH (Context): Bối cảnh thời đại, tình huống cốt truyện chính.\n"
        f"3. DANH SÁCH NHÂN VẬT & XƯNG HÔ: Tên dịch chuẩn từng nhân vật và cách xưng hô giữa các cặp nhân vật (bắt buộc nhất quán xuyên suốt).\n"
        f"4. VĂN PHONG (Style Guide): Giọng điệu (review phim cuốn hút / cổ trang trang trọng / đời thường / hành động...), các thuật ngữ cố định cần giữ nguyên.\n\n"
        f"[TEXT_START]\n{text_block}\n[TEXT_END]"
    )

    guide_text = await llm_caller(cfg, system_prompt, user_prompt)
    stem = re.sub(r'[<>:"/\\|?*]+', "_", Path(project.name).stem)
    out_path = _project_work_dir(project) / f"{stem}_style_guide.md"
    out_path.write_text(guide_text, encoding="utf-8")

    if on_progress:
        on_progress(15.0, f"Đã lưu Style Guide: {out_path.name}")
    return out_path


async def translate_project_srt(
    project: VideoProject,
    reset_checkpoint: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    llm_caller=invoke_llm,
) -> Path:
    """Phase 2B: Opens fresh sessions per batch with gem.txt + saved Style Guide, enforces line length parity,
    persists checkpoint after every batch, and writes translated .srt.
    """
    if not project.srt_path or not Path(project.srt_path).exists():
        raise RuntimeError("Video chưa có file phụ đề .srt gốc.")

    cues = parse_srt(project.srt_path)
    if not cues:
        raise RuntimeError("File SRT rỗng hoặc không đọc được.")

    cfg = project.config.translate
    work_dir = _project_work_dir(project)
    src_hash = _srt_hash(project.srt_path)

    # Step 1: Ensure Style Guide exists (or regenerate if reset)
    style_path = Path(project.style_guide_path) if project.style_guide_path else None
    if reset_checkpoint or not style_path or not style_path.exists():
        style_path = await analyze_style_guide(project, on_progress=on_progress, llm_caller=llm_caller)
        project.style_guide_path = str(style_path)

    style_guide = style_path.read_text(encoding="utf-8").strip()
    system_prompt = _load_gem_system_prompt()
    src_name = LANGUAGE_NAME_MAP.get(cfg.source_lang, cfg.source_lang)
    tgt_name = LANGUAGE_NAME_MAP.get(cfg.target_lang, cfg.target_lang)

    # Step 2: Load or initialize checkpoint
    cp = {} if reset_checkpoint else load_translate_checkpoint(project)
    if (
        cp.get("src_hash") != src_hash
        or cp.get("target_lang") != cfg.target_lang
        or cp.get("total_cues") != len(cues)
    ):
        cp = {
            "src_hash": src_hash,
            "source_lang": cfg.source_lang,
            "target_lang": cfg.target_lang,
            "total_cues": len(cues),
            "translated": {},
            "updated_at": time.time(),
        }
        save_translate_checkpoint(project, cp)

    translated_map: Dict[int, str] = {int(k): v for k, v in cp.get("translated", {}).items()}
    batch_size = max(10, min(120, cfg.batch_size))

    # Build batches of cues (skipping already-translated batches cleanly)
    batches: List[List[SubtitleCue]] = [
        cues[i : i + batch_size] for i in range(0, len(cues), batch_size)
    ]

    n_threads = max(1, min(16, int(getattr(cfg, "concurrency", 1) or 1)))
    sem = asyncio.Semaphore(n_threads)
    pos_of = {c.index: i for i, c in enumerate(cues)}
    done_batches = 0
    active_batches: set = set()

    def _report(extra: str = "") -> None:
        if not on_progress or is_cancelled():
            return
        pct = 15.0 + (len(translated_map) / len(cues)) * 83.0
        running = ", ".join(str(i) for i in sorted(active_batches)) or "-"
        on_progress(
            round(pct, 1),
            f"Pha 2/2: Dịch {n_threads} luồng song song — xong {done_batches}/{len(batches)} lô "
            f"({len(translated_map)}/{len(cues)} dòng), đang chạy lô [{running}]{extra}...",
        )

    async def _call_with_retry(user_prompt: str, b_idx: int) -> str:
        # Retries transient backend errors (rate limit / timeouts) — more likely with many threads.
        last_err: Optional[Exception] = None
        for n in range(3):
            check_cancel()
            try:
                return await llm_caller(cfg, system_prompt, user_prompt)
            except (JobCancelled, asyncio.CancelledError):
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                if is_cancelled():
                    raise JobCancelled("Đã dừng tiến trình theo yêu cầu.") from e
                _report(f" (lô {b_idx} lỗi, thử lại {n + 1}/2: {str(e)[:80]})")
                await asyncio.sleep(3 * (n + 1))
        assert last_err is not None
        raise last_err

    async def _translate_batch(b_idx: int, batch: List[SubtitleCue]) -> None:
        nonlocal done_batches
        pending = [c for c in batch if c.index not in translated_map]
        if not pending:
            done_batches += 1
            return
        async with sem:
            check_cancel()
            active_batches.add(b_idx)
            _report()
            try:
                await _run_batch(b_idx, batch, pending)
            finally:
                active_batches.discard(b_idx)
        done_batches += 1
        # Persist checkpoint immediately after each batch (single event-loop thread -> no race)
        cp["translated"] = {str(k): v for k, v in translated_map.items()}
        cp["updated_at"] = time.time()
        save_translate_checkpoint(project, cp)
        _report()

    async def _run_batch(b_idx: int, batch: List[SubtitleCue], pending: List[SubtitleCue]) -> None:
        # Continuity context: 4 lines right before this batch. With parallel threads the previous batch
        # is often not translated yet, so fall back to the original source lines.
        first_pos = pos_of[batch[0].index]
        ctx = cues[max(0, first_pos - 4) : first_pos]
        if ctx and all(c.index in translated_map for c in ctx):
            prev_block = (
                "VÀI DÒNG CUỐI LÔ TRƯỚC ĐỂ NỐI MẠCH (BẢN DỊCH, CHỈ THAM KHẢO, KHÔNG DỊCH LẠI):\n"
                + "\n".join(f"{c.index}|||{translated_map[c.index]}" for c in ctx)
                + "\n\n"
            )
        elif ctx:
            prev_block = (
                "VÀI DÒNG GỐC NGAY TRƯỚC LÔ NÀY ĐỂ NỐI MẠCH (CHỈ THAM KHẢO, KHÔNG DỊCH, KHÔNG TRẢ VỀ):\n"
                + "\n".join(f"{c.index}|||{c.text.replace(chr(10), ' ')}" for c in ctx)
                + "\n\n"
            )
        else:
            prev_block = ""

        # Retry loop for any missing/malformed lines in the batch
        remaining = pending
        for attempt in range(1, 4):
            if attempt > 1:
                _report(f" (lô {b_idx} thử lại lần {attempt - 1})")

            lines_payload = "\n".join(
                f"{c.index}|||{c.text.replace(chr(10), ' ')}" for c in remaining
            )
            user_prompt = (
                f"NGÔN NGỮ NGUỒN: {src_name}\n"
                f"NGÔN NGỮ ĐÍCH: {tgt_name}\n\n"
                f"=== STYLE GUIDE ĐÃ CHỐT (TUÂN THỦ TUYỆT ĐỐI) ===\n"
                f"{style_guide}\n"
                f"================================================\n\n"
                f"YÊU CẦU ĐỘ DÀI DÒNG SRT CHO VOICE AI:\n"
                f"- Dịch súc tích, tự nhiên sao cho độ dài câu dịch vừa đủ hoặc chỉ chênh tối đa ±{cfg.length_tolerance_pct}% so với nhịp nói câu gốc, để khi lồng tiếng/thuyết minh không bị nói quá nhanh hay tràn khung thời gian.\n"
                f"- KHÔNG dùng dấu '...' hay '…' để nối câu.\n\n"
                f"{prev_block}"
                f"[TEXT_START]\n{lines_payload}\n[TEXT_END]"
            )

            raw_resp = await _call_with_retry(user_prompt, b_idx)
            got = parse_translated_lines(raw_resp)
            for c in remaining:
                if c.index in got:
                    translated_map[c.index] = got[c.index]

            remaining = [c for c in remaining if c.index not in translated_map]
            if not remaining:
                break

        # Fallback: if any single stubborn line still missed after 3 tries, keep original text rather than dropping cue
        for c in remaining:
            translated_map[c.index] = c.text

    restored = sum(1 for b in batches if all(c.index in translated_map for c in b))
    if restored and on_progress:
        on_progress(
            round(15.0 + (len(translated_map) / len(cues)) * 83.0, 1),
            f"Đã khôi phục {restored}/{len(batches)} lô từ checkpoint, dịch tiếp bằng {n_threads} luồng...",
        )

    tasks = [asyncio.create_task(_translate_batch(i, b)) for i, b in enumerate(batches, start=1)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        # One batch failed or user pressed Dừng -> stop the other threads; finished batches stay in checkpoint
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        cp["translated"] = {str(k): v for k, v in translated_map.items()}
        cp["updated_at"] = time.time()
        save_translate_checkpoint(project, cp)
        raise

    # Write final translated SRT preserving exact original timestamps
    out_cues = [
        SubtitleCue(
            index=c.index,
            start_sec=c.start_sec,
            end_sec=c.end_sec,
            text=translated_map.get(c.index, c.text),
        )
        for c in cues
    ]
    stem = re.sub(r'[<>:"/\\|?*]+', "_", Path(project.name).stem)
    out_srt = work_dir / f"{stem}.{cfg.target_lang}.srt"
    write_srt(out_cues, str(out_srt))

    if on_progress:
        on_progress(100.0, f"Dịch xong {len(out_cues)} dòng! Đã lưu: {out_srt.name}")
    return out_srt
