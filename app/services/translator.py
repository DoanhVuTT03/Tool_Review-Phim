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
    "minimal": 512,
    "low": 1024,
    "medium": 4096,
    "high": 12288,
    "xhigh": 24576,
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


# ---------------------------------------------------------------------------------------------------------------
# Antigravity model catalog (live, from the running language server)
# ---------------------------------------------------------------------------------------------------------------

_LEVELS = ["low", "medium", "high", "xhigh"]
_THINK_TO_LEVEL = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "xhigh"}
_CATALOG_CACHE: Dict[str, Any] = {"t": 0.0, "data": None}


def _split_level(model_id: str) -> Tuple[str, Optional[str]]:
    for lv in sorted(_LEVELS, key=len, reverse=True):
        if model_id.endswith("-" + lv):
            return model_id[: -len(lv) - 1], lv
    return model_id, None


def fetch_antigravity_models(max_age: float = 300.0) -> Optional[Dict[str, Any]]:
    """{"families": [{"id", "label", "levels": [...], "tag"}], "ids": {model_id: label}} from the running
    Antigravity language server (GetCascadeModelConfigData), cached a few minutes. None if the app is not running."""
    now = time.time()
    if _CATALOG_CACHE["data"] is not None and now - _CATALOG_CACHE["t"] < max_age:
        return _CATALOG_CACHE["data"]
    env = get_antigravity_ls_env()
    if not env or not env.get("ANTIGRAVITY_LS_ADDRESS"):
        return None
    try:
        r = httpx.post(
            f"http://{env['ANTIGRAVITY_LS_ADDRESS']}/exa.language_server_pb.LanguageServerService/GetCascadeModelConfigData",
            json={},
            headers={"Content-Type": "application/json", "x-codeium-csrf-token": env.get("ANTIGRAVITY_CSRF_TOKEN", ""),
                     "Connect-Protocol-Version": "1"},
            timeout=8.0,
        )
        r.raise_for_status()
        raw = r.json()
    except Exception:
        return None
    groups = {k: (v or {}).get("displayName", k) for k, v in (raw.get("groupedModels") or {}).items()}
    fam: Dict[str, Dict[str, Any]] = {}
    ids: Dict[str, str] = {}
    for c in raw.get("clientModelConfigs") or []:
        mid = str(c.get("modelId") or "")
        if not mid:
            continue
        ids[mid] = str(c.get("label") or mid)
        base, lv = _split_level(mid)
        # models with levels: "Gemini 4 Neon (High)" -> "Gemini 4 Neon"; without: keep the full label
        # ("Orchestrator (Neon)" / "(Carbon)" / "(Argon)" would otherwise all read "Orchestrator")
        label = groups.get(base) or (str(c.get("label") or base).split(" (")[0] if lv else str(c.get("label") or base))
        f = fam.setdefault(base, {"id": base, "label": label, "levels": [], "tag": str(c.get("tagTitle") or "")})
        if lv and lv not in f["levels"]:
            f["levels"].append(lv)
    for f in fam.values():
        f["levels"].sort(key=_LEVELS.index)
    data = {"families": list(fam.values()), "ids": ids}
    _CATALOG_CACHE.update(t=now, data=data)
    return data


def _normalize_family(model: str) -> str:
    m = (model or "").strip().lower()
    m = re.sub(r"(\d)\.(\d)(?=$|-)", r"\1-\2", m) if m.startswith("claude") else m  # claude-opus-5.5 -> 5-5
    return {"orchestrator": "gemini-pro-orchestrator-neon"}.get(m, m)


def resolve_agentapi_model(model: str, thinking_level: str) -> Tuple[str, str]:
    """(exact Antigravity model id, note). Neon + medium -> "gemini-4-neon-medium". When the model has no such
    level (e.g. Flash has no XHigh) the nearest one is used; a model that no longer exists falls back to Neon."""
    want = _THINK_TO_LEVEL.get((thinking_level or "medium").lower(), "medium")
    fam = _normalize_family(model) or "gemini-4-neon"
    cat = fetch_antigravity_models()
    if cat is None:  # app catalog unreachable: build the id by the known naming scheme
        return (fam if fam.startswith(("gemini-pro-orchestrator", "gpt-oss")) else f"{fam}-{want}"), ""
    if fam in cat["ids"]:
        return fam, ""
    families = {f["id"]: f for f in cat["families"]}
    note = ""
    if fam not in families:
        note = f"model '{model}' không còn trong Antigravity → dùng Gemini 4 Neon"
        fam = "gemini-4-neon" if "gemini-4-neon" in families else next(iter(families), fam)
    levels = families.get(fam, {}).get("levels") or []
    if not levels:
        return fam, note
    if want in levels:
        return f"{fam}-{want}", note
    wi = _LEVELS.index(want)
    best = min(levels, key=lambda lv: (abs(_LEVELS.index(lv) - wi), -_LEVELS.index(lv)))
    return f"{fam}-{best}", (note or f"{fam} không có mức {want} → dùng {best}")


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


# The prompt goes to agentapi as a command-line argument (Windows caps a command line at 32 767 chars). Longer
# inputs are written to a file and the agent reads it itself with its read_file tool (verified: it pages
# through the whole file), so a whole movie's SRT can be handled in ONE conversation, nothing is truncated.
_AGENTAPI_INLINE_LIMIT = 24_000


def _agentapi_final_answer(compact_path: Path, full_path: Path) -> Optional[str]:
    """The conversation's final answer = the last PLANNER_RESPONSE that is DONE and calls no tool.
    (Returning the first DONE planner step was a bug: when the agent used a tool - e.g. read_file - that step
    has no text, and the tool returned "" after ~4 s instead of the real answer.)"""
    lines = compact_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    for back, ln in enumerate(reversed(lines)):
        if not ln.strip():
            continue
        item = json.loads(ln)
        if item.get("type") != "PLANNER_RESPONSE":
            continue
        if item.get("status") != "DONE" or item.get("tool_calls"):
            return None  # the agent is still working (or just asked for a tool)
        if "content" in (item.get("truncated_fields") or []):
            line_idx = len(lines) - 1 - back
            full_lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line_idx < len(full_lines):
                return str(json.loads(full_lines[line_idx]).get("content", "")).strip()
        return str(item.get("content", "")).strip()
    return None


async def _call_agentapi(
    exec_info: Tuple[str, List[str]],
    model: str,
    thinking_level: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float = 900.0,
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
    input_file: Optional[Path] = None
    if len(combined) > _AGENTAPI_INLINE_LIMIT:
        from app.config import TEMP_DIR

        work = TEMP_DIR / "agentapi_inputs"
        work.mkdir(parents=True, exist_ok=True)
        input_file = work / f"input_{os.getpid()}_{time.time_ns()}.txt"
        input_file.write_text(user_prompt, encoding="utf-8")
        n_lines = user_prompt.count("\n") + 1
        combined = (
            f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n[/SYSTEM INSTRUCTIONS]\n\n"
            f"Dữ liệu đầu vào dài ({n_lines} dòng) nên được lưu trong file:\n{input_file}\n\n"
            "Dùng công cụ đọc file để đọc TOÀN BỘ file này. Nếu công cụ chỉ trả về một phần, tiếp tục đọc các "
            "đoạn tiếp theo cho tới dòng cuối cùng - không được bỏ sót dòng nào. Không sửa, không tạo, không xóa "
            "file nào. Đọc xong thì làm đúng theo SYSTEM INSTRUCTIONS với toàn bộ nội dung file và trả lời trực "
            "tiếp trong chat theo đúng định dạng yêu cầu."
        )

    # Exact Antigravity model id = model + thinking level ("gemini-4-neon-medium"). The old mapping sent
    # "gemini-4-neon" / "flash" / "pro", which agentapi does not know, so every conversation silently ran on
    # the model currently selected in the Antigravity app (seen in the transcripts: always the app default).
    model_arg, _note = resolve_agentapi_model(model, thinking_level)

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
        res = run_tracked(cmd, timeout=120, env=env)
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
                    ans = _agentapi_final_answer(compact_path, full_path)
                    if ans is not None:
                        return ans
                except Exception:
                    pass
            time.sleep(0.8)
        raise TimeoutError(f"Hết thời gian chờ phản hồi từ Antigravity (quá {int(timeout)} giây).")

    try:
        return await asyncio.to_thread(_exec)
    finally:
        if input_file is not None:
            try:
                input_file.unlink()
            except OSError:
                pass


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


# Gemini API: 503 (model overloaded) and 429 (per-minute limit) pass after a while -> wait and retry,
# up to ~6 minutes in total per request before giving up.
_GEMINI_TRANSIENT = {429, 500, 502, 503, 504}
_GEMINI_RETRY_WAITS = [5, 10, 20, 40, 60, 90, 120]


def _is_daily_quota(body: str) -> bool:
    return "PerDay" in body or "per day" in body.lower()


def _retry_delay_hint(resp: httpx.Response) -> float:
    """Seconds the server asks us to wait (Retry-After header or RetryInfo.retryDelay), 0 when none."""
    try:
        ra = resp.headers.get("retry-after")
        if ra:
            return min(180.0, float(ra))
        m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', resp.text)
        if m:
            return min(180.0, float(m.group(1)) + 1.0)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


async def _sleep_cancellable(seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        check_cancel()
        await asyncio.sleep(min(1.0, end - time.time()))


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

    # long timeout: a whole movie's SRT can go in one request (cleaning) and thinking models take a while
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0)) as client:
        for attempt in range(len(_GEMINI_RETRY_WAITS) + 1):
            check_cancel()
            try:
                resp = await client.post(url, json=payload)
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt >= len(_GEMINI_RETRY_WAITS):
                    raise RuntimeError(f"Gemini API mất kết nối: {e}") from e
                await _sleep_cancellable(_GEMINI_RETRY_WAITS[attempt])
                continue
            if resp.status_code == 400 and "thinkingConfig" in resp.text:
                # Retry without thinkingConfig for non-thinking models
                payload["generationConfig"].pop("thinkingConfig", None)
                resp = await client.post(url, json=payload)
            if resp.status_code in _GEMINI_TRANSIENT and attempt < len(_GEMINI_RETRY_WAITS):
                if resp.status_code == 429 and _is_daily_quota(resp.text):
                    raise RuntimeError(
                        "Gemini API: đã hết hạn mức (quota) MIỄN PHÍ trong ngày của key này. "
                        "Chờ sang ngày mới (giờ Mỹ) rồi bấm Dịch tiếp — tiến trình đã lưu checkpoint. "
                        f"Chi tiết: {resp.text[:200]}"
                    )
                # 503 overloaded / 429 per-minute limit / 500: wait (server hint if any) and try again
                wait = _retry_delay_hint(resp) or _GEMINI_RETRY_WAITS[attempt]
                await _sleep_cancellable(wait)
                continue
            break
        if resp.status_code != 200:
            hint = ""
            if resp.status_code == 503:
                hint = " → Máy chủ Gemini đang quá tải. Thử lại sau ít phút, hoặc đổi sang model khác / giảm số luồng."
            elif resp.status_code == 429:
                hint = " → Vượt giới hạn gọi/phút của key miễn phí. Giảm 'Số luồng dịch song song' xuống 1–2."
            raise RuntimeError(f"Gemini API lỗi HTTP {resp.status_code}{hint}: {resp.text[:300]}")
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
    agentapi_timeout: float = 900.0,
) -> str:
    """Dispatches to Antigravity (Local agentapi with signed-in user account / CLI), Gemini API key or
    Gemini Web (gemini.google.com, sent by the user or by the tool's own Chrome window)."""
    check_cancel()
    if cfg.provider == "gemini_web":
        from app.services import gemini_web

        return await gemini_web.call(cfg, system_prompt, user_prompt)
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
                timeout=agentapi_timeout,
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
            # the input carries a length budget ("<số>|||<N>|||<câu>"): a model that copies it back must not
            # get "9|||" spoken by the voice
            txt = re.sub(r"^\d{1,3}\s*\|\|\|\s*", "", txt).strip()
            if txt:
                parsed[idx] = txt
    return parsed


def _line_budgets(project: VideoProject, cues: List[SubtitleCue]) -> Dict[int, int]:
    from app.services.length_fit import project_budgets

    try:
        return project_budgets(project, cues)
    except Exception:  # noqa: BLE001
        return {}


def _input_line(c: SubtitleCue, budgets: Dict[int, int]) -> str:
    txt = c.text.replace(chr(10), " ")
    return f"{c.index}|||{budgets[c.index]}|||{txt}" if c.index in budgets else f"{c.index}|||{txt}"


def _length_rules(cfg: TranslateConfig, budgets: Dict[int, int]) -> str:
    """Length rules of the translation prompt. With budgets every input line is '<số>|||<N>|||<câu gốc>'."""
    if not budgets:
        return (
            "YÊU CẦU ĐỘ DÀI DÒNG SRT CHO VOICE AI:\n"
            f"- Dịch súc tích, tự nhiên sao cho độ dài câu dịch vừa đủ hoặc chỉ chênh tối đa ±{cfg.length_tolerance_pct}% "
            "so với nhịp nói câu gốc, để khi lồng tiếng/thuyết minh không bị nói quá nhanh hay tràn khung thời gian.\n"
            "- KHÔNG dùng dấu '...' hay '…' để nối câu.\n\n"
        )
    from app.services.length_fit import unit_name

    unit = unit_name(cfg.target_lang)
    return (
        "YÊU CẦU ĐỘ DÀI (BẮT BUỘC - câu dịch sẽ được giọng AI đọc đúng lúc câu gốc xuất hiện trên video):\n"
        f"- Mỗi dòng đầu vào có dạng <số>|||<N>|||<câu gốc>. N = SỐ {unit.upper()} TỐI ĐA của câu dịch, đã tính theo "
        "thời lượng câu đó trên video và tốc độ đọc tự nhiên của giọng đọc. Câu dịch KHÔNG được vượt N.\n"
        "- Tiếng gốc (vd tiếng Trung) rất cô đọng: dịch đủ từng chữ thường dài gấp đôi. Hãy dịch theo kiểu lồng tiếng: "
        "giữ ý chính, tên riêng và xưng hô; bỏ từ đệm, từ lặp, chủ ngữ hiểu ngầm; chọn từ ngắn; được phép tóm ý.\n"
        "- Câu phải tự nhiên, trôi chảy khi đọc lên. KHÔNG dùng dấu '...' hay '…' để nối câu.\n"
        "- ĐẦU RA mỗi dòng CHỈ là <số>|||<bản dịch> (KHÔNG chép lại N).\n\n"
    )


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

    four_items = (
        "1. CHỦ ĐỀ (Topic): Thể loại phim, đề tài chính.\n"
        "2. NGỮ CẢNH (Context): Bối cảnh thời đại, tình huống cốt truyện chính.\n"
        "3. DANH SÁCH NHÂN VẬT & XƯNG HÔ: Tên dịch chuẩn từng nhân vật và cách xưng hô giữa các cặp nhân vật (bắt buộc nhất quán xuyên suốt).\n"
        "4. VĂN PHONG (Style Guide): Giọng điệu (review phim cuốn hút / cổ trang trang trọng / đời thường / hành động...), các thuật ngữ cố định cần giữ nguyên.\n"
    )
    head = (
        "PHÂN TÍCH FILE ĐÍNH KÈM (KHÔNG DỊCH).\n"
        f"- Ngôn ngữ nguồn: {src_name}\n"
        f"- Ngôn ngữ đích: {tgt_name}\n"
        f"- Tên phim/tệp: {project.name}\n\n"
    )

    if cfg.provider == "gemini_web" and llm_caller is invoke_llm:
        # Gemini Web: the WHOLE .srt goes as an ATTACHED FILE and the message is only the short instruction
        # (no subtitle lines pasted into the chat box)
        from app.services import gemini_web

        user_prompt = (
            head
            + f"File phụ đề .srt ĐÍNH KÈM ({Path(project.srt_path).name}, {len(cues)} dòng phụ đề) là toàn bộ lời thoại "
            "của phim. Hãy đọc HẾT file từ đầu đến cuối (không chỉ phần đầu), hiểu toàn bộ nội dung, rồi chốt ra "
            "STYLE GUIDE ngắn gọn, chính xác theo đúng 4 mục:\n" + four_items
        )
        guide_text = await gemini_web.call(cfg, system_prompt, user_prompt, attachment=project.srt_path)
    else:
        # The WHOLE file (was: ~250 sampled lines, a leftover of the 32 K command-line limit). Big inputs go to the
        # Antigravity agent as a file it reads itself (_call_agentapi), or in one HTTPS request.
        text_block = "\n".join(f"{c.index}|||{c.text.replace(chr(10), ' ')}" for c in cues)
        user_prompt = (
            head
            + "Hãy đọc và hiểu toàn bộ nội dung phụ đề dưới đây, chốt ra STYLE GUIDE ngắn gọn, chính xác theo đúng 4 mục:\n"
            + four_items
            + f"\n[TEXT_START]\n{text_block}\n[TEXT_END]"
        )
        guide_text = await llm_caller(cfg, system_prompt, user_prompt)
    stem = re.sub(r'[<>:"/\\|?*]+', "_", Path(project.name).stem)
    out_path = _project_work_dir(project) / f"{stem}_style_guide.md"
    out_path.write_text(guide_text, encoding="utf-8")

    if on_progress:
        on_progress(15.0, f"Đã lưu Style Guide: {out_path.name}")
    return out_path


_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def _looks_untranslated(src: str, out: str, target_lang: str) -> bool:
    """A 'translation' that is still the Chinese source (or still mostly CJK when the target is not CJK)."""
    if target_lang.startswith(("zh", "ja")):
        return False
    o = out.strip()
    if not o:
        return True
    cjk = len(_CJK_RE.findall(o))
    return o == src.strip() or cjk > max(2, len(o) * 0.3)


def _whole_file_agent_available(cfg: TranslateConfig) -> bool:
    has_key = bool((cfg.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip())
    return cfg.provider == "antigravity" and not has_key and get_antigravity_agentapi_path(cfg.agy_path) is not None


def _read_whole_file_parts(out_dir: Path, valid: Dict[int, str], target_lang: str) -> Dict[int, str]:
    got: Dict[int, str] = {}
    for fp in sorted(out_dir.glob("part_*.txt")):
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for idx, line in parse_translated_lines(txt).items():
            if idx in valid and not _looks_untranslated(valid[idx], line, target_lang):
                got[idx] = line
    return got


async def _translate_whole_file(
    project: VideoProject,
    cfg: TranslateConfig,
    cues: List[SubtitleCue],
    translated_map: Dict[int, str],
    cp: Dict[str, Any],
    system_prompt: str,
    style_guide: str,
    src_name: str,
    tgt_name: str,
    on_progress: Optional[ProgressCallback],
) -> None:
    """Whole SRT in ONE Antigravity conversation. A chat answer cannot hold a whole movie (10 000+ lines ~
    200 K tokens > the model's answer limit), so the agent reads the input file with its read tool and writes
    the translation into part files with its write tool (verified with agentapi). We poll the part files for
    progress / checkpoint, and validate every line (index present, not still Chinese). Missing lines are left
    to the caller's normal batch pass, so the final SRT is always complete."""
    if not _whole_file_agent_available(cfg):
        if on_progress:
            on_progress(15.0, "Dịch cả file cần app Antigravity (không dùng API Key) → chuyển sang dịch theo lô...")
        return
    pending = [c for c in cues if c.index not in translated_map]
    work = _project_work_dir(project) / "whole_file_translate"
    if work.exists():
        for old in work.glob("*.txt"):
            try:
                old.unlink()
            except OSError:
                pass
    work.mkdir(parents=True, exist_ok=True)
    in_path = work / "input.txt"
    budgets = _line_budgets(project, cues) if cfg.fit_timing else {}
    in_path.write_text("\n".join(_input_line(c, budgets) for c in pending), encoding="utf-8")
    valid = {c.index: c.text for c in pending}
    per_part = 600
    n_parts = -(-len(pending) // per_part)
    user_prompt = (
        f"NGÔN NGỮ NGUỒN: {src_name}\nNGÔN NGỮ ĐÍCH: {tgt_name}\n\n"
        f"=== STYLE GUIDE ĐÃ CHỐT (TUÂN THỦ TUYỆT ĐỐI) ===\n{style_guide}\n"
        f"================================================\n\n"
        f"{_length_rules(cfg, budgets)}"
        f"NHIỆM VỤ: DỊCH TOÀN BỘ PHỤ ĐỀ CỦA CẢ BỘ PHIM TRONG CUỘC TRÒ CHUYỆN NÀY.\n"
        f"1. Đầu vào: file {in_path} gồm {len(pending)} dòng dạng "
        f"'{'<số>|||<tối đa N>|||<câu gốc>' if budgets else '<số>|||<câu gốc>'}'. Dùng công cụ đọc file đọc "
        f"TOÀN BỘ file (nếu công cụ chỉ trả về một phần thì đọc tiếp cho tới dòng cuối cùng).\n"
        f"2. Đầu ra: ghi bản dịch vào các file {work}\\part_001.txt, part_002.txt, ... (khoảng {n_parts} file), "
        f"mỗi file tối đa {per_part} dòng, theo đúng thứ tự đầu vào. Mỗi dòng đúng định dạng '<số>|||<bản dịch>', "
        f"giữ nguyên <số> của dòng gốc. Ghi xong file này mới sang file tiếp theo.\n"
        f"3. TUYỆT ĐỐI không bỏ sót, không gộp, không tách dòng; không ghi giải thích hay markdown vào file; "
        f"không sửa file đầu vào; không tạo file nào khác.\n"
        f"4. Giữ nhất quán tên nhân vật và xưng hô xuyên suốt cả phim theo Style Guide.\n"
        f"5. Ghi xong toàn bộ thì chỉ trả lời trong chat: DONE <tổng số dòng đã ghi>"
    )
    if on_progress:
        on_progress(15.0, f"Dịch cả file: 1 chat Antigravity cho {len(pending)} dòng (agent tự đọc & ghi file)...")

    stop = asyncio.Event()

    async def _monitor() -> None:
        last = -1
        t0 = time.time()
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            got = _read_whole_file_parts(work, valid, cfg.target_lang)
            if len(got) != last:
                last = len(got)
                for k, v in got.items():
                    translated_map[k] = v
                cp["translated"] = {str(k): v for k, v in translated_map.items()}
                cp["updated_at"] = time.time()
                save_translate_checkpoint(project, cp)
            if on_progress and not is_cancelled():
                el = time.time() - t0
                rate = last / el if el > 0 and last > 0 else 0
                eta = f" · còn ~{int((len(pending) - last) / rate // 60)} phút" if rate > 0 else ""
                on_progress(
                    round(15.0 + (len(translated_map) / len(cues)) * 83.0, 1),
                    f"Dịch cả file (1 chat Antigravity): agent đã ghi {last}/{len(pending)} dòng · "
                    f"đã chạy {int(el // 60)} phút{eta}",
                )

    mon = asyncio.create_task(_monitor())
    try:
        # long timeout: one agent translating a whole movie writes ~200 K tokens
        await invoke_llm(cfg, system_prompt, user_prompt, agentapi_timeout=6 * 3600.0)
    except (JobCancelled, asyncio.CancelledError):
        raise
    except Exception as e:  # noqa: BLE001  (whatever was written is kept; the rest goes to the batch pass)
        if on_progress:
            on_progress(15.0, f"Chat cả file dừng giữa chừng ({str(e)[:80]}) → giữ phần đã dịch, dịch bù phần còn lại...")
    finally:
        stop.set()
        await asyncio.gather(mon, return_exceptions=True)
    got = _read_whole_file_parts(work, valid, cfg.target_lang)
    for k, v in got.items():
        translated_map[k] = v
    cp["translated"] = {str(k): v for k, v in translated_map.items()}
    cp["updated_at"] = time.time()
    save_translate_checkpoint(project, cp)


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
    # lines restored from the checkpoint are not length-checked again (they may have been shortened from their
    # REAL voice length already; re-estimating could rewrite them and force their voice to be made again)
    restored_lines = set(translated_map)
    # up to 2000 lines per request: prompts over the command-line limit go to the agent as a file
    batch_size = max(10, min(2000, cfg.batch_size))
    web = cfg.provider == "gemini_web"
    if web:
        # web chat: a short file goes whole, a long one in equal parts of <= web_chunk_lines, one at a time
        from app.services import gemini_web

        batch_size = gemini_web.chunk_size_for(len(cues), cfg.web_chunk_lines)

    # Build batches of cues (skipping already-translated batches cleanly)
    batches: List[List[SubtitleCue]] = [
        cues[i : i + batch_size] for i in range(0, len(cues), batch_size)
    ]

    n_threads = 1 if web else max(1, min(16, int(getattr(cfg, "concurrency", 1) or 1)))
    sem = asyncio.Semaphore(n_threads)
    pos_of = {c.index: i for i, c in enumerate(cues)}
    budgets = _line_budgets(project, cues) if cfg.fit_timing else {}
    untranslated: set = set()
    done_batches = 0
    active_batches: set = set()

    def _report(extra: str = "") -> None:
        if not on_progress or is_cancelled():
            return
        pct = 15.0 + (len(translated_map) / len(cues)) * 83.0
        running = ", ".join(str(i) for i in sorted(active_batches)) or "-"
        if web:
            mode = "tự động" if cfg.web_mode == "auto" else "gửi tay — xem bảng Gemini Web"
            on_progress(
                round(pct, 1),
                f"Pha 2/2: Gemini Web ({mode}) — xong {done_batches}/{len(batches)} phần "
                f"({len(translated_map)}/{len(cues)} dòng), đang làm phần {running}{extra}",
            )
            return
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
                _report(f" (lô {b_idx} lỗi, thử lại {n + 1}/3: {str(e)[:80]})")
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
            if web:
                from app.services import gemini_web

                gemini_web.part_label.set(
                    f"Phần {b_idx}/{len(batches)}" + (f" · dịch bù dòng thiếu (lần {attempt - 1})" if attempt > 1 else "")
                )

            lines_payload = "\n".join(_input_line(c, budgets) for c in remaining)
            user_prompt = (
                f"NGÔN NGỮ NGUỒN: {src_name}\n"
                f"NGÔN NGỮ ĐÍCH: {tgt_name}\n\n"
                f"=== STYLE GUIDE ĐÃ CHỐT (TUÂN THỦ TUYỆT ĐỐI) ===\n"
                f"{style_guide}\n"
                f"================================================\n\n"
                f"{_length_rules(cfg, budgets)}"
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

        # Fallback: if any single stubborn line still missed after 3 tries, keep original text rather than dropping
        # cue - and count it, so the user is told (it used to be silent: untranslated lines in the final video).
        for c in remaining:
            translated_map[c.index] = c.text
            untranslated.add(c.index)

    # ---- Whole-file mode: one Antigravity conversation for every line still to translate ----
    if cfg.whole_file and not web and llm_caller is invoke_llm and any(c.index not in translated_map for c in cues):
        await _translate_whole_file(
            project, cfg, cues, translated_map, cp, system_prompt, style_guide, src_name, tgt_name, on_progress
        )
        missing = [c for c in cues if c.index not in translated_map]
        if missing and on_progress:
            on_progress(
                round(15.0 + (len(translated_map) / len(cues)) * 83.0, 1),
                f"Chat cả file thiếu {len(missing)} dòng → tự dịch bù {len(missing)} dòng theo lô ({n_threads} luồng)...",
            )

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

    fit_txt = ""
    if cfg.fit_timing:
        # the model does not always keep to the budget: rewrite the lines that are still clearly too long
        # (once per line: lines already shortened are remembered in the checkpoint)
        from app.services.length_fit import fit_project_lines

        prev_srt = project.translated_srt_path
        project.translated_srt_path = str(out_srt)
        try:
            fit = await fit_project_lines(
                project,
                on_progress=(lambda p, m: on_progress(99.0, f"[Độ dài] {m}")) if on_progress else None,
                llm_caller=llm_caller,
                skip=set(int(k) for k in cp.get("fitted", [])) | restored_lines,
            )
        finally:
            project.translated_srt_path = prev_srt
        if fit["over"]:
            cp = load_translate_checkpoint(project) or cp
            cp["fitted"] = sorted(set(int(k) for k in cp.get("fitted", [])) | set(fit["targets"]))
            save_translate_checkpoint(project, cp)
            fit_txt = f" · rút gọn {len(fit['changed'])}/{fit['over']} câu dài quá khung thời gian"

    if on_progress:
        warn = f" ⚠ {len(untranslated)} dòng AI không trả về sau 3 lần thử, giữ nguyên chữ gốc." if untranslated else ""
        on_progress(100.0, f"Dịch xong {len(out_cues)} dòng{fit_txt}! Đã lưu: {out_srt.name}{warn}")
    return out_srt
