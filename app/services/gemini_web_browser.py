"""Drives gemini.google.com in the tool's OWN Chrome window (auto mode of the "Gemini Web" translator).

How: Chrome (or Edge) is started with its own profile folder (data/gemini_web_profile) and a DevTools port on
127.0.0.1. The user signs in to Google once in that window; the tool then talks to the page over the Chrome
DevTools Protocol (plain websocket, no Playwright / Selenium): it opens a new chat, types the prompt, presses
Send, waits until the answer stops changing and reads its text.

Gemini's page changes from time to time. All the CSS selectors used are in SELECTORS and can be overridden
without touching code in data/gemini_web_selectors.json (same keys, each a list of selectors tried in order).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from app.config import DATA_DIR
from app.services.cancel import check_cancel

GEMINI_URL = "https://gemini.google.com/app"
GEMINI_HOST = "gemini.google.com"  # how the Gemini tab is recognised
CDP_PORT = 9333
PROFILE_DIR = DATA_DIR / "gemini_web_profile"
SELECTORS_FILE = DATA_DIR / "gemini_web_selectors.json"

SELECTORS: Dict[str, List[str]] = {
    # the prompt box (a Quill rich-text editor)
    "editor": [
        "rich-textarea .ql-editor[contenteditable='true']",
        "div.ql-editor[contenteditable='true']",
        "div[contenteditable='true'][role='textbox']",
    ],
    "send": [
        "button.send-button:not(.stop)",
        "button[aria-label*='Send' i]",
        "button[aria-label*='Gửi']",
    ],
    # present while an answer is being written
    "stop": [
        "button.send-button.stop",
        "button[aria-label*='Stop' i]",
        "button[aria-label*='Dừng']",
    ],
    # the chip of an attached file in the prompt box
    "attachment": [
        "gem-attachment",  # verified on gemini.google.com 28/11/2026
        "uploader-file-preview",
        "uploader-file-preview-container",
        "[data-test-id='file-preview']",
        ".file-preview",
        ".attachment-preview",
    ],
    # an attached file still uploading (only looked for INSIDE the attachment chips)
    "uploading": ["[role='progressbar']", "mat-progress-spinner", "mat-spinner", ".loading", ".uploading"],
    # one element per answer of the model, in order (thoughts are outside message-content)
    "response": [
        "model-response message-content",
        "message-content.model-response-text",
        "model-response .markdown",
    ],
}


def selectors() -> Dict[str, List[str]]:
    sel = {k: list(v) for k, v in SELECTORS.items()}
    try:
        if SELECTORS_FILE.exists():
            user = json.loads(SELECTORS_FILE.read_text(encoding="utf-8"))
            for k, v in user.items():
                if k in sel and isinstance(v, list) and v:
                    sel[k] = [str(x) for x in v]
    except Exception:  # noqa: BLE001  (a broken override file must not stop the tool)
        pass
    return sel


# ---------------------------------------------------------------------------------------------------------------
# Browser process
# ---------------------------------------------------------------------------------------------------------------
def find_browser() -> Optional[str]:
    local = os.environ.get("LOCALAPPDATA", "")
    cands = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.join(local, r"Google\Chrome\Application\chrome.exe") if local else "",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in cands:
        if c and Path(c).exists():
            return c
    return shutil.which("chrome") or shutil.which("msedge")


def port_open(timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", CDP_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def launch_browser() -> Dict[str, Any]:
    """Opens the tool's Gemini window (own profile + DevTools port). Already open -> nothing to do."""
    if port_open():
        return {"ok": True, "already": True}
    exe = find_browser()
    if not exe:
        raise RuntimeError("Không tìm thấy Google Chrome hoặc Microsoft Edge trên máy.")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    # Own --user-data-dir: Chrome refuses the DevTools port on the normal profile, and the user's everyday
    # browser stays untouched. The port is bound to 127.0.0.1 only.
    subprocess.Popen(
        [
            exe,
            f"--remote-debugging-port={CDP_PORT}",
            "--remote-debugging-address=127.0.0.1",
            f"--user-data-dir={PROFILE_DIR}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            GEMINI_URL,
        ],
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        close_fds=True,
    )
    for _ in range(40):
        if port_open():
            return {"ok": True, "already": False, "browser": Path(exe).name}
        time.sleep(0.25)
    raise RuntimeError(
        "Đã mở trình duyệt nhưng không kết nối được cổng điều khiển. Nếu cửa sổ Chrome của tool đang mở sẵn từ "
        "trước (không có cổng), hãy đóng nó rồi bấm lại."
    )


# ---------------------------------------------------------------------------------------------------------------
# DevTools protocol
# ---------------------------------------------------------------------------------------------------------------
async def _targets() -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(f"http://127.0.0.1:{CDP_PORT}/json/list")
        r.raise_for_status()
        return r.json()


async def _gemini_target(create: bool = True) -> Optional[Dict[str, Any]]:
    pages = [t for t in await _targets() if t.get("type") == "page"]
    for t in pages:
        if GEMINI_HOST in (t.get("url") or ""):
            return t
    if not create:
        return None
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.put(f"http://127.0.0.1:{CDP_PORT}/json/new?{GEMINI_URL}")
        r.raise_for_status()
        return r.json()


class _Cdp:
    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url
        self.ws = None
        self._id = 0

    async def __aenter__(self) -> "_Cdp":
        from websockets.asyncio.client import connect

        self.ws = await connect(self.ws_url, max_size=None, open_timeout=15, ping_interval=None)
        return self

    async def __aexit__(self, *exc) -> None:
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"Chrome không phản hồi lệnh {method}")
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), left))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"Chrome lỗi {method}: {msg['error'].get('message')}")
                return msg.get("result") or {}
            # other messages are events: ignored

    async def js(self, expr: str, timeout: float = 30.0) -> Any:
        r = await self.call(
            "Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}, timeout
        )
        if r.get("exceptionDetails"):
            raise RuntimeError(f"Lỗi JS trên trang Gemini: {str(r['exceptionDetails'])[:200]}")
        return (r.get("result") or {}).get("value")


def _probe_js(sel: Dict[str, List[str]]) -> str:
    return (
        "(() => {"
        f"const S = {json.dumps(sel)};"
        "const all = (k) => { for (const s of S[k]) { const e = document.querySelectorAll(s); if (e.length) return [...e]; } return []; };"
        "const vis = (e) => !!(e && (e.offsetWidth || e.offsetHeight || e.getClientRects().length));"
        "const ed = all('editor')[0]; const send = all('send').filter(vis)[0]; const stop = all('stop').filter(vis)[0];"
        "const resp = all('response'); const last = resp.length ? resp[resp.length - 1] : null;"
        "return { url: location.href, editor: !!ed, editor_len: ed ? (ed.innerText || '').trim().length : 0,"
        " send: !!send, send_disabled: send ? (send.disabled || send.getAttribute('aria-disabled') === 'true') : true,"
        " busy: !!stop, n_resp: resp.length, last: last ? last.innerText : '' };"
        "})()"
    )


def _focus_clear_js(sel: Dict[str, List[str]]) -> str:
    return (
        "(() => {"
        f"const S = {json.dumps(sel['editor'])};"
        "let ed = null; for (const s of S) { ed = document.querySelector(s); if (ed) break; }"
        "if (!ed) return false; ed.focus();"
        "document.execCommand('selectAll', false, null); document.execCommand('delete', false, null);"
        "return true; })()"
    )


def _insert_fallback_js(sel: Dict[str, List[str]], text: str) -> str:
    return (
        "(() => {"
        f"const S = {json.dumps(sel['editor'])}; const T = {json.dumps(text)};"
        "let ed = null; for (const s of S) { ed = document.querySelector(s); if (ed) break; }"
        "if (!ed) return 0; ed.focus();"
        "if (!document.execCommand('insertText', false, T)) {"
        "  ed.innerText = T; ed.dispatchEvent(new InputEvent('input', {bubbles: true}));"
        "}"
        "return (ed.innerText || '').length; })()"
    )


def _click_send_js(sel: Dict[str, List[str]]) -> str:
    return (
        "(() => {"
        f"const S = {json.dumps(sel['send'])};"
        "for (const s of S) { const b = [...document.querySelectorAll(s)].find(e => !e.disabled && e.getAttribute('aria-disabled') !== 'true');"
        " if (b) { b.click(); return true; } }"
        "return false; })()"
    )


def _attach_state_js(sel: Dict[str, List[str]], name: str) -> str:
    return (
        "(() => {"
        f"const A = {json.dumps(sel['attachment'])}; const U = {json.dumps(sel['uploading'])}; const N = {json.dumps(name)};"
        "let chips = []; for (const s of A) { chips = [...document.querySelectorAll(s)]; if (chips.length) break; }"
        "const uploading = chips.some(c => U.some(u => c.querySelector(u)));"
        "return { chips: chips.length, uploading, name_hits: (document.body.innerText.split(N).length - 1) }; })()"
    )


def _synth_file_js(sel: Dict[str, List[str]], name: str, content: str, how: str) -> str:
    # paste = like Ctrl+V of a copied file into the prompt box; drop = like dragging the file onto it
    return (
        "(() => {"
        f"const S = {json.dumps(sel['editor'])}; const NAME = {json.dumps(name)}; const C = {json.dumps(content)};"
        "let ed = null; for (const s of S) { ed = document.querySelector(s); if (ed) break; }"
        "if (!ed) return false;"
        "const dt = new DataTransfer(); dt.items.add(new File([C], NAME, {type: 'text/plain'}));"
        f"if ({json.dumps(how)} === 'paste') {{ ed.focus();"
        "  ed.dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt, bubbles: true, cancelable: true})); }"
        "else { for (const t of ['dragenter', 'dragover', 'drop'])"
        "  ed.dispatchEvent(new DragEvent(t, {dataTransfer: dt, bubbles: true, cancelable: true})); }"
        "return true; })()"
    )


async def _wait_attached(cdp: "_Cdp", sel: Dict[str, List[str]], name: str, before: Dict[str, Any], wait: float) -> bool:
    """True once a new attachment chip (or the file name) shows up and nothing is still uploading."""
    end = time.monotonic() + wait
    seen = False
    while time.monotonic() < end:
        check_cancel()
        await asyncio.sleep(0.7)
        st = await cdp.js(_attach_state_js(sel, name))
        if st["chips"] > before["chips"] or st["name_hits"] > before["name_hits"]:
            if not seen:  # once attached, give the upload itself up to 3 more minutes
                seen = True
                end = max(end, time.monotonic() + 180)
            if not st["uploading"]:
                await asyncio.sleep(1.5)
                return True
    return seen


async def _attach_file(cdp: "_Cdp", path: str, sel: Dict[str, List[str]], say: Callable[[str], None]) -> None:
    """Attaches a local file to the Gemini prompt box (before the text is typed)."""
    fp = Path(path).resolve()
    name = fp.name
    say(f"Đang đính kèm file {name} vào Gemini...")
    before = await cdp.js(_attach_state_js(sel, name))
    # 1. a real <input type=file> on the page (accepting text files)
    try:
        doc = await cdp.call("DOM.getDocument", {"depth": 0})
        r = await cdp.call("DOM.querySelectorAll", {"nodeId": doc["root"]["nodeId"], "selector": "input[type=file]"})
        for nid in r.get("nodeIds", []):
            attrs = (await cdp.call("DOM.getAttributes", {"nodeId": nid})).get("attributes", [])
            accept = dict(zip(attrs[::2], attrs[1::2])).get("accept", "").lower()
            if accept and not any(k in accept for k in ("text", "*", ".srt", ".txt")):
                continue
            await cdp.call("DOM.setFileInputFiles", {"nodeId": nid, "files": [str(fp)]})
            if await _wait_attached(cdp, sel, name, before, 15):
                return
            break
    except Exception:  # noqa: BLE001  (no usable input: try paste / drop)
        pass
    content = fp.read_text(encoding="utf-8-sig", errors="replace")
    for how in ("paste", "drop"):
        await cdp.js(_synth_file_js(sel, name, content, how), timeout=120)
        if await _wait_attached(cdp, sel, name, before, 15):
            return
    raise RuntimeError(
        f"Không đính kèm được file {name} vào Gemini (Gemini đổi giao diện?) → làm tay phần này: tải file, kéo vào Gemini."
    )


async def status() -> Dict[str, Any]:
    """For the 'Kiểm tra' button: is the window open, is Gemini loaded, is the user signed in."""
    out: Dict[str, Any] = {"browser": find_browser(), "port_open": port_open(), "gemini_tab": False, "ready": False}
    if not out["port_open"]:
        out["message"] = "Chưa mở cửa sổ Chrome Gemini của tool (bấm 'Mở Chrome Gemini')."
        return out
    try:
        t = await _gemini_target(create=False)
        if not t:
            out["message"] = "Cửa sổ Chrome của tool đang mở nhưng chưa có tab Gemini (bấm 'Mở Chrome Gemini')."
            return out
        out["gemini_tab"] = True
        async with _Cdp(t["webSocketDebuggerUrl"]) as cdp:
            st = await cdp.js(_probe_js(selectors()))
        out.update({k: st.get(k) for k in ("url", "editor", "n_resp")})
        if "accounts.google.com" in (st.get("url") or "") or not st.get("editor"):
            out["message"] = "Chưa thấy ô nhập của Gemini: hãy đăng nhập Google trong cửa sổ Chrome của tool."
        else:
            out["ready"] = True
            out["message"] = "Sẵn sàng: Gemini đã mở và đã đăng nhập."
    except Exception as e:  # noqa: BLE001
        out["message"] = f"Lỗi kết nối Chrome: {e}"
    return out


async def _sleep(sec: float) -> None:
    end = time.monotonic() + sec
    while time.monotonic() < end:
        check_cancel()
        await asyncio.sleep(min(0.5, end - time.monotonic()))


async def run_prompt(
    text: str,
    new_chat: bool = True,
    timeout: float = 900.0,
    on_status: Optional[Callable[[str], None]] = None,
    attachment: str = "",
) -> str:
    """Sends one prompt (optionally with an attached file) in the tool's Gemini window; returns the answer text."""
    say = on_status or (lambda m: None)
    if not port_open():
        say("Đang mở cửa sổ Chrome Gemini của tool...")
        await asyncio.to_thread(launch_browser)
    sel = selectors()
    t = await _gemini_target(create=True)
    async with _Cdp(t["webSocketDebuggerUrl"]) as cdp:
        if new_chat or GEMINI_HOST not in (t.get("url") or ""):
            await cdp.call("Page.navigate", {"url": GEMINI_URL})
            await _sleep(2.0)
        # wait for the prompt box (or the sign-in page)
        st: Dict[str, Any] = {}
        for _ in range(60):
            check_cancel()
            try:
                st = await cdp.js(_probe_js(sel))
            except Exception:  # noqa: BLE001  (page still loading)
                st = {}
            if st.get("editor"):
                break
            if "accounts.google.com" in (st.get("url") or ""):
                raise RuntimeError("Gemini chưa đăng nhập: hãy đăng nhập Google trong cửa sổ Chrome của tool.")
            await _sleep(0.5)
        if not st.get("editor"):
            raise RuntimeError(
                "Không tìm thấy ô nhập của Gemini (chưa đăng nhập, hoặc Gemini đổi giao diện → sửa "
                f"{SELECTORS_FILE.name})."
            )
        n_before = int(st.get("n_resp") or 0)

        if attachment:
            await _attach_file(cdp, attachment, sel, say)
        say("Đang dán prompt vào Gemini...")
        await cdp.js(_focus_clear_js(sel))
        await cdp.call("Input.insertText", {"text": text}, timeout=180)
        st = await cdp.js(_probe_js(sel))
        if int(st.get("editor_len") or 0) < min(20, len(text.strip())):
            await cdp.js(_insert_fallback_js(sel, text), timeout=180)

        # the send button turns active once the editor has text (a long paste can take a moment)
        sent = False
        for _ in range(240 if attachment else 40):
            check_cancel()
            if await cdp.js(_click_send_js(sel)):
                sent = True
                break
            await _sleep(0.5)
        if not sent:
            for typ in ("keyDown", "keyUp"):
                await cdp.call(
                    "Input.dispatchKeyEvent",
                    {"type": typ, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13},
                )

        # wait until a new answer exists and is finished: no stop button and the text stopped changing
        say("Đã gửi, đang chờ Gemini trả lời...")
        t0 = time.monotonic()
        started = False
        last_text = ""
        stable = 0
        while True:
            check_cancel()
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"Gemini không trả lời xong sau {int(timeout)} giây.")
            await _sleep(2.0)
            try:
                st = await cdp.js(_probe_js(sel))
            except Exception:  # noqa: BLE001
                continue
            n = int(st.get("n_resp") or 0)
            if n <= n_before and not st.get("busy"):
                if not started and time.monotonic() - t0 > 60:
                    raise RuntimeError("Đã bấm gửi nhưng Gemini không bắt đầu trả lời (bị chặn / giới hạn?).")
                continue
            started = True
            cur = (st.get("last") or "").strip() if n > n_before else ""
            if st.get("busy") or not cur:
                stable = 0
                last_text = cur
                if cur:
                    say(f"Gemini đang viết... ({len(cur)} ký tự)")
                continue
            if cur == last_text:
                stable += 1
                if stable >= 2:  # unchanged for ~4 s with no stop button -> done
                    return cur
            else:
                stable = 0
                last_text = cur
