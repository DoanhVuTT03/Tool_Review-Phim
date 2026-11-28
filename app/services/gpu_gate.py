"""GPU gate between Higgs TTS 3 and the other GPU work of the tool.

The Higgs server takes ~11-12 GB of the 12 GB card, so while it runs nothing else may use the GPU (OCR
pre-scan, AI Clean Sub inpaint, render). Higgs voice jobs hold the gate EXCLUSIVELY; OCR / render jobs hold it
SHARED (any number together, as before). A shared holder also stops a Higgs server that was left running (after
the review table's "Tạo lại"), so its VRAM is free before OCR starts.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator, Optional


def release_tool_gpu() -> None:
    """Frees the GPU memory this process keeps between renders (OCR detectors, inpainter, LaMa: cached for speed).
    Found 25/11/2026: after an AI Clean Sub render the server process still held ~1.5 GB, so the Higgs server
    (needs ~10.3 GB of the 12 GB) could not start. Only called with the gate held exclusively (no OCR / render
    running); they are rebuilt on the next render (~2 s)."""
    import gc
    import sys

    try:
        from app.services import sub_remover

        with sub_remover._DETECTORS_LOCK:
            for d in sub_remover._DETECTORS or []:
                close = getattr(d, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            sub_remover._DETECTORS = None
        sub_remover._LAMA_SESSION = None
    except Exception:
        pass
    for mod in ("app.services.gpu_ocr", "app.services.gpu_inpaint"):
        m = sys.modules.get(mod)
        if m is not None:
            with getattr(m, "_LOCK", contextlib.nullcontext()):
                m._INST = None
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


class _Gate:
    def __init__(self) -> None:
        self._cond: Optional[asyncio.Condition] = None
        self._shared = 0
        self._exclusive = False
        self._waiting_exclusive = 0

    def _c(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    @contextlib.asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        c = self._c()
        async with c:
            self._waiting_exclusive += 1
            try:
                await c.wait_for(lambda: not self._exclusive and self._shared == 0)
            finally:
                self._waiting_exclusive -= 1
            self._exclusive = True
        try:
            await asyncio.to_thread(release_tool_gpu)
            yield
        finally:
            async with c:
                self._exclusive = False
                c.notify_all()

    @contextlib.asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        c = self._c()
        async with c:
            # a waiting Higgs job goes first (otherwise a stream of renders could starve it)
            await c.wait_for(lambda: not self._exclusive and self._waiting_exclusive == 0)
            self._shared += 1
        try:
            from app.services import higgs_tts

            await asyncio.to_thread(higgs_tts.stop_if_running)
            yield
        finally:
            async with c:
                self._shared -= 1
                c.notify_all()

    def busy_text(self) -> str:
        if self._exclusive:
            return "Higgs TTS 3 đang dùng GPU"
        if self._shared:
            return f"{self._shared} việc OCR / xuất video đang dùng GPU"
        return ""

    @property
    def exclusive_held(self) -> bool:
        return self._exclusive


gpu_gate = _Gate()
