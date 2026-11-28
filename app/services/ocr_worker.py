"""GPU text detector (PP-OCR det on DirectML) running in its own process.

Why a process: on a strong GPU the AI Clean Sub pipeline runs at 300-450 fps with 8 inpaint threads + 2 ffmpeg
processes saturating the CPU. Inside that busy server process each DirectML Run() of the detector took ~100 ms
(6-9 ms when idle) and the render waited for OCR most of the time. The same detector in a separate process,
during the same render, measured ~24 ms. The process is started once and reused by every render.

Protocol over stdin/stdout (binary):
    parent -> child: struct "<II" (h, w) + h*w*3 bytes BGR strip
    child  -> parent: struct "<I" n + n bytes JSON: null (no text) | [[[x, y] x4], ...] | {"error": "..."}
The child exits when its stdin is closed (server stopped / killed).
"""
from __future__ import annotations

import json
import struct
import sys


def _read_exact(stream, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def main() -> None:
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00008000)  # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:
        pass

    import numpy as np
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    from app.services.sub_remover import _TextDetector

    inp, out = sys.stdin.buffer, sys.stdout.buffer
    det = _TextDetector("dml", device_id)
    while True:
        hdr = _read_exact(inp, 8)
        if hdr is None:
            break
        h, w = struct.unpack("<II", hdr)
        raw = _read_exact(inp, h * w * 3)
        if raw is None:
            break
        try:
            boxes = det.detect(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3))
            payload = None if boxes is None else boxes.tolist()
        except Exception as e:  # noqa: BLE001
            payload = {"error": f"{type(e).__name__}: {e}"[:500]}
        data = json.dumps(payload, separators=(",", ":")).encode()
        out.write(struct.pack("<I", len(data)) + data)
        out.flush()


if __name__ == "__main__":
    main()
