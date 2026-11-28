"""AI Clean Sub: removes burned-in subtitles inside a user-drawn ROI strip.

Speed design (measured on i7-1065G7 + Iris Plus + MX230, 1080p VP9):
- Text detection calls the PP-OCR *detector only* directly at the ROI's own aspect ratio (RapidOCR's wrapper
  letterboxes the strip to 416x1600 and also runs recognition: ~1050 ms -> ~60-85 ms per sample, fewer false
  positives), spread over CPU + every DirectML GPU (Iris / MX230).
- Single pass (AICleanStream): a light ffmpeg decode streams only the ROI strips to Python; OCR runs on the fly
  on every Nth strip with a small look-ahead; worker threads inpaint strips containing text; the cleaned strips
  are piped straight into the FINAL render ffmpeg, which overlays them and encodes once. No intermediate
  "clean" video, no second encode, Python never touches full 1080p frames. RAM is used as bounded queues /
  look-ahead buffers so decode, OCR, inpaint and encode all overlap.

Strong-GPU tuning (measured on i5-12400F 6C/12T + RTX 3060 12 GB, 1080p, ~203 -> ~236 fps, ~270 with pre-scan):
- inpaint pool sized from CPU threads (was fixed at 4 = the bottleneck), NVENC p2, overlay format=yuv420;
- GPU OCR in separate processes (_ProcDetector): in-process DirectML Run() took ~100 ms under a full render;
- A-Z pipeline pre-scans OCR keys (AICleanStream.prescan) while translation / TTS wait on the network.
"""
from __future__ import annotations

import contextvars
import hashlib
import itertools
import json
import queue
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import MODELS_DIR, PROJECTS_DIR, get_ffmpeg_path
from app.models import BoxRect, SubRemovalConfig, VideoProject
from app.services.cancel import check_cancel, is_cancelled, register_proc, unregister_proc
from app.services.hardware import detect_hardware


ProgressCallback = Callable[[float, str], None]

_LAMA_SESSION = None
_DETECTORS: Optional[List["_TextDetector"]] = None
_DETECTORS_LOCK = threading.Lock()
# Detector input: longest side scaled to <= this. 2000 = never downscale a 1080p strip, i.e. exactly what
# RapidOCR does (its get_preprocess raises the limit to 1500/2000). Smaller sizes were faster but missed
# 3-5 of 195 text samples in tests, so full resolution is kept.
_DET_LIMIT_SIDE = 2000
_DET_MIN_BOX_SCORE = 0.5
_SCAN_QUEUE_FRAMES = 256        # RAM buffer between decoder and detectors (~170 MB for a 1600x140 strip)
_CLEAN_CHUNK = 48               # frames inpainted per parallel chunk in pass 2
_REF_WINDOW_SEC = 1.5           # temporal propagation: max distance to a clean reference frame
_GPU_OCR_PROCS = 3              # detector processes per strong (>= 4 GB) GPU, see _ProcDetector


# ----------------------------------------------------------------------------------------------------
# Text detector pool (CPU + DirectML GPUs)
# ----------------------------------------------------------------------------------------------------


class _TextDetector:
    """PP-OCR text detector (no recognition) on one device."""

    def __init__(self, device: str, dml_device_id: int = 0, label: str = "") -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.device = device
        self.label = label or device
        import onnxruntime as ort

        ocr = RapidOCR(det_limit_type="max", det_limit_side_len=_DET_LIMIT_SIDE, det_box_thresh=_DET_MIN_BOX_SCORE)
        self._det = ocr.text_det
        model_path = self._det.infer.session._model_path
        so = ort.SessionOptions()
        # ORT's default worker threads busy-spin between calls and burn every core even while the GPU does
        # the work -> they starved the render's decoder/encoder. No spinning; GPU sessions need 1 CPU thread.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if device == "dml":
            so.intra_op_num_threads = 1
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [("DmlExecutionProvider", {"device_id": dml_device_id}), "CPUExecutionProvider"]
        else:
            so.intra_op_num_threads = 4
            providers = ["CPUExecutionProvider"]
        self._det.infer.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        if device == "dml" and self._det.infer.session.get_providers()[0] != "DmlExecutionProvider":
            raise RuntimeError("DirectML not active")
        self.count = 0
        self.vram_mb = 0
        # Detectors are cached and shared: the A-Z batch runs up to 2 AI renders at once, and a DirectML
        # session crashes when Run() is called from two threads concurrently.
        self._lock = threading.Lock()

    def detect(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns int32 polygons (N,4,2) in ROI coordinates, or None when no text.

        Own pre-processing instead of RapidOCR's: its get_preprocess() silently raises the size limit to 1500
        (so the strip was never downscaled) and normalises in float64 numpy (~30 ms CPU per call, which also
        stole CPU from the render while the GPUs did the inference). blobFromImage computes exactly the same
        (x/255 - 0.5)/0.5 = (x - 127.5)/127.5, in float32, in C++.
        """
        h, w = roi_bgr.shape[:2]
        ratio = min(1.0, _DET_LIMIT_SIDE / max(h, w))
        dh = max(32, int(round(h * ratio / 32)) * 32)
        dw = max(32, int(round(w * ratio / 32)) * 32)
        img = cv2.resize(roi_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR) if (dw, dh) != (w, h) else roi_bgr
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 127.5, mean=(127.5, 127.5, 127.5), swapRB=False)
        with self._lock:
            preds = self._det.infer(blob)[0]
        boxes, _ = self._det.postprocess_op(preds, (h, w))
        boxes = self._det.filter_tag_det_res(boxes, (h, w))
        self.count += 1
        if boxes is None or len(boxes) == 0:
            return None
        return np.asarray(boxes, dtype=np.int32)


class _ChunkItem:
    """Future-like view on one frame of a GPU chunk result (the writer only calls .result())."""

    __slots__ = ("_f", "_k")

    def __init__(self, fut, k: int) -> None:
        self._f, self._k = fut, k

    def result(self):
        return self._f.result()[self._k]


class _ProcDetector:
    """Same interface as _TextDetector, but the DirectML detector lives in its own process
    (app/services/ocr_worker.py), isolated from the busy render process. Used for strong GPUs only:
    in-process Run() took ~100 ms under a full render vs ~24 ms in a separate process (RTX 3060)."""

    device = "dml"

    def __init__(self, device_id: int, label: str, vram_mb: int) -> None:
        import struct
        import sys

        from app.config import BASE_DIR

        self._struct = struct
        self.label = label
        self.vram_mb = vram_mb
        self.count = 0
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "app.services.ocr_worker", str(device_id)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(BASE_DIR),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def detect(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = roi_bgr.shape[:2]
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError("tiến trình OCR GPU đã dừng")
            self.proc.stdin.write(self._struct.pack("<II", h, w))
            self.proc.stdin.write(np.ascontiguousarray(roi_bgr).data)
            self.proc.stdin.flush()
            hdr = _read_exact(self.proc.stdout, 4)
            data = _read_exact(self.proc.stdout, self._struct.unpack("<I", hdr)[0]) if hdr else None
        if data is None:
            raise RuntimeError("tiến trình OCR GPU không phản hồi")
        res = json.loads(data)
        if isinstance(res, dict):
            raise RuntimeError(res.get("error", "OCR GPU lỗi"))
        self.count += 1
        return None if res is None else np.asarray(res, dtype=np.int32)

    def close(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass


def _dxgi_adapters() -> List[Dict[str, Any]]:
    """Physical GPUs in DXGI order (index = DirectML device_id). Software adapters are skipped and the same
    card listed twice (e.g. MuMuPlayer / virtual-display drivers add a 2nd LUID for one RTX) is kept once.
    Windows only; [] on failure."""
    try:
        import ctypes
        from ctypes import POINTER, byref, c_void_p

        class GUID(ctypes.Structure):
            _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort), ("d3", ctypes.c_ushort), ("d4", ctypes.c_ubyte * 8)]

        class DESC1(ctypes.Structure):
            _fields_ = [
                ("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", ctypes.c_int64), ("Flags", ctypes.c_uint),
            ]

        iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
        factory = c_void_p()
        if ctypes.windll.dxgi.CreateDXGIFactory1(byref(iid), byref(factory)) != 0:
            return []

        def _method(obj, idx, *argtypes):
            vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
            return ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, *argtypes)(vtbl[idx])

        out: List[Dict[str, Any]] = []
        seen = set()
        i = 0
        while True:
            adapter = c_void_p()
            if _method(factory, 12, ctypes.c_uint, POINTER(c_void_p))(factory, i, byref(adapter)) != 0:  # EnumAdapters1
                break
            desc = DESC1()
            _method(adapter, 10, POINTER(DESC1))(adapter, byref(desc))  # GetDesc1
            _method(adapter, 2)(adapter)  # Release
            vram_mb = int(desc.DedicatedVideoMemory) >> 20
            ident = (desc.VendorId, desc.DeviceId, desc.SubSysId, desc.Revision, vram_mb)
            if not desc.Flags & 2 and ident not in seen:  # DXGI_ADAPTER_FLAG_SOFTWARE
                seen.add(ident)
                out.append({"device_id": i, "name": desc.Description, "vram_mb": vram_mb})
            i += 1
        _method(factory, 2)(factory)
        return out
    except Exception:
        return []


def _short_gpu_name(name: str) -> str:
    low = name.lower()
    if "iris" in low:
        return "Iris"
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "Intel(R) "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _set_thread_priority(level: int) -> None:
    """Windows thread priority: -2 LOWEST, 0 NORMAL, 2 HIGHEST."""
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.SetThreadPriority(k32.GetCurrentThread(), level)
    except Exception:
        pass


def _lower_thread_priority() -> None:
    _set_thread_priority(-2)  # THREAD_PRIORITY_LOWEST


def _get_detectors() -> List[_TextDetector]:
    """CPU detector + one DirectML session per physical GPU that works. Built once and cached.

    Exactly one session per GPU: two sessions on the same DirectML device run from two threads crash
    (DmlCommandRecorder 0x80004005 / segfault), and batching several strips per Run() is not faster on DML.
    One RTX 3060 session does ~6-10 ms per sample, i.e. far more samples/s than a 450 fps render needs."""
    global _DETECTORS
    with _DETECTORS_LOCK:
        if _DETECTORS is not None:
            return _DETECTORS
        dets: List[Any] = [_TextDetector("cpu", label="CPU")]
        # 1st choice on NVIDIA >= 4 GB with torch CUDA: batched detector in this process's CUDA context
        # (gpu_ocr.py). Then the NVIDIA card needs no DirectML processes.
        cuda_det = None
        try:
            from app.services.gpu_ocr import get_torch_detector

            cuda_det = get_torch_detector(_DET_LIMIT_SIDE, _DET_MIN_BOX_SCORE)
        except Exception:
            cuda_det = None
        if cuda_det is not None:
            dets.append(cuda_det)
        try:
            import onnxruntime as ort

            if "DmlExecutionProvider" in ort.get_available_providers():
                adapters = _dxgi_adapters() or [
                    {"device_id": i, "name": n, "vram_mb": 0}
                    for i, n in enumerate(detect_hardware().get("gpus") or [])
                    if n and "basic render" not in n.lower()
                ]
                probe = np.zeros((96, 960, 3), dtype=np.uint8)
                for ad in adapters:
                    label = _short_gpu_name(ad["name"])
                    d: Any = None
                    if cuda_det is not None and ad["vram_mb"] >= 4000:
                        continue  # this card already runs the CUDA detector
                    if ad["vram_mb"] >= 4000:  # strong GPU: detectors in their own processes (see _ProcDetector)
                        procs = [_ProcDetector(ad["device_id"], f"{label}#{i + 1}", ad["vram_mb"])
                                 for i in range(_GPU_OCR_PROCS)]
                        for p in procs:  # started together, so the ~2 s model loads overlap
                            try:
                                p.detect(probe)
                                dets.append(p)
                            except Exception:
                                p.close()
                        if any(isinstance(x, _ProcDetector) for x in dets):
                            continue
                    try:
                        if d is None:
                            d = _TextDetector("dml", ad["device_id"], label=label)
                            d.detect(probe)
                            d.vram_mb = ad["vram_mb"]
                        dets.append(d)
                    except Exception:
                        continue
        except Exception:
            pass
        _DETECTORS = dets
        return dets


# ----------------------------------------------------------------------------------------------------
# Inpainting helpers
# ----------------------------------------------------------------------------------------------------


_LAMA_LOCK = threading.Lock()
_LAMA_ERROR = ""


def _get_lama_session():
    """Loads optional LaMa ONNX model (data/models/lama_fp32.onnx) on a GPU provider and test-runs it once.

    Returns None (caller falls back to the fast Navier-Stokes fill) when the model is missing or cannot run on
    the GPU: the public LaMa ONNX export fails on DirectML (MatMul '...rttn/MatMul_5' -> 0x80070057) and on CPU
    it takes ~1 s per frame, which would make a render take hours. Measured on RTX 3060: CUDA ~150 ms/frame."""
    global _LAMA_SESSION, _LAMA_ERROR
    with _LAMA_LOCK:
        if _LAMA_SESSION is not None:
            return _LAMA_SESSION
        if _LAMA_ERROR:
            return None
        model_path = MODELS_DIR / "lama_fp32.onnx"
        if not model_path.exists():
            _LAMA_ERROR = "chưa có data/models/lama_fp32.onnx"
            return None
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            gpu = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in providers]
            if not gpu:
                _LAMA_ERROR = "không có GPU provider (CUDA/DirectML)"
                return None
            so = ort.SessionOptions()
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            if gpu[0] == "DmlExecutionProvider":
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess = ort.InferenceSession(str(model_path), sess_options=so, providers=[gpu[0]])
            z = np.zeros((1, 3, 512, 512), dtype=np.float32)
            m = np.zeros((1, 1, 512, 512), dtype=np.float32)
            m[..., 200:260, 100:400] = 1.0
            sess.run(None, {sess.get_inputs()[0].name: z, sess.get_inputs()[1].name: m})
            _LAMA_SESSION = sess
            return sess
        except Exception as e:  # noqa: BLE001
            _LAMA_ERROR = f"LaMa không chạy được trên GPU này ({str(e).splitlines()[0][:80]})"
            return None


def _inpaint_lama_roi(roi_bgr: np.ndarray, mask: np.ndarray, session) -> np.ndarray:
    """Runs LaMa ONNX on a 512x512 padded ROI patch and blends strictly inside the mask."""
    h, w = roi_bgr.shape[:2]
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    resized_mask = (resized_mask > 127).astype(np.float32)

    img_in = np.transpose(resized_img, (2, 0, 1))[None, ...]
    mask_in = resized_mask[None, None, ...]

    inputs = {session.get_inputs()[0].name: img_in, session.get_inputs()[1].name: mask_in}
    with _LAMA_LOCK:  # one GPU session shared by the inpaint pool; DirectML forbids concurrent Run()
        out = session.run(None, inputs)[0][0]
    out_rgb = np.clip(np.transpose(out, (1, 2, 0)), 0, 255).astype(np.uint8)
    if out_rgb.max() <= 1:
        out_rgb = np.clip(np.transpose(out, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)

    # Strictly preserve unmasked pixels 100% original
    result = roi_bgr.copy()
    m_bool = mask > 127
    result[m_bool] = out_bgr[m_bool]
    return result


def _temporal_and_clean_inpaint(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
    clean_ref_roi: Optional[np.ndarray],
    lama_session=None,
) -> np.ndarray:
    """Two-stage crisp reconstruction:
    1) Temporal Background Propagation: if an adjacent clean frame (no text) has matching border context,
       copies real unoccluded pixels directly (100% razor sharp, zero blur).
    2) For dynamic/moving areas, runs LaMa ONNX (if installed) or fine Navier-Stokes inpainting strictly inside text mask.
    """
    if mask is None or not np.any(mask):
        return roi_bgr

    # Check if temporal clean reference matches surrounding unmasked context
    if clean_ref_roi is not None and clean_ref_roi.shape == roi_bgr.shape:
        # Build narrow ring around the mask to test background similarity
        ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        ring = cv2.subtract(cv2.dilate(mask, ring_k), mask)
        ring_bool = ring > 127
        if np.any(ring_bool):
            d = np.abs(roi_bgr[ring_bool].astype(np.float32) - clean_ref_roi[ring_bool].astype(np.float32)).max(axis=-1)
            # Strict: a loose mean-only test (old: mean < 18) pasted moved objects (e.g. a walking character's
            # jacket) from the reference frame. Require the whole border to match, not just on average.
            if float(d.mean()) < 7.0 and float(np.percentile(d, 95)) < 20.0:
                # Copy real unoccluded pixels with soft 2px feather edge!
                alpha = cv2.GaussianBlur((mask > 127).astype(np.float32), (5, 5), 0)[..., None]
                blended = roi_bgr.astype(np.float32) * (1.0 - alpha) + clean_ref_roi.astype(np.float32) * alpha
                return np.clip(blended, 0, 255).astype(np.uint8)

    if lama_session is not None:
        return _inpaint_lama_roi(roi_bgr, mask, lama_session)

    # Navier-Stokes reconstruction strictly inside the OCR mask. Computed at half resolution and upscaled:
    # side-by-side tests on this video looked identical to full-res NS (the fill is a smooth smear either way)
    # but it is ~5x cheaper (27.9 -> 5.4 ms/frame), and inpainting was the slowest stage of the render.
    h, w = mask.shape[:2]
    if h >= 24 and w >= 24:
        small = cv2.resize(roi_bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        small_m = cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        filled = cv2.resize(cv2.inpaint(small, small_m, 2, cv2.INPAINT_NS), (w, h), interpolation=cv2.INTER_LINEAR)
        out = roi_bgr.copy()
        sel = mask > 0
        out[sel] = filled[sel]
        return out
    return cv2.inpaint(roi_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)


def _inpaint_text_region(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
    clean_ref_roi: Optional[np.ndarray],
    lama_session=None,
) -> np.ndarray:
    """Runs the reconstruction only on the bounding box of the text mask (+ margin) instead of the whole strip.
    Subtitles usually cover a small part of the ROI, so this is several times cheaper; pixels outside the
    box are untouched either way."""
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return roi_bgr
    pad = 16
    H, W = mask.shape[:2]
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    ref = clean_ref_roi[y0:y1, x0:x1] if clean_ref_roi is not None and clean_ref_roi.shape == roi_bgr.shape else None
    patch = _temporal_and_clean_inpaint(roi_bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1], ref, lama_session)
    out = roi_bgr.copy()
    out[y0:y1, x0:x1] = patch
    return out


# ----------------------------------------------------------------------------------------------------
# ffmpeg helpers
# ----------------------------------------------------------------------------------------------------


def _start_proc(cmd: List[str], **kw) -> subprocess.Popen:
    proc = subprocess.Popen(cmd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **kw)
    register_proc(proc)
    return proc


def _stop_proc(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    unregister_proc(proc)


def _drain_stderr(proc: subprocess.Popen, sink: List[bytes]) -> threading.Thread:
    def _run():
        try:
            for chunk in iter(lambda: proc.stderr.read(4096), b""):
                sink.append(chunk)
                if len(sink) > 64:
                    del sink[:-32]
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _roi_reader_cmd(ffmpeg: str, src: Path, rect: Tuple[int, int, int, int], limit_sec: Optional[float]) -> List[str]:
    """Decodes the source and emits ONLY the ROI strip of every frame (raw BGR). Uses the same decoder and
    the same input -t as the render command, so it yields exactly the same frames, in the same order.
    (NVDEC was tried for this reader on an RTX 3060: no speed gain, and its nv12 path changed pixel rounding
    enough to alter the glyph test results, so the software decoder is kept.)"""
    rx, ry, rw, rh = rect
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        *(["-t", f"{limit_sec:.3f}"] if limit_sec else []), "-i", str(src), "-an", "-sn",
        "-vf", f"crop={rw}:{rh}:{rx}:{ry}", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]


def _read_frame(stream, shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """Reads one raw frame straight into a new numpy array (readinto: no bytearray growth + bytes() copy,
    i.e. 2 fewer 600 KB copies per frame done while holding the GIL)."""
    arr = np.empty(shape, dtype=np.uint8)
    mv = memoryview(arr).cast("B")
    got, n = 0, mv.nbytes
    while got < n:
        k = stream.readinto(mv[got:])
        if not k:
            return None
        got += k
    return arr


def _read_exact(stream, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


# ----------------------------------------------------------------------------------------------------
# Single-pass AI clean stream (OCR + inpaint run while the final render encodes)
# ----------------------------------------------------------------------------------------------------

_OCR_CACHE_VERSION = "det-v2"


class AICleanStream:
    """Produces the cleaned ROI strip for every frame of the video, to be overlaid by the FINAL render ffmpeg
    (one decode + one encode for the whole edit). OCR runs on the fly with a small look-ahead:

        ffmpeg (ROI strips) -> reader -> look-ahead RAM buffer -> inpaint pool -> writer -> render ffmpeg stdin
                                   \\-> every Nth strip -> OCR pool (MX230 / Iris via DirectML, CPU only on backlog)

    OCR results are cached on disk per (video, ROI, sample rate) so preview -> full export or re-exports
    don't scan again.
    """

    def __init__(
        self,
        project: VideoProject,
        preview_seconds: Optional[float] = None,
        box: Optional[BoxRect] = None,
        source: "Optional[queue.Queue]" = None,
        text_logo: bool = False,
    ) -> None:
        """box: region to clean (default: the subtitle-removal box; text-logo / watermark regions pass theirs).
        source: frames come from this queue (MultiCleanStream: one shared decode for every region) instead of an
        own ffmpeg reader; None in the queue = end of video."""
        cfg: SubRemovalConfig = project.config.sub_removal
        box = box or cfg.box
        self._source = source
        # Text logos / watermarks are thin, semi-transparent text: the detector's boxes stop short of the faint
        # first/last glyph (measured: box started 13 px inside "本" of "本作品含AI生成内容"). Widen each box by ~half
        # a character (0.45 x text height) horizontally for these regions; subtitle boxes are unchanged.
        self.text_logo = text_logo
        self.cfg = cfg
        self.project = project
        self.src = Path(project.video_path)
        if not self.src.exists():
            raise RuntimeError(f"Không tìm thấy video: {self.src}")
        cap = cv2.VideoCapture(str(self.src))
        if not cap.isOpened():
            raise RuntimeError("Không thể mở file video bằng OpenCV.")
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or project.fps or 30.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or int(project.duration * self.fps))
        cap.release()
        self.limit_sec = preview_seconds if preview_seconds and preview_seconds > 0 else None
        self.total_frames = min(frames, int(round(self.limit_sec * self.fps))) if self.limit_sec else frames

        def _even(v: int) -> int:
            return v - (v % 2)

        self.video_size = (vw, vh)
        rx = _even(max(0, min(vw - 16, int(round(box.x * vw)))))
        ry = _even(max(0, min(vh - 16, int(round(box.y * vh)))))
        rw = _even(max(16, min(vw - rx, int(round(box.w * vw)))))
        rh = _even(max(16, min(vh - ry, int(round(box.h * vh)))))
        self.rect = (rx, ry, rw, rh)
        self.step = max(1, int(round(self.fps / max(1.0, cfg.ocr_sample_fps))))
        self.ffmpeg = get_ffmpeg_path()
        self.lama = _get_lama_session() if cfg.ai_engine == "lama" else None
        self.lama_note = f"LaMa tắt: {_LAMA_ERROR} → dùng tái tạo nhanh" if cfg.ai_engine == "lama" and self.lama is None else ""
        self.inpaint_workers = int(detect_hardware().get("inpaint_workers") or 4)
        self.inpaint_device = ""

        st = self.src.stat()
        key = f"{_OCR_CACHE_VERSION}|{self.src.resolve()}|{st.st_size}|{int(st.st_mtime)}|{self.rect}|{self.step}|{_DET_LIMIT_SIDE}"
        self.cache_path = PROJECTS_DIR / project.id / f"ocr_scan_{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"
        self.key_boxes: Dict[int, Optional[np.ndarray]] = {}
        self.ivl: Dict[int, str] = {}
        self._load_cache()
        self.cached_keys = len(self.key_boxes)

        # live counters (read by the render progress loop)
        self.written = 0
        self.cleaned = 0
        self.ocr_new = 0
        self.ocr_usage: Dict[str, int] = {}
        self.ocr_wait_sec = 0.0
        self.trans_fast = 0
        self.trans_ocr = 0
        self.trans_union = 0
        self.error: Optional[BaseException] = None

    # ---------- render integration ----------
    def ffmpeg_input_args(self) -> List[str]:
        rx, ry, rw, rh = self.rect
        return ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{rw}x{rh}", "-framerate", f"{self.fps:.6f}", "-i", "pipe:0"]

    def overlay_filters(self, src_label: str, strip_idx: int, out_label: str, gpu: bool = False) -> List[str]:
        # Both streams are re-timed by frame number so strip N always lands on frame N.
        rx, ry, _, _ = self.rect
        f = f"{self.fps:.6f}"
        if gpu:  # main frames are yuv420p CUDA frames: upload the small strip and overlay in VRAM
            return [
                f"[{src_label}]setpts=N/({f}*TB)[aic_bg]",
                f"[{strip_idx}:v]setpts=N/({f}*TB),format=yuv420p,hwupload_cuda[aic_fg]",
                f"[aic_bg][aic_fg]overlay_cuda={rx}:{ry}:eof_action=pass[{out_label}]",
            ]
        return [
            f"[{src_label}]setpts=N/({f}*TB)[aic_bg]",
            f"[{strip_idx}:v]setpts=N/({f}*TB)[aic_fg]",
            # format=yuv420 (not auto): source and output are yuv420p and rx/ry are even, so this avoids a
            # full-frame pixel-format round trip; ffmpeg-only render measured 535 -> 620 fps.
            f"[aic_bg][aic_fg]overlay={rx}:{ry}:eof_action=pass:format=yuv420[{out_label}]",
        ]

    def status_text(self) -> str:
        devs = ", ".join(f"{k} {v}" for k, v in self.ocr_usage.items() if v)
        ocr = f"OCR mới {self.ocr_new}" + (f" ({devs})" if devs else "") + (f", cache {self.cached_keys}" if self.cached_keys else "")
        dev = f" · tái tạo nền: {self.inpaint_device}" if self.inpaint_device else ""
        return f"Xóa sub AI: {self.cleaned} frame đã làm sạch · {ocr}{dev}" + (f" · {self.lama_note}" if self.lama_note else "")

    # ---------- cache ----------
    def _load_cache(self) -> None:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self.key_boxes = {
                int(k): (np.asarray(v, dtype=np.int32) if v is not None else None) for k, v in data["keys"].items()
            }
            self.ivl = {int(k): str(v) for k, v in data.get("ivl", {}).items()}
        except Exception:
            self.key_boxes, self.ivl = {}, {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "keys": {str(k): (v.tolist() if v is not None else None) for k, v in sorted(self.key_boxes.items())},
                "ivl": {str(k): v for k, v in sorted(self.ivl.items())},
            }
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            tmp.replace(self.cache_path)
        except Exception:
            pass

    # ---------- OCR pre-scan (A-Z pipeline: runs while translation / TTS wait on the network) ----------
    def prescan(self) -> int:
        """OCR of the sampled key frames only, saved to the same cache the render reads, so the render later
        skips OCR (RTX 3060 1080p: 236 -> 277 fps). Decodes the video once but crops/converts/pipes only every
        `step`-th frame. Returns the number of new samples. Full renders only (no preview limit)."""
        if not self.cfg.auto_detect_ocr or self.limit_sec:
            return 0
        step = self.step
        if self.total_frames and all(k in self.key_boxes for k in range(0, self.total_frames, step)):
            return 0
        rx, ry, rw, rh = self.rect
        detectors = _get_detectors()
        gpu = [d for d in detectors if d.device != "cpu"]
        dets = gpu or detectors
        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(self.src), "-an", "-sn",
            # n = decoded frame index = the render reader's frame index (same decoder, all frames)
            "-vf", f"select=not(mod(n\\,{step})),crop={rw}:{rh}:{rx}:{ry}", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        proc = _start_proc(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        q: "queue.Queue[Optional[Tuple[int, np.ndarray]]]" = queue.Queue(maxsize=64)
        errors: List[BaseException] = []
        new = [0]

        def _worker(det) -> None:
            batched = hasattr(det, "detect_many")
            try:
                while True:
                    items = [q.get()]
                    end = items[0] is None
                    while batched and not end and len(items) < 16:
                        try:
                            it = q.get_nowait()
                        except queue.Empty:
                            break
                        end = it is None
                        if not end:
                            items.append(it)
                    work = [it for it in items if it is not None]
                    if work:
                        res = det.detect_many([s for _, s in work]) if batched else [det.detect(work[0][1])]
                        for (k, _), boxes in zip(work, res):
                            self.key_boxes[k] = boxes
                        new[0] += len(work)
                    if end:
                        break
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(d,), daemon=True) for d in dets]
        for t in threads:
            t.start()
        try:
            i = 0
            while not errors:
                check_cancel()
                strip = _read_frame(proc.stdout, (rh, rw, 3))
                if strip is None:
                    break
                k = i * step
                while k not in self.key_boxes and not errors:
                    try:
                        q.put((k, strip), timeout=0.5)
                        break
                    except queue.Full:
                        check_cancel()
                i += 1
        finally:
            if errors or is_cancelled():
                while True:  # drop pending work so the sentinels below fit
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            for _ in threads:
                q.put(None)
            for t in threads:
                t.join()
            _stop_proc(proc)
            self._save_cache()
        if errors:
            raise RuntimeError(f"OCR quét trước lỗi: {errors[0]}")
        return new[0]

    # ---------- main loop ----------
    def run(self, write: Callable[[bytes], None]) -> Dict[str, Any]:
        """Streams one cleaned BGR strip per source frame into `write`. Blocking; call from a worker thread."""
        try:
            return self._run(write)
        except BaseException as e:  # noqa: BLE001
            self.error = e
            raise
        finally:
            self._save_cache()

    def _run(self, write: Callable[[bytes], None]) -> Dict[str, Any]:
        cfg = self.cfg
        rx, ry, rw, rh = self.rect
        step = self.step
        key_boxes = self.key_boxes
        window = max(step + 1, int(self.fps * _REF_WINDOW_SEC))
        stop = threading.Event()
        errors: List[BaseException] = []

        if not cfg.auto_detect_ocr:
            full = np.array([[[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]]], dtype=np.int32)
            for k in range(0, self.total_frames + step, step):
                key_boxes.setdefault(k, full)

        # ---- OCR pool ----
        detectors: List[_TextDetector] = _get_detectors() if cfg.auto_detect_ocr else []
        gpu_dets = [d for d in detectors if d.device != "cpu"]
        # The CPU detector (4 threads, ~22 ms/sample) only joins when the GPUs fall behind. With a strong GPU
        # (~6-10 ms/sample) that is rare, and those CPU threads are worth more to inpainting.
        cpu_backlog = 32 if any(d.vram_mb >= 4000 for d in gpu_dets) else 6
        warm = np.zeros((rh, rw, 3), dtype=np.uint8)
        for d in detectors:
            try:
                d.detect(warm)
            except Exception:
                pass
        start_counts = {d.label: d.count for d in detectors}
        # Priority queue: transition probes (prio 0) jump ahead of regular samples (prio 1).
        ocr_q: "queue.PriorityQueue[Tuple[int, int, Optional[str], int, Optional[np.ndarray]]]" = queue.PriorityQueue()
        seq = itertools.count()
        probe_res: Dict[int, bool] = {}
        ivl = self.ivl              # k0 -> per-frame subtitle codes for the frames between samples (cached)
        strip_store: Dict[int, np.ndarray] = {}   # RAM: strips not yet written, for transition probes

        def _ocr_worker(det: _TextDetector):
            # With GPUs available, the CPU detector only helps when the GPUs fall behind: the CPU is busy with
            # decoding / inpainting / the render's filters, and stealing it would slow the encode down.
            lazy_cpu = det.device == "cpu" and bool(gpu_dets)
            # DirectML busy-waits one full core per call while the GPU computes (~75 ms CPU per sample, measured).
            # Lower priority lets ffmpeg + inpainting go first; OCR runs ~20 s ahead (reader RAM buffer), so it
            # only has to keep up on average.
            # Strong GPU: the opposite. There the pipeline runs at 300-450 fps, so the look-ahead buffer is only
            # ~1 s of wall time, and with 8 inpaint threads + 2 ffmpeg saturating the CPU each DirectML Run()
            # waited for a time slice at every GPU sync point: 6-9 ms idle -> 112 ms under load (main loop then
            # waited 7.9 s of an 11 s render). The thread mostly sleeps on the GPU, so HIGHEST costs little CPU.
            if det.device != "cpu" and det.vram_mb >= 4000:
                _set_thread_priority(2)  # THREAD_PRIORITY_HIGHEST
            else:
                _lower_thread_priority()
            batched = hasattr(det, "detect_many")
            try:
                while not stop.is_set():
                    if lazy_cpu and ocr_q.qsize() < cpu_backlog:
                        time.sleep(0.02)
                        continue
                    try:
                        items = [ocr_q.get(timeout=0.2)]
                    except queue.Empty:
                        continue
                    # batched detector (CUDA): take whatever else is queued, one forward pass for all
                    end = items[0][2] is None
                    while batched and not end and len(items) < 16:
                        try:
                            it = ocr_q.get_nowait()
                        except queue.Empty:
                            break
                        if it[2] is None:
                            end = True
                        else:
                            items.append(it)
                    work = [it for it in items if it[2] is not None]
                    if work:
                        res = det.detect_many([it[4] for it in work]) if batched else [det.detect(work[0][4])]
                        for (_, _, kind, k, _), boxes in zip(work, res):
                            if kind == "key":
                                key_boxes[k] = boxes
                            else:
                                probe_res[k] = boxes is not None
                            self.ocr_new += 1
                        self.ocr_usage[det.label] = det.count - start_counts[det.label]
                    if end:
                        break
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                stop.set()

        ocr_threads = [threading.Thread(target=_ocr_worker, args=(d,), daemon=True) for d in detectors]
        for t in ocr_threads:
            t.start()

        # ---- reader (decode ROI strips) ----
        r_err: List[bytes] = []
        reader: Optional[subprocess.Popen] = None
        if self._source is None:
            reader = _start_proc(_roi_reader_cmd(self.ffmpeg, self.src, self.rect, self.limit_sec),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _drain_stderr(reader, r_err)

            def _next_strip() -> Optional[np.ndarray]:
                return _read_frame(reader.stdout, (rh, rw, 3))
        else:
            src_q = self._source

            def _next_strip() -> Optional[np.ndarray]:
                while not stop.is_set():
                    try:
                        return src_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                return None
        in_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=512)   # RAM buffer
        n_read = [0]
        read_pos = [0]  # live: frames [0, read_pos) are in strip_store (or already written)

        def _reader_thread():
            try:
                idx = 0
                while not stop.is_set():
                    strip = _next_strip()
                    if strip is None:
                        break
                    strip_store[idx] = strip
                    read_pos[0] = idx + 1
                    if idx % step == 0 and idx not in key_boxes:
                        ocr_q.put((1, next(seq), "key", idx, strip))
                    while not stop.is_set():
                        try:
                            in_q.put(strip, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                    idx += 1
                n_read[0] = idx
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
            finally:
                in_q.put(None)

        # ---- writer (keeps frame order, resolves inpaint futures) ----
        out_q: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=256)          # RAM buffer

        def _writer_thread():
            try:
                while True:
                    item = out_q.get()
                    if item is None:
                        break
                    arr = item.result() if hasattr(item, "result") else item
                    write(np.ascontiguousarray(arr).data)  # buffer protocol: no tobytes() copy
                    self.written += 1
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                stop.set()
                while out_q.get() is not None:
                    pass

        rt = threading.Thread(target=_reader_thread, daemon=True)
        wt = threading.Thread(target=_writer_thread, daemon=True)
        rt.start()
        wt.start()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.mask_dilate_px * 2 + 1, cfg.mask_dilate_px * 2 + 1))
        mask_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
        last_key = [None]  # last sampled key index (known once the reader hits EOF)

        def _keys_of(f: int) -> Tuple[int, int]:
            k0 = (f // step) * step
            k1 = k0 + step
            if f == k0 or (last_key[0] is not None and k1 > last_key[0]):
                k1 = k0
            return k0, k1

        def _text(k: int) -> bool:
            return key_boxes[k] is not None

        def _needs_ivl(k0: int, k1: int) -> bool:
            return k0 != k1 and (_text(k0) or _text(k1))

        def _known(f: int) -> bool:
            k0, k1 = _keys_of(f)
            if k0 not in key_boxes or k1 not in key_boxes:
                return False
            return not _needs_ivl(k0, k1) or k0 in ivl

        _CODE_KEYS = {"0": (), "1": (0,), "2": (1,), "3": (0, 1)}

        def _keys_used(f: int) -> Tuple[int, ...]:
            """Which samples' text boxes apply to frame f (between samples k0 and k1).
            ivl[k0] holds one code per in-between frame: '0' no text, '1' k0's text, '2' k1's text, '3' both.
            Codes come from the per-frame glyph test or OCR probes, so frames in a gap between two subtitles
            (or after a subtitle ends) stay 100% untouched instead of being smeared."""
            k0, k1 = _keys_of(f)
            if f == k0 or k0 == k1:
                return (k0,) if _text(k0) else ()
            if not _needs_ivl(k0, k1):
                return ()
            pair = (k0, k1)
            return tuple(pair[i] for i in _CODE_KEYS[ivl[k0][f - k0 - 1]])

        def _has_text(f: int) -> Optional[bool]:
            if not _known(f):
                return None
            return len(_keys_used(f)) > 0

        def _mask(f: int) -> Optional[np.ndarray]:
            used = _keys_used(f)
            if not used:
                return None
            if used in mask_cache:
                return mask_cache[used]
            m = np.zeros((rh, rw), dtype=np.uint8)
            for k in used:
                if self.text_logo:
                    for poly in key_boxes[k]:
                        x0, y0 = poly[:, 0].min(), poly[:, 1].min()
                        x1, y1 = poly[:, 0].max(), poly[:, 1].max()
                        th = max(4, int(y1 - y0))
                        px, py = int(0.45 * th) + 2, int(0.12 * th) + 2
                        cv2.rectangle(m, (max(0, x0 - px), max(0, y0 - py)), (min(rw - 1, x1 + px), min(rh - 1, y1 + py)), 255, -1)
                else:
                    cv2.fillPoly(m, list(key_boxes[k]), 255)
            m = cv2.dilate(m, kernel, iterations=1)
            if len(mask_cache) > 256:
                mask_cache.clear()
            mask_cache[used] = m
            return m

        def _white(s: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
            px = s[ys, xs].astype(np.int16)
            mx, mn = px.max(axis=1), px.min(axis=1)
            return (mx >= 185) & ((mx - mn) <= 60)   # bright & unsaturated = subtitle glyph colour

        def _glyph_presence(k_txt: int, k_other: int, frames: range) -> Optional[List[bool]]:
            """Is k_txt's subtitle still on screen in each frame? Subtitles are static white glyphs: take the
            glyph pixels of k_txt inside its OCR boxes and check whether they are still white (~0.1 ms/frame).
            None = not reliable (few glyph pixels / background white there too / fading) -> caller falls back."""
            s_txt, s_oth = strip_store.get(k_txt), strip_store.get(k_other)
            if s_txt is None or s_oth is None:
                return None
            box = np.zeros((rh, rw), dtype=np.uint8)
            cv2.fillPoly(box, list(key_boxes[k_txt]), 255)
            ys, xs = np.nonzero(box)
            if len(ys) == 0:
                return None
            g = _white(s_txt, ys, xs)
            if g.sum() < 150:
                return None
            oth_white = _white(s_oth, ys, xs)
            if _text(k_other):
                if (g & oth_white).sum() >= 0.75 * g.sum():
                    return [True] * len(frames)    # same sentence still shown at the other sample
                # Two different (usually centred, overlapping) sentences: use only the glyph pixels that
                # belong to THIS sentence, i.e. not white in the other sample.
                g = g & ~oth_white
                if g.sum() < 150:
                    return None
                ys, xs = ys[g], xs[g]
                base = 0.0
            else:
                ys, xs = ys[g], xs[g]
                base = float(oth_white[g].mean())
                if base > 0.25:
                    return None
            thr = (1.0 + base) / 2.0
            out: List[bool] = []
            for f in frames:
                s = strip_store.get(f)
                if s is None:
                    return None
                frac = float(_white(s, ys, xs).mean())
                if abs(frac - thr) < 0.12:
                    return None                    # ambiguous (fade / occlusion)
                out.append(frac >= thr)
            return out

        def _glyph_codes(k0: int, k1: int) -> Optional[str]:
            frames = range(k0 + 1, k1)
            p0 = p1 = [False] * len(frames)
            if _text(k0):
                r = _glyph_presence(k0, k1, frames)
                if r is None:
                    return None
                # k0's subtitle can only END inside the interval: keep it up to its last visible frame
                last = max((i for i, v in enumerate(r) if v), default=-1)
                p0 = [i <= last for i in range(len(frames))]
            if _text(k1):
                r = _glyph_presence(k1, k0, frames)
                if r is None:
                    return None
                # k1's subtitle can only START inside the interval: from its first visible frame on
                first = min((i for i, v in enumerate(r) if v), default=len(frames))
                p1 = [i >= first for i in range(len(frames))]
            return "".join(str(int(a) + 2 * int(b)) for a, b in zip(p0, p1))

        def _resolver_thread():
            """For every pair of neighbouring samples with text at either end, decide per in-between frame which
            subtitle is visible: glyph test first; otherwise OCR binary search for the single transition
            (text appears/disappears), or the safe union when both samples have text."""
            nxt = 0
            active: Dict[int, List[int]] = {}  # k0 -> [lo, hi, outstanding probe frame or -1]
            try:
                while not stop.is_set():
                    busy = False
                    # Wait until the reader has the strips up to k1: with OCR keys from the cache / pre-scan
                    # the keys are known long before the frames, and a glyph test without the strips fell
                    # back to the "union" (both subtitles smeared over every in-between frame).
                    while nxt in key_boxes and (nxt + step) in key_boxes and nxt + step < read_pos[0]:
                        k0, k1 = nxt, nxt + step
                        if _needs_ivl(k0, k1) and k0 not in ivl:
                            codes = _glyph_codes(k0, k1)
                            if codes is not None:
                                ivl[k0] = codes
                                self.trans_fast += 1
                            elif _text(k0) and _text(k1):
                                ivl[k0] = "3" * (step - 1)   # safe: union of both subtitles
                                self.trans_union += 1
                            else:
                                active[k0] = [k0, k1, -1]
                                self.trans_ocr += 1
                        nxt += step
                        busy = True
                    for k0, st in list(active.items()):
                        lo, hi, mid = st
                        k1_text = _text(k0 + step)
                        if mid >= 0:
                            if mid not in probe_res:
                                continue
                            if probe_res.pop(mid) == k1_text:
                                hi = mid
                            else:
                                lo = mid
                            st[:] = [lo, hi, -1]
                            busy = True
                        if hi - lo <= 1 or strip_store.get((lo + hi) // 2) is None:
                            t = hi if hi - lo <= 1 else (k0 + 1 if k1_text else k0 + step)  # conservative
                            ivl[k0] = "".join(
                                ("1" if not k1_text else "0") if f < t else ("2" if k1_text else "0")
                                for f in range(k0 + 1, k0 + step)
                            )
                            del active[k0]
                            continue
                        mid = (lo + hi) // 2
                        st[2] = mid
                        ocr_q.put((0, next(seq), "probe", mid, strip_store[mid]))
                    if not busy:
                        time.sleep(0.003)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                stop.set()

        res_t = threading.Thread(target=_resolver_thread, daemon=True) if cfg.auto_detect_ocr else None
        if res_t is not None:
            res_t.start()

        buf: "deque[Tuple[int, np.ndarray]]" = deque()
        reader_eof = False
        last_clean: Optional[Tuple[int, np.ndarray]] = None
        processed = 0
        # Inpainting was THE bottleneck with a fixed 4 threads (228 fps on a 6C/12T i5-12400F vs 447 fps with 8).
        # NVIDIA >= 4 GB + torch CUDA: whole chunks are reconstructed on the GPU instead (gpu_inpaint.py).
        gpu_inp = None
        if self.lama is None:
            from app.services.gpu_inpaint import get_gpu_inpainter

            gpu_inp = get_gpu_inpainter()
        self.inpaint_device = f"GPU {gpu_inp.name}" if gpu_inp is not None else f"CPU {self.inpaint_workers} luồng"
        gpu_pool = ThreadPoolExecutor(max_workers=1) if gpu_inp is not None else None
        pool = ThreadPoolExecutor(max_workers=self.inpaint_workers if gpu_inp is None else 1)

        def _fill():
            nonlocal reader_eof
            while not reader_eof and len(buf) < window + _CLEAN_CHUNK:
                item = in_q.get()
                if item is None:
                    reader_eof = True
                    last_key[0] = ((n_read[0] - 1) // step) * step if n_read[0] else 0
                    break
                buf.append(((buf[-1][0] + 1) if buf else processed, item))

        def _future_clean(f: int) -> Optional[np.ndarray]:
            for j, s in buf:
                if j <= f:
                    continue
                if j - f > window:
                    break
                if _has_text(j) is False:
                    return s
            return None

        def _raise_if_failed():
            if errors:
                check_cancel()
                e = errors[0]
                raise RuntimeError(f"AI xóa sub lỗi: {e}") from e

        try:
            while True:
                check_cancel()
                _raise_if_failed()
                _fill()
                if not buf:
                    break
                n = min(_CLEAN_CHUNK, len(buf))
                # wait until OCR has answered for every key this chunk depends on (look-ahead keeps GPUs busy)
                if not all(_known(buf[i][0]) for i in range(n)):
                    t_wait = time.time()
                    while not all(_known(buf[i][0]) for i in range(n)):
                        check_cancel()
                        _raise_if_failed()
                        time.sleep(0.005)
                    self.ocr_wait_sec += time.time() - t_wait
                jobs = []
                for i in range(n):
                    f, strip = buf[i]
                    m = _mask(f)
                    if m is None:
                        jobs.append(None)
                        continue
                    ref = last_clean[1] if last_clean is not None and f - last_clean[0] <= window else None
                    if ref is None:
                        ref = _future_clean(f)
                    jobs.append((strip, m, ref))
                for i in range(n):
                    if jobs[i] is None:
                        last_clean = buf[i]
                if gpu_inp is not None:
                    # whole chunk in one GPU call; each frame's "future" is a slice of the chunk result
                    futures: List[Any] = [None] * n
                    idx = [i for i in range(n) if jobs[i] is not None]
                    if idx:
                        cf = gpu_pool.submit(gpu_inp.process, [jobs[i][0] for i in idx],
                                             [jobs[i][1] for i in idx], [jobs[i][2] for i in idx])
                        for k, i in enumerate(idx):
                            futures[i] = _ChunkItem(cf, k)
                else:
                    futures = [pool.submit(_inpaint_text_region, j[0], j[1], j[2], self.lama) if j else None for j in jobs]
                for i in range(n):
                    f_i, strip = buf.popleft()
                    strip_store.pop(f_i, None)
                    if futures[i] is not None:
                        self.cleaned += 1
                    while True:
                        _raise_if_failed()
                        try:
                            out_q.put(futures[i] if futures[i] is not None else strip, timeout=0.5)
                            break
                        except queue.Full:
                            check_cancel()
                    processed += 1
            out_q.put(None)
            wt.join()
            _raise_if_failed()
        finally:
            stop.set()
            pool.shutdown(wait=False, cancel_futures=True)
            if gpu_pool is not None:
                gpu_pool.shutdown(wait=False, cancel_futures=True)
            for _ in ocr_threads:
                ocr_q.put((99, next(seq), None, 0, None))
            try:
                out_q.put_nowait(None)
            except queue.Full:
                pass
            _stop_proc(reader)

        if processed == 0:
            raise RuntimeError("Không đọc được frame nào: " + b"".join(r_err).decode("utf-8", "replace")[-300:])
        return {
            "frames": processed,
            "cleaned_frames": self.cleaned,
            "ocr_new": self.ocr_new,
            "ocr_cached": self.cached_keys,
            "ocr_devices": dict(self.ocr_usage),
            "ocr_wait_sec": round(self.ocr_wait_sec, 1),
            "transitions_glyph": self.trans_fast,
            "transitions_ocr": self.trans_ocr,
            "transitions_union": self.trans_union,
        }


# ----------------------------------------------------------------------------------------------------
# Standalone "clean video" export (kept for API compatibility; the A-Z pipeline renders in one pass)
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# Several regions (subtitle box + text-logo / watermark boxes) from ONE decode
# ----------------------------------------------------------------------------------------------------


def ai_clean_regions(project: VideoProject) -> List[Tuple[BoxRect, bool]]:
    """Every region the AI cleaner must process, as (box, is_text_logo): the subtitle box (mode ai_clean) +
    the text-logo / watermark boxes (wm_removal enabled, mode ai_clean)."""
    cfg = project.config
    regions: List[Tuple[BoxRect, bool]] = []
    if cfg.sub_removal.mode == "ai_clean":
        regions.append((cfg.sub_removal.box, False))
    wm = getattr(cfg, "wm_removal", None)
    if wm is not None and wm.enabled and wm.mode == "ai_clean":
        regions.extend((b, True) for b in wm.boxes)
    return regions


def ai_clean_boxes(project: VideoProject) -> List[BoxRect]:
    return [b for b, _ in ai_clean_regions(project)]


class MultiCleanStream:
    """N AICleanStreams fed by ONE ffmpeg decode: the reader crops every region (converted to BGR per region,
    so colours at a region's edge never mix with its neighbour), pads them to the same width and stacks them
    vertically; a distributor thread slices each frame back into the regions' queues. The cleaned regions are
    re-stacked into one strip per frame for the render ffmpeg, which crops and overlays each at its place.
    (One decode instead of one per region: an extra region costs OCR + inpainting only.)"""

    def __init__(self, project: VideoProject, preview_seconds: Optional[float], regions: List[Tuple[BoxRect, bool]]) -> None:
        self.queues: List[queue.Queue] = [queue.Queue(maxsize=64) for _ in regions]
        self.streams = [
            AICleanStream(project, preview_seconds, box=b, source=q, text_logo=tl)
            for (b, tl), q in zip(regions, self.queues)
        ]
        s0 = self.streams[0]
        self.fps, self.limit_sec, self.total_frames = s0.fps, s0.limit_sec, s0.total_frames
        self.src, self.ffmpeg = s0.src, s0.ffmpeg
        self.width = max(s.rect[2] for s in self.streams)
        self.offsets: List[int] = []
        y = 0
        for st in self.streams:
            self.offsets.append(y)
            y += st.rect[3]
        self.height = y
        self.error: Optional[BaseException] = None

    @property
    def written(self) -> int:
        return min(st.written for st in self.streams)

    def status_text(self) -> str:
        cleaned = sum(st.cleaned for st in self.streams)
        ocr = sum(st.ocr_new for st in self.streams)
        cached = sum(st.cached_keys for st in self.streams)
        dev = self.streams[0].inpaint_device
        return (
            f"Xóa sub + chữ logo AI ({len(self.streams)} vùng): {cleaned} lượt làm sạch · OCR mới {ocr}"
            + (f", cache {cached}" if cached else "") + (f" · tái tạo nền: {dev}" if dev else "")
        )

    def ffmpeg_input_args(self) -> List[str]:
        return ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}",
                "-framerate", f"{self.fps:.6f}", "-i", "pipe:0"]

    def overlay_filters(self, src_label: str, strip_idx: int, out_label: str, gpu: bool = False) -> List[str]:
        f = f"{self.fps:.6f}"
        n = len(self.streams)
        flt = [
            f"[{src_label}]setpts=N/({f}*TB)[mc_bg0]",
            f"[{strip_idx}:v]setpts=N/({f}*TB),split={n}" + "".join(f"[mc_s{i}]" for i in range(n)),
        ]
        cur = "mc_bg0"
        for i, st in enumerate(self.streams):
            rx, ry, rw, rh = st.rect
            chain = f"[mc_s{i}]crop={rw}:{rh}:0:{self.offsets[i]}"
            chain += ",format=yuv420p,hwupload_cuda" if gpu else ""
            flt.append(f"{chain}[mc_c{i}]")
            nxt = out_label if i == n - 1 else f"mc_bg{i + 1}"
            if gpu:
                flt.append(f"[{cur}][mc_c{i}]overlay_cuda={rx}:{ry}:eof_action=pass[{nxt}]")
            else:
                flt.append(f"[{cur}][mc_c{i}]overlay={rx}:{ry}:eof_action=pass:format=yuv420[{nxt}]")
            cur = nxt
        return flt

    def _reader_cmd(self) -> List[str]:
        n = len(self.streams)
        parts = [f"[0:v]split={n}" + "".join(f"[a{i}]" for i in range(n))]
        for i, st in enumerate(self.streams):
            rx, ry, rw, rh = st.rect
            parts.append(f"[a{i}]crop={rw}:{rh}:{rx}:{ry},format=bgr24,pad={self.width}:{rh}:0:0:black[p{i}]")
        parts.append("".join(f"[p{i}]" for i in range(n)) + f"vstack=inputs={n}[o]")
        return [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            *(["-t", f"{self.limit_sec:.3f}"] if self.limit_sec else []), "-i", str(self.src), "-an", "-sn",
            "-filter_complex", ";".join(parts), "-map", "[o]", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]

    def prescan(self) -> int:
        return sum(st.prescan() for st in self.streams)

    def run(self, write: Callable[[Any], None]) -> Dict[str, Any]:
        n = len(self.streams)
        stop = threading.Event()
        errors: List[BaseException] = []
        outs: List[queue.Queue] = [queue.Queue(maxsize=64) for _ in range(n)]
        results: List[Dict[str, Any]] = [{} for _ in range(n)]
        reader = _start_proc(self._reader_cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        r_err: List[bytes] = []
        _drain_stderr(reader, r_err)

        def _put(q: queue.Queue, item) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def _distribute() -> None:
            try:
                while not stop.is_set():
                    frame = _read_frame(reader.stdout, (self.height, self.width, 3))
                    if frame is None:
                        break
                    for i, st in enumerate(self.streams):
                        _, _, rw, rh = st.rect
                        y0 = self.offsets[i]
                        if not _put(self.queues[i], np.ascontiguousarray(frame[y0 : y0 + rh, :rw])):
                            return
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
            finally:
                for q in self.queues:
                    try:
                        q.put(None, timeout=5)
                    except queue.Full:
                        pass

        def _stream(i: int) -> None:
            st = self.streams[i]
            _, _, rw, rh = st.rect

            def _w(buf) -> None:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(rh, rw, 3).copy()
                if not _put(outs[i], arr):
                    raise RuntimeError("đã dừng")

            try:
                results[i] = st.run(_w)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                stop.set()
            finally:
                try:
                    outs[i].put(None, timeout=5)
                except queue.Full:
                    pass

        threads = [threading.Thread(target=contextvars.copy_context().run, args=(_distribute,), daemon=True)]
        threads += [threading.Thread(target=contextvars.copy_context().run, args=(_stream, i), daemon=True) for i in range(n)]
        for t in threads:
            t.start()
        frames = 0
        try:
            while True:
                check_cancel()
                if errors:
                    raise RuntimeError(f"AI xóa sub / chữ logo lỗi: {errors[0]}") from errors[0]
                parts = []
                for q in outs:
                    item = None
                    while not stop.is_set():
                        try:
                            item = q.get(timeout=0.5)
                            break
                        except queue.Empty:
                            check_cancel()
                            if errors:
                                break
                    parts.append(item)
                if errors:
                    continue
                if any(p is None for p in parts):
                    break
                stacked = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                for i, arr in enumerate(parts):
                    y0 = self.offsets[i]
                    stacked[y0 : y0 + arr.shape[0], : arr.shape[1]] = arr
                write(stacked.data)
                frames += 1
        except BaseException as e:  # noqa: BLE001
            self.error = e
            raise
        finally:
            stop.set()
            _stop_proc(reader)
            for q in self.queues:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
        if frames == 0:
            raise RuntimeError("Không đọc được frame nào: " + b"".join(r_err).decode("utf-8", "replace")[-300:])
        merged: Dict[str, Any] = {"frames": frames, "regions": n}
        for key in ("cleaned_frames", "ocr_new", "ocr_cached"):
            merged[key] = sum(int(r.get(key, 0) or 0) for r in results)
        merged["ocr_wait_sec"] = round(max(float(r.get("ocr_wait_sec", 0) or 0) for r in results), 1)
        merged["per_region"] = results
        return merged


def make_clean_stream(project: VideoProject, preview_seconds: Optional[float] = None):
    """The AI cleaner for this project: None (no AI region), one AICleanStream (single region: unchanged
    path) or a MultiCleanStream (subtitle + text-logo / watermark regions, one shared decode)."""
    regions = ai_clean_regions(project)
    if not regions:
        return None
    if len(regions) == 1:
        return AICleanStream(project, preview_seconds, box=regions[0][0], text_logo=regions[0][1])
    return MultiCleanStream(project, preview_seconds, regions)


def prescan_ocr_sync(project: VideoProject) -> int:
    """Fills the OCR cache of every AI region for a full render of `project` (see AICleanStream.prescan)."""
    st = make_clean_stream(project)
    return st.prescan() if st is not None else 0


def _encoder_args(enc: str) -> List[str]:
    if enc == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p2", "-cq", "18", "-pix_fmt", "yuv420p"]
    if enc == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "faster", "-global_quality", "19", "-pix_fmt", "nv12"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]


def clean_hardsub_video_sync(
    project: VideoProject,
    preview_seconds: Optional[float] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Tuple[Path, Dict[str, Any]]:
    stream = AICleanStream(project, preview_seconds)
    enc = detect_hardware()["recommended_encoder"]
    out_dir = PROJECTS_DIR / project.id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"clean_sub_{'preview' if preview_seconds else 'full'}.mp4"
    lim = ["-t", f"{stream.limit_sec:.3f}"] if stream.limit_sec else []
    cmd = [
        stream.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *lim, "-i", str(stream.src),
        *stream.ffmpeg_input_args(),
        "-filter_complex", ";".join(stream.overlay_filters("0:v", 1, "v")),
        "-map", "[v]", "-map", "0:a?", *_encoder_args(enc), "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(out_video),
    ]
    proc = _start_proc(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    err: List[bytes] = []
    _drain_stderr(proc, err)
    t0 = time.time()
    result: Dict[str, Any] = {}

    def _feed():
        try:
            result.update(stream.run(proc.stdin.write))
        except BaseException:  # noqa: BLE001
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    th = threading.Thread(target=contextvars.copy_context().run, args=(_feed,), daemon=True)
    th.start()
    while th.is_alive():
        th.join(1.0)
        if is_cancelled():
            _stop_proc(proc)
        if on_progress and stream.total_frames:
            on_progress(round(min(98.0, stream.written / stream.total_frames * 98.0), 1), stream.status_text())
    ret = proc.wait()
    unregister_proc(proc)
    check_cancel()
    if stream.error is not None:
        raise RuntimeError(f"AI Clean Sub lỗi: {stream.error}")
    if ret != 0:
        raise RuntimeError(f"AI Clean Sub xuất video lỗi: {b''.join(err).decode('utf-8', 'replace')[-400:]}")
    stats = {**result, "total_frames": result.get("frames", 0), "encoder": enc,
             "untouched_frames": result.get("frames", 0) - result.get("cleaned_frames", 0),
             "seconds": round(time.time() - t0, 1), "out_video": str(out_video)}
    if on_progress:
        on_progress(100.0, f"Xóa sub sạch xong! {stream.status_text()}")
    return out_video, stats
