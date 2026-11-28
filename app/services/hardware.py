from __future__ import annotations
import platform
import shutil
import subprocess
from typing import Any, Dict, List
import psutil

from app.config import get_ffmpeg_path


_HARDWARE_CACHE: Dict[str, Any] | None = None


def _run_cmd(cmd: List[str], timeout: int = 8) -> tuple[int, str, str]:
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def _probe_ffmpeg_encoder(ffmpeg_path: str, encoder: str) -> bool:
    """Runs a real 1-frame encode test to verify silicon support (avoids false positives like MX230)."""
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.1:r=10",
        "-c:v",
        encoder,
        "-f",
        "null",
        "-",
    ]
    code, _, _ = _run_cmd(cmd, timeout=8)
    return code == 0


def _detect_nvidia_gpus() -> List[Dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    code, out, _ = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if code != 0 or not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                vram_mb = int(float(parts[1]))
            except ValueError:
                vram_mb = 0
            gpus.append(
                {
                    "name": parts[0],
                    "vram_mb": vram_mb,
                    "driver": parts[2] if len(parts) > 2 else "",
                }
            )
    return gpus


def _detect_windows_gpus() -> List[str]:
    code, out, _ = _run_cmd(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_VideoController).Name",
        ],
        timeout=6,
    )
    if code != 0 or not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def detect_hardware(force_refresh: bool = False) -> Dict[str, Any]:
    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is not None and not force_refresh:
        return _HARDWARE_CACHE

    ffmpeg_path = get_ffmpeg_path()
    ffmpeg_ok = False
    ffmpeg_version = "Chưa cài đặt"
    code, out, _ = _run_cmd([ffmpeg_path, "-version"], timeout=5)
    if code == 0 and out:
        ffmpeg_ok = True
        ffmpeg_version = out.splitlines()[0]

    cpu_cores = psutil.cpu_count(logical=False) or 2
    cpu_threads = psutil.cpu_count(logical=True) or 4
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    nvidia_gpus = _detect_nvidia_gpus()
    all_gpu_names = _detect_windows_gpus()
    has_intel_igpu = any("intel" in g.lower() for g in all_gpu_names)

    nvenc_ok = False
    qsv_ok = False
    if ffmpeg_ok:
        if nvidia_gpus:
            nvenc_ok = _probe_ffmpeg_encoder(ffmpeg_path, "h264_nvenc")
        if has_intel_igpu or not nvenc_ok:
            qsv_ok = _probe_ffmpeg_encoder(ffmpeg_path, "h264_qsv")

    if nvenc_ok:
        recommended_encoder = "h264_nvenc"
    elif qsv_ok:
        recommended_encoder = "h264_qsv"
    else:
        recommended_encoder = "libx264"

    max_nvidia_vram = max((g["vram_mb"] for g in nvidia_gpus), default=0)
    dml_ok = False
    try:
        import onnxruntime as ort

        dml_ok = "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        pass
    # Physical GPUs only (no "Basic Render" / virtual display adapters such as MuMuPlayer's).
    real_gpus = [g for g in all_gpu_names if not any(x in g.lower() for x in ("basic render", "virtual", "mumu", "parsec", "remote"))]
    ai_device = "cpu+directml" if dml_ok else "cpu"
    ocr_devices = "CPU + " + " + ".join(dict.fromkeys(real_gpus)) + " (DirectML)" if dml_ok and real_gpus else "CPU"
    dual_gpu_capable = bool(qsv_ok and nvidia_gpus and not nvenc_ok) or bool(qsv_ok and nvenc_ok)
    # AI Clean Sub inpainting threads (OpenCV releases the GIL, so this scales ~linearly). Leave ~4 hardware
    # threads for the render ffmpeg (decode + overlays) and the ROI reader: 4C/8T laptop -> 4, 6C/12T -> 8.
    inpaint_workers = max(2, min(16, cpu_threads - 4))

    import importlib.util

    torch_cuda_pkg = bool(importlib.util.find_spec("torch")) and bool(importlib.util.find_spec("onnx2torch"))
    if nvenc_ok and max_nvidia_vram >= 6000 and cpu_cores >= 6:
        tier = "workstation"
        # One NVENC chip on GeForce cards: 2 renders in parallel keep it busy while the other one is in
        # Python/CPU stages; more only fights over the same CPU cores.
        rec_render_workers = 2
        rec_io_workers = 10
        if torch_cuda_pkg:
            summary = (
                f"Máy khỏe: {nvidia_gpus[0]['name']} ({max_nvidia_vram} MB VRAM) — toàn bộ trên GPU: NVDEC giải mã, "
                f"CUDA ghép lớp, NVENC nén; AI OCR + tái tạo nền xóa sub chạy CUDA (PyTorch)."
            )
        else:
            summary = (
                f"Máy khỏe: {nvidia_gpus[0]['name']} ({max_nvidia_vram} MB VRAM) — NVDEC/CUDA/NVENC; AI OCR qua "
                f"DirectML, tái tạo nền {inpaint_workers} luồng CPU (chạy setup.bat để cài PyTorch CUDA: nhanh ~2x)."
            )
    elif qsv_ok and nvidia_gpus:
        tier = "dual_gpu_laptop"
        rec_render_workers = 1
        rec_io_workers = 6
        summary = (
            f"Dual-GPU Laptop: Intel Quick Sync (h264_qsv) nén video. "
            f"{nvidia_gpus[0]['name']} không có NVENC nên không nén video; "
            f"AI OCR xóa sub chạy song song trên {ocr_devices}."
        )
    elif qsv_ok:
        tier = "igpu_laptop"
        rec_render_workers = 1
        rec_io_workers = 6
        summary = "Intel Quick Sync Video (h264_qsv) xuất video bằng GPU tích hợp."
    else:
        tier = "cpu_only"
        rec_render_workers = 1
        rec_io_workers = 4
        summary = "Chế độ CPU phần mềm (libx264)."

    if not dml_ok and platform.system() == "Windows":
        summary += (
            " ⚠ AI OCR đang chạy CPU vì gói onnxruntime thường đã đè lên onnxruntime-directml"
            " — chạy setup.bat để sửa."
        )

    _HARDWARE_CACHE = {
        "cpu_name": platform.processor() or "CPU",
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "ram_gb": ram_gb,
        "gpus": all_gpu_names,
        "nvidia_gpus": nvidia_gpus,
        "has_intel_igpu": has_intel_igpu,
        "ffmpeg_available": ffmpeg_ok,
        "ffmpeg_path": ffmpeg_path,
        "ffmpeg_version": ffmpeg_version,
        "encoders": {
            "h264_nvenc": nvenc_ok,
            "h264_qsv": qsv_ok,
            "libx264": ffmpeg_ok,
        },
        "recommended_encoder": recommended_encoder,
        "ai_device": ai_device,
        "directml_ok": dml_ok,
        "inpaint_workers": inpaint_workers,
        "torch_cuda_installed": torch_cuda_pkg,
        "ocr_devices": ocr_devices,
        "dual_gpu_capable": dual_gpu_capable,
        "tier": tier,
        "recommended_render_workers": rec_render_workers,
        "recommended_io_workers": rec_io_workers,
        "summary": summary,
    }
    return _HARDWARE_CACHE
