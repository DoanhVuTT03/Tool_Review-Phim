"""PP-OCR text detector on an NVIDIA GPU with PyTorch CUDA (same model as RapidOCR, converted with onnx2torch).

Why: the DirectML detector processes shared the GPU with the render (NVDEC / CUDA overlays / NVENC) and with
the CUDA inpainting; every Run() waited for a GPU time slice (~100 ms) and DirectML busy-waits a CPU core
meanwhile. In the render's own CUDA context, batched, the detector does ~240 strips/s and costs no CPU spin.

Accuracy: fp32 with TF32 disabled matches onnxruntime to ~1e-4 on the probability map (120/120 identical
text/no-text decisions on the test video). TF32 (default on Ampere) changed 2/120 decisions; fp16 overflows.
The DB post-processing is RapidOCR's own code, so boxes are the same as the DirectML / CPU detector's.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import cv2
import numpy as np

_LOCK = threading.Lock()
_INST: Optional["TorchTextDetector"] = None
_ERROR = ""


def get_torch_detector(det_limit_side: int, box_thresh: float, min_vram_mb: int = 4000) -> Optional["TorchTextDetector"]:
    global _INST, _ERROR
    with _LOCK:
        if _INST is not None:
            return _INST
        if _ERROR:
            return None
        try:
            import torch

            if not torch.cuda.is_available():
                _ERROR = "torch không thấy CUDA"
                return None
            if (torch.cuda.get_device_properties(0).total_memory >> 20) < min_vram_mb:
                _ERROR = "GPU nhỏ"
                return None
            _INST = TorchTextDetector(det_limit_side, box_thresh)
            return _INST
        except Exception as e:  # noqa: BLE001
            _ERROR = f"OCR CUDA không khởi động được: {e}"[:200]
            return None


def _load_detector_module(model_path: str):
    """Paddle2ONNX stores every weight as a Constant node; onnx2torch wants initializers."""
    import onnx
    from onnx import numpy_helper
    from onnx2torch import convert

    model = onnx.load(model_path)
    g = model.graph
    keep = []
    for n in g.node:
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
            g.initializer.append(numpy_helper.from_array(numpy_helper.to_array(n.attribute[0].t), n.output[0]))
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    return convert(model)


class TorchTextDetector:
    device = "cuda"

    def __init__(self, det_limit_side: int, box_thresh: float) -> None:
        import onnxruntime as ort
        import torch
        from rapidocr_onnxruntime import RapidOCR

        ort.set_default_logger_severity(3)
        # exact fp32 convolutions (see module docstring); the fp16 inpainting is not affected by these flags
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.t = torch
        self.limit = det_limit_side
        ocr = RapidOCR(det_limit_type="max", det_limit_side_len=det_limit_side, det_box_thresh=box_thresh)
        self._det = ocr.text_det  # DB post-processing (+ the ORT CPU session, unused)
        self.model = _load_detector_module(str(self._det.infer.session._model_path)).cuda().eval()
        self.label = f"{torch.cuda.get_device_name(0).replace('NVIDIA GeForce ', '')} CUDA"
        self.vram_mb = torch.cuda.get_device_properties(0).total_memory >> 20
        self.count = 0
        self._lock = threading.Lock()
        self._stream = torch.cuda.Stream()
        self.detect(np.zeros((96, 960, 3), np.uint8))

    def _blob(self, roi_bgr: np.ndarray) -> np.ndarray:
        # identical to sub_remover._TextDetector.detect's pre-processing
        h, w = roi_bgr.shape[:2]
        ratio = min(1.0, self.limit / max(h, w))
        dh = max(32, int(round(h * ratio / 32)) * 32)
        dw = max(32, int(round(w * ratio / 32)) * 32)
        img = cv2.resize(roi_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR) if (dw, dh) != (w, h) else roi_bgr
        return cv2.dnn.blobFromImage(img, scalefactor=1.0 / 127.5, mean=(127.5, 127.5, 127.5), swapRB=False)

    def detect(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        return self.detect_many([roi_bgr])[0]

    def detect_many(self, rois: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """One batched forward pass for strips of the same size (all strips of a video are)."""
        if not rois:
            return []
        t = self.t
        blob = np.concatenate([self._blob(r) for r in rois])
        with self._lock, t.cuda.stream(self._stream), t.inference_mode():
            preds = self.model(t.from_numpy(blob).cuda(non_blocking=False)).float().cpu().numpy()
        out: List[Optional[np.ndarray]] = []
        for i, r in enumerate(rois):
            h, w = r.shape[:2]
            boxes, _ = self._det.postprocess_op(preds[i : i + 1], (h, w))
            boxes = self._det.filter_tag_det_res(boxes, (h, w))
            out.append(None if boxes is None or len(boxes) == 0 else np.asarray(boxes, dtype=np.int32))
        self.count += len(rois)
        return out
