"""AI Clean Sub background reconstruction on an NVIDIA GPU (PyTorch CUDA), a whole chunk of frames per call.

Same two-stage logic as the CPU path in sub_remover (_temporal_and_clean_inpaint), run batched in VRAM:
1) Temporal propagation: if a nearby clean frame matches the ring around the text mask (mean |diff| < 7 and
   95th percentile < 20), its real pixels are blended in with a ~2 px feather (razor sharp, no smear).
2) Otherwise the masked text is filled from the surrounding pixels: push-pull pyramid (membrane fill) + a
   few smoothing passes inside the mask, instead of OpenCV's CPU Navier-Stokes. Pixels outside the mask are
   untouched.

Why: on an i5-12400F + RTX 3060 the CPU inpainting (8 threads) was the slowest stage of the whole render once
decode/overlay/encode moved to NVDEC/CUDA/NVENC; the GPU does a 48-frame 1458x140 chunk in a few ms.
Optional: without torch / CUDA the caller keeps the CPU path.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

_LOCK = threading.Lock()
_INST: Optional["GpuInpainter"] = None
_ERROR = ""


def get_gpu_inpainter(min_vram_mb: int = 4000) -> Optional["GpuInpainter"]:
    """Shared instance, or None when torch/CUDA is missing or the GPU is small (laptop MX230 keeps the CPU)."""
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
            vram = torch.cuda.get_device_properties(0).total_memory >> 20
            if vram < min_vram_mb:
                _ERROR = f"GPU chỉ có {vram} MB VRAM"
                return None
            inst = GpuInpainter(torch)
            inst.warmup()
            _INST = inst
            return inst
        except Exception as e:  # noqa: BLE001
            _ERROR = f"GPU inpaint không khởi động được: {e}"[:200]
            return None


def gpu_inpaint_error() -> str:
    return _ERROR


class GpuInpainter:
    def __init__(self, torch) -> None:
        self.t = torch
        self.dev = torch.device("cuda")
        self.name = torch.cuda.get_device_name(0)
        # Several worker threads may call process() at once: each gets its own CUDA stream + pinned buffers,
        # so one chunk's upload/download overlaps another's kernels (the GPU is time-sliced with the render
        # ffmpeg's CUDA context, so a lone chunk spent ~70 ms mostly waiting for its turn).
        self._tls = threading.local()
        F = torch.nn.functional
        self.F = F
        # 5x5 Gaussian, sigma 1.1 = cv2.GaussianBlur(ksize=5, sigma=0) used by the CPU feather
        g = torch.exp(-((torch.arange(5, dtype=torch.float32) - 2) ** 2) / (2 * 1.1**2))
        g = g / g.sum()
        self.gauss = (g[:, None] * g[None, :]).to(self.dev)[None, None]
        # 11x11 ellipse (cv2.MORPH_ELLIPSE) for the ring test, as a conv kernel
        import cv2

        ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)).astype(np.float32)
        self.ring_k = torch.from_numpy(ell).to(self.dev)[None, None]

    def warmup(self) -> None:
        h, w = 64, 256
        s = np.zeros((h, w, 3), np.uint8)
        m = np.zeros((h, w), np.uint8)
        m[20:40, 50:200] = 255
        self.process([s, s], [m, m], [None, s])
        self.t.cuda.synchronize()

    # ------------------------------------------------------------------------------------------------
    def _upload_unique(self, arrs: List[np.ndarray]):
        """Stacks the distinct arrays once (refs / masks are shared by many frames) + an index per item."""
        t = self.t
        uniq: dict = {}
        order: List[int] = []
        for a in arrs:
            k = id(a)
            if k not in uniq:
                uniq[k] = len(uniq)
            order.append(uniq[k])
        by_idx = {v: k for k, v in uniq.items()}
        objs = {id(a): a for a in arrs}
        items = [objs[by_idx[i]] for i in range(len(uniq))]
        shape = (len(items),) + items[0].shape
        # reusable page-locked upload buffers, per thread (H2D 3.2 -> 1.2 ms per 48 strips)
        pinned = self._tls.__dict__.setdefault("pinned", {})
        key = (shape[1:], arrs[0].dtype.str)
        buf = pinned.get(key)
        if buf is None or buf.shape[0] < shape[0]:
            buf = t.empty((max(shape[0], 64),) + shape[1:], dtype=t.uint8, pin_memory=True)
            pinned[key] = buf
        np.stack(items, out=buf.numpy()[: shape[0]])
        gpu = buf[: shape[0]].to(self.dev, non_blocking=True)
        return gpu, t.tensor(order, device=self.dev)

    def _fill(self, x, known):
        """Fill of the unknown pixels (known: B,1,H,W in {0,1}). Computed at half resolution (the fill is a
        smooth membrane anyway; the CPU path ran Navier-Stokes at half res too) and composited at full res,
        so known pixels stay exactly as they were. 4x less work than full res: 82 ms -> a few ms / 48 frames."""
        F = self.F
        H, W = x.shape[-2:]
        k_sum = F.avg_pool2d(known, 2, ceil_mode=True)
        x_h = F.avg_pool2d(x * known, 2, ceil_mode=True) / k_sum.clamp_min(1e-3)
        known_h = (k_sum > 0.99).to(x.dtype)  # half-res pixel is known only if no text pixel fell in it
        pp = self._fill_lowres(x_h, known_h)
        # Text is a wide, short shape: interpolate each column between the pixel just above and just below the
        # hole. Vertical structures (walls, pillars, colour boundaries) stay sharp and vertical gradients (sky)
        # stay right; the isotropic push-pull alone blended left/right colours into a visible band (magenta |
        # cyan -> purple strip). Push-pull remains the fallback where a hole touches the region's top/bottom.
        vf, both = self._fill_vertical(x_h, known_h)
        fill_h = self.t.where(both, vf, pp)
        unknown_h = 1.0 - known_h
        # light horizontal smoothing inside the hole only (no vertical streaks)
        sm = F.avg_pool2d(F.pad(fill_h, (2, 2, 0, 0), mode="replicate"), (1, 5), stride=1)
        fill_h = x_h * known_h + sm * unknown_h
        up = F.interpolate(fill_h, size=(H, W), mode="bilinear", align_corners=False)
        return x * known + up * (1.0 - known)

    def _fill_vertical(self, x, known):
        """Per-column linear interpolation between the nearest known pixel above and below each unknown pixel.
        Returns (fill, mask of pixels that had a known pixel on BOTH sides)."""
        t = self.t
        B, C, h, w = x.shape
        rows = t.arange(h, device=x.device).view(1, 1, h, 1).expand(B, 1, h, w)
        kn = known > 0.5
        above = t.where(kn, rows, t.full_like(rows, -1)).cummax(dim=2).values
        below = t.where(kn, rows, t.full_like(rows, h)).flip(2).cummin(dim=2).values.flip(2)
        has_a, has_b = above >= 0, below < h
        va = x.gather(2, above.clamp(0, h - 1).expand(B, C, h, w))
        vb = x.gather(2, below.clamp(0, h - 1).expand(B, C, h, w))
        wgt = ((rows - above).to(x.dtype) / (below - above).clamp_min(1).to(x.dtype)).clamp(0, 1)
        return va * (1 - wgt) + vb * wgt, (has_a & has_b)

    def _fill_lowres(self, x, known):
        """Push-pull membrane fill + smoothing passes inside the holes."""
        t, F = self.t, self.F
        levels = []
        v, k = x * known, known
        while min(v.shape[-2:]) > 3:
            levels.append((v, k))
            v = F.avg_pool2d(v, 2, ceil_mode=True)
            k = F.avg_pool2d(k, 2, ceil_mode=True)
        c = v / k.clamp_min(1e-6)
        # coarsest level may still have holes (text across the whole strip height): use the frame mean
        mean = (x * known).float().sum(dim=(2, 3), keepdim=True) / known.float().sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        c = t.where(k > 1e-6, c, mean.to(c.dtype).expand_as(c))
        for v_l, k_l in reversed(levels):
            # v_l = k_l * mean(known pixels of the cell): blend it with the coarser estimate by the
            # fraction of known pixels. At full res (k in {0,1}) this keeps known pixels exactly.
            up = F.interpolate(c, size=v_l.shape[-2:], mode="bilinear", align_corners=False)
            c = v_l + (1.0 - k_l) * up
        # a few Jacobi passes inside the mask remove the pyramid's blockiness
        unknown = 1.0 - known
        base = x * known
        for _ in range(8):  # 4-neighbour average via shifted views (memory-bound, no conv setup)
            p = F.pad(c, (1, 1, 1, 1), mode="replicate")
            sm = (p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:]) * 0.25
            c = base + sm * unknown
        return c

    @staticmethod
    def _percentile95(t, d, ring):
        """Per-frame 95th percentile of d over the ring pixels (B,H,W), vectorised with one sort."""
        B = d.shape[0]
        flat = t.where(ring, d, t.full_like(d, -1.0)).reshape(B, -1)
        srt, _ = t.sort(flat, dim=1, descending=True)
        n = ring.reshape(B, -1).sum(dim=1)
        idx = (n.float() * 0.05).floor().long().clamp(min=0)
        idx = t.minimum(idx, (n - 1).clamp(min=0))
        return srt.gather(1, idx[:, None])[:, 0], n

    def process(self, strips: List[np.ndarray], masks: List[np.ndarray], refs: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        """strips: B x (H,W,3) uint8 BGR; masks: B x (H,W) uint8 (255 = text); refs: clean frame or None."""
        t, F = self.t, self.F
        B = len(strips)
        if B == 0:
            return []
        stream = self._tls.__dict__.get("stream")
        if stream is None:
            stream = self._tls.stream = t.cuda.Stream()
        with t.cuda.stream(stream), t.inference_mode():
            xs, xi = self._upload_unique(strips)
            ms, mi = self._upload_unique(masks)
            x = xs[xi].permute(0, 3, 1, 2).half()                   # B,3,H,W  (fp16: half the bandwidth)
            m = (ms[mi] > 127).half()[:, None]                       # B,1,H,W
            known = 1.0 - m
            out = self._fill(x, known)

            has_ref = [r is not None and r.shape == strips[0].shape for r in refs]
            if any(has_ref):
                sel = [i for i, h in enumerate(has_ref) if h]
                rs, ri = self._upload_unique([refs[i] for i in sel])
                r = rs[ri].permute(0, 3, 1, 2).half()
                xm, mm = x[sel], m[sel]
                ring = (F.conv2d(mm, self.ring_k.to(mm.dtype), padding=5) > 0.5) & (mm < 0.5)      # B',1,H,W
                d = (xm - r).abs().amax(dim=1, keepdim=True).float()  # stats in fp32 (fp16 sums overflow)
                n = ring.sum(dim=(1, 2, 3))
                mean = (d * ring).sum(dim=(1, 2, 3)) / n.clamp_min(1)
                p95, _ = self._percentile95(t, d[:, 0], ring[:, 0])
                ok = (n > 0) & (mean < 7.0) & (p95 < 20.0)
                # no `if ok.any()`: that would be a GPU->CPU sync (= waiting for a GPU time slice)
                alpha = F.conv2d(F.pad(mm, (2, 2, 2, 2), mode="reflect"), self.gauss.to(mm.dtype))
                blended = xm * (1.0 - alpha) + r * alpha
                sel_t = t.tensor(sel, device=self.dev)
                out[sel_t] = t.where(ok[:, None, None, None], blended, out[sel_t])
            res = out.round_().clamp_(0, 255).to(t.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [res[i] for i in range(B)]
