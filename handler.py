"""RunPod Serverless handler — CatVTON inference on CUDA (RTX 4090)."""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import runpod
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

# ── Configuration via environment variables ──
CATVTON_SRC = os.environ.get("CATVTON_SRC", "/app/CatVTON")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "booksforcharlie/stable-diffusion-inpainting")
ATTN_REPO = os.environ.get("ATTN_REPO", "zhengchong/CatVTON")
ATTN_VERSION = os.environ.get("ATTN_VERSION", "mix")
ATTN_VERSION_MAP = {
    "mix": "mix-48k-1024",
    "vitonhd": "vitonhd-16k-512",
    "dresscode": "dresscode-16k-512",
}

# Global pipeline (loaded once on cold start)
_pipe = None
_pipe_dtype = None


def _import_catvton():
    catvton_src = Path(CATVTON_SRC).resolve()
    if not catvton_src.exists():
        raise RuntimeError(f"CatVTON source missing: {catvton_src}")
    if str(catvton_src) not in sys.path:
        sys.path.insert(0, str(catvton_src))
    from model.pipeline import CatVTONPipeline
    from utils import init_weight_dtype
    return CatVTONPipeline, init_weight_dtype


def _get_pipe():
    """Lazy-load and cache the CatVTON pipeline (survives across warm requests)."""
    global _pipe, _pipe_dtype
    if _pipe is not None:
        return _pipe

    CatVTONPipeline, init_weight_dtype = _import_catvton()

    # RTX 4090: bf16 for best speed/quality tradeoff
    mixed = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    _pipe_dtype = mixed

    _pipe = CatVTONPipeline(
        base_ckpt=BASE_MODEL_PATH,
        attn_ckpt=ATTN_REPO,
        attn_ckpt_version=ATTN_VERSION,
        weight_dtype=init_weight_dtype(mixed),
        device="cuda",
        skip_safety_check=True,
        use_tf32=True,
    )

    # Memory optimizations
    if hasattr(_pipe, "vae"):
        if hasattr(_pipe.vae, "enable_tiling"):
            _pipe.vae.enable_tiling()
        if hasattr(_pipe.vae, "enable_slicing"):
            _pipe.vae.enable_slicing()

    print(f"[handler] Pipeline loaded: device=cuda, dtype={mixed}", flush=True)
    return _pipe


def _ensure_multiple_of_64(value: int, *, min_value: int, max_value: int) -> int:
    clamped = max(min_value, min(max_value, int(value)))
    return max(64, int(round(clamped / 64) * 64))


def _decode_b64_image(data: str, mode: str = "RGB") -> Image.Image:
    """Decode base64 (plain or data-URL) into a PIL Image."""
    payload = data
    if data.startswith("data:"):
        parts = data.split(",", 1)
        if len(parts) == 2:
            payload = parts[1]
    raw = base64.b64decode(payload)
    import io
    return Image.open(io.BytesIO(raw)).convert(mode)


def _encode_image_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image to a data-URL base64 string."""
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    raw = buf.getvalue()
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("utf-8")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod handler — expects base64 images, returns base64 result."""
    started = time.perf_counter()
    try:
        inp = job.get("input", {})

        person_b64 = inp.get("person_image_base64")
        cloth_b64 = inp.get("cloth_image_base64")
        mask_b64 = inp.get("mask_image_base64")

        if not person_b64:
            return {"error": "input.person_image_base64 is required"}
        if not cloth_b64:
            return {"error": "input.cloth_image_base64 is required"}
        if not mask_b64:
            return {"error": "input.mask_image_base64 is required"}

        width = _ensure_multiple_of_64(
            int(inp.get("width", 768)), min_value=384, max_value=1024
        )
        height = _ensure_multiple_of_64(
            int(inp.get("height", 1024)), min_value=512, max_value=1536
        )
        steps = max(10, min(80, int(inp.get("steps", 30))))
        guidance = max(1.0, min(8.0, float(inp.get("guidance", 2.5))))
        seed = int(inp.get("seed", -1))

        # Decode images
        person = ImageOps.exif_transpose(_decode_b64_image(person_b64, "RGB"))
        cloth = ImageOps.exif_transpose(_decode_b64_image(cloth_b64, "RGB"))
        mask = ImageOps.exif_transpose(_decode_b64_image(mask_b64, "L"))
        mask = mask.point(lambda p: 255 if p >= 10 else 0, mode="L")

        # Generator for reproducible seeds
        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        pipe = _get_pipe()
        result = pipe(
            image=person,
            condition_image=cloth,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )[0]

        torch.cuda.empty_cache()

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "ok": True,
            "result_image_base64": _encode_image_b64(result),
            "device": "cuda",
            "mixed_precision": _pipe_dtype or "bf16",
            "size": [width, height],
            "steps": steps,
            "guidance": guidance,
            "elapsed_ms": elapsed_ms,
            "engine": "catvton-runpod-cuda",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
