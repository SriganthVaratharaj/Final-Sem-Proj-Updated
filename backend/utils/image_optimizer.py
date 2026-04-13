"""
backend/utils/image_optimizer.py
Resizes and compresses an image before sending to HF Inference API.
Source: vlm_llava_project/utils/image_optimizer.py
"""
from __future__ import annotations

import io

from PIL import Image


def optimize_image(image_bytes: bytes, max_size: tuple = (500, 500), quality: int = 50) -> bytes:
    """
    Resize and compress image bytes to prevent IncompleteRead errors
    when uploading to the Hugging Face Inference API.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        optimized = buffer.getvalue()
        return optimized
    except Exception as exc:
        print(f"[image_optimizer] Optimization failed ({exc}). Using original bytes.")
        return image_bytes
