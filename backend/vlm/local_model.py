"""
backend/vlm/local_model.py
Local BLIP-base fallback — used only when both LLaVA and BLIP-2 API calls fail.
Requires: pip install -r requirements-ml.txt (torch + transformers)
Source: vlm_llava_project/model/local_model.py
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_processor = None
_model = None
_device = "cpu"


def _load_local_model() -> bool:
    global _processor, _model, _device
    if _processor is not None:
        return True
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[local_model] Loading BLIP-base on %s ...", _device)
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(_device)
        return True
    except Exception as exc:
        logger.error("[local_model] Could not load BLIP-base: %s", exc)
        return False


def generate(image, prompt: str) -> str:
    """Run BLIP-base locally. Returns generated text or an error string."""
    if not _load_local_model():
        return "LOCAL_ERROR: Model not loaded"
    try:
        formatted = f"Question: {prompt} Answer:"
        inputs = _processor(image, formatted, return_tensors="pt").to(_device)
        out = _model.generate(**inputs, max_new_tokens=20, do_sample=False, repetition_penalty=1.2)
        return _processor.decode(out[0], skip_special_tokens=True)
    except Exception as exc:
        return f"LOCAL_ERROR: {exc}"
