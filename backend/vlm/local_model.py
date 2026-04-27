"""
backend/vlm/local_model.py
Local BLIP-base fallback — used only when both LLaVA and BLIP-2 API calls fail.
Requires: pip install -r requirements-ml.txt (torch + transformers)
Source: vlm_llava_project/model/local_model.py
"""
from __future__ import annotations

import logging

from backend.config import VLM_ENABLE_LOCAL_TRANSFORMERS, VLM_LOCAL_MAX_NEW_TOKENS, VLM_LOCAL_MODEL_ID

logger = logging.getLogger(__name__)

_processor = None
_model = None
_device = "cpu"
_uses_chat_template = False


def _load_local_model() -> bool:
    global _processor, _model, _device, _uses_chat_template
    if _processor is not None:
        return True
    if not VLM_ENABLE_LOCAL_TRANSFORMERS:
        logger.info("[local_model] Local transformers fallback disabled via config")
        return False

    try:
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[local_model] Loading local VLM on %s with model %s", _device, VLM_LOCAL_MODEL_ID)

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            _processor = AutoProcessor.from_pretrained(VLM_LOCAL_MODEL_ID)
            _model = AutoModelForImageTextToText.from_pretrained(VLM_LOCAL_MODEL_ID).to(_device)
            _uses_chat_template = hasattr(_processor, "apply_chat_template")
        except Exception as auto_text_exc:
            logger.warning("[local_model] AutoModelForImageTextToText unavailable for %s: %s", VLM_LOCAL_MODEL_ID, auto_text_exc)

            try:
                from transformers import AutoModelForVision2Seq, AutoProcessor

                _processor = AutoProcessor.from_pretrained(VLM_LOCAL_MODEL_ID)
                _model = AutoModelForVision2Seq.from_pretrained(VLM_LOCAL_MODEL_ID).to(_device)
                _uses_chat_template = hasattr(_processor, "apply_chat_template")
            except Exception as auto_vision_exc:
                logger.warning("[local_model] AutoModelForVision2Seq unavailable for %s: %s", VLM_LOCAL_MODEL_ID, auto_vision_exc)
                from transformers import BlipForConditionalGeneration, BlipProcessor

                _processor = BlipProcessor.from_pretrained(VLM_LOCAL_MODEL_ID)
                _model = BlipForConditionalGeneration.from_pretrained(VLM_LOCAL_MODEL_ID).to(_device)
                _uses_chat_template = False

        _model.eval()
        return True
    except (ImportError, OSError, Exception) as exc:
        logger.error("[local_model] CRITICAL: Could not load local ML dependencies (Torch/Transformers). Error: %s", exc)
        return False


def generate(image, prompt: str) -> str:
    """Run BLIP-base locally. Returns generated text or an error string."""
    if not _load_local_model():
        return "LOCAL_ERROR: Model file not loaded (Check logs for download/import errors)"
    try:
        import torch

        if _uses_chat_template and hasattr(_processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            formatted = _processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            formatted = f"Question: {prompt} Answer:"

        inputs = _processor(images=image, text=formatted, return_tensors="pt")
        inputs = {k: (v.to(_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=VLM_LOCAL_MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.15,
            )

        if hasattr(_processor, "batch_decode"):
            decoded = _processor.batch_decode(out, skip_special_tokens=True)
            return decoded[0] if decoded else ""
        return _processor.decode(out[0], skip_special_tokens=True)
    except Exception as exc:
        return f"LOCAL_ERROR: {exc}"
