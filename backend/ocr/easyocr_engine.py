"""
backend/ocr/easyocr_engine.py
EasyOCR engine running in PARALLEL with PaddleOCR.
Pure pip install — no external binary required.
Supports: Tamil, Telugu, Hindi, Bengali, Gujarati, Kannada, Malayalam, English
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Windows: Fix torch shm.dll DLL search path ──────────────────────────────
# After numpy reinstalls, Windows loses the torch/lib DLL directory from its
# search path, causing WinError 127 (shm.dll not found). Register it explicitly.
def _fix_torch_dll_path():
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists() and sys.platform == "win32":
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

_fix_torch_dll_path()

# EasyOCR language codes for Indic scripts
# Maps PaddleOCR lang codes → EasyOCR lang list
PADDLE_TO_EASY_LANGS = {
    'latin':      ['en'],
    'devanagari': ['hi', 'mr', 'ne', 'en'],  # Hindi, Marathi, Nepali + English
    'ta':         ['ta', 'en'],               # Tamil
    'te':         ['te', 'en'],               # Telugu
    'ka':         ['kn', 'en'],               # Kannada
    'bn':         ['bn', 'en'],               # Bengali
    'bengali':    ['bn', 'en'],
    'gu':         ['en'],                     # Gujarati not supported in EasyOCR, use English fallback
    'gujarati':   ['en'],
    'ml':         ['en'],                     # Malayalam not supported in EasyOCR, use English fallback
    'arabic':     ['ar', 'en'],
}

# Default: try all Indic scripts when language is unknown
ALL_INDIC_LANGS = ['en', 'hi', 'ta', 'te', 'kn', 'bn']

_readers: dict = {}   # cache readers per lang-tuple to avoid reloading models
_easyocr_available: Optional[bool] = None


def is_easyocr_available() -> bool:
    global _easyocr_available
    if _easyocr_available is not None:
        return _easyocr_available
    try:
        import easyocr  # noqa: F401
        _easyocr_available = True
        logger.info("[easyocr] Available")
    except ImportError as e:
        _easyocr_available = False
        logger.warning("[easyocr] Not available: %s", e)
    return _easyocr_available


def _get_reader(lang_list: list[str]):
    """Get or create a cached EasyOCR Reader for the given language list."""
    import easyocr
    key = tuple(sorted(lang_list))
    if key not in _readers:
        logger.info("[easyocr] Loading reader for languages: %s", lang_list)
        # gpu=True uses CUDA if available, falls back to CPU automatically
        _readers[key] = easyocr.Reader(lang_list, gpu=True, verbose=False)
    return _readers[key]


def _resize_for_easyocr(nparr):
    """
    EasyOCR silently fails on very large images.
    Resize so max dimension <= 1500px before passing to readtext.
    """
    import cv2
    h, w = nparr.shape[:2]
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(nparr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return nparr


def run_easyocr(image_bytes: bytes, paddle_lang: str = 'latin') -> tuple:
    """
    Run EasyOCR on image_bytes.
    paddle_lang: PaddleOCR lang code (used to pick appropriate EasyOCR languages).
    Returns: (texts, boxes, confidences, metadata)
    """
    if not is_easyocr_available():
        return [], [], [], {"mode": "easyocr_unavailable"}

    try:
        import numpy as np

        lang_list = PADDLE_TO_EASY_LANGS.get(paddle_lang, ['en'])
        reader = _get_reader(lang_list)

        # Run EasyOCR — accepts bytes or numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        import cv2
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = _resize_for_easyocr(img)
        results = reader.readtext(img, detail=1, paragraph=False)

        texts, boxes, confidences = [], [], []
        for (bbox, text, conf) in results:
            text = text.strip()
            if text and conf > 0.2:
                # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                texts.append(text)
                boxes.append(bbox)
                confidences.append(float(conf))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        logger.info(
            "[easyocr] lang=%s | lines=%d | avg_conf=%.2f",
            lang_list, len(texts), avg_conf
        )

        return texts, boxes, confidences, {
            "mode": "easyocr",
            "selected_language": paddle_lang,
            "easy_langs": lang_list,
            "avg_confidence": round(avg_conf, 3),
        }

    except Exception as e:
        logger.error("[easyocr] Error: %s", e)
        return [], [], [], {"mode": "easyocr_error", "error": str(e)}


def run_easyocr_all_indic(image_bytes: bytes) -> tuple:
    """
    Run EasyOCR across all supported Indic scripts.
    EasyOCR cannot mix most Indic scripts in one reader, so we run each separately
    and merge the unique results.
    Returns: (texts, boxes, confidences, metadata)
    """
    if not is_easyocr_available():
        return [], [], [], {"mode": "easyocr_unavailable"}

    # Each Indic script must run with its own reader (+ English)
    SCRIPT_GROUPS = [
        ['en', 'hi'],   # Hindi/Devanagari
        ['ta', 'en'],   # Tamil
        ['te', 'en'],   # Telugu
        ['kn', 'en'],   # Kannada
        ['bn', 'en'],   # Bengali
    ]

    all_texts, all_boxes, all_confs = [], [], []
    seen = set()

    try:
        import numpy as np
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = _resize_for_easyocr(img)

        for lang_group in SCRIPT_GROUPS:
            try:
                reader = _get_reader(lang_group)
                results = reader.readtext(img, detail=1, paragraph=False)
                for (bbox, text, conf) in results:
                    text = text.strip()
                    if text and conf > 0.25 and text.lower() not in seen:
                        seen.add(text.lower())
                        all_texts.append(text)
                        all_boxes.append(bbox)
                        all_confs.append(float(conf))
            except Exception as eg:
                logger.debug("[easyocr] script group %s error: %s", lang_group, eg)
                continue

        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0
        logger.info("[easyocr] all-indic | lines=%d | avg_conf=%.2f", len(all_texts), avg_conf)

        return all_texts, all_boxes, all_confs, {
            "mode": "easyocr_all_indic",
            "scripts_tried": len(SCRIPT_GROUPS),
            "avg_confidence": round(avg_conf, 3),
        }

    except Exception as e:
        logger.error("[easyocr] run_easyocr_all_indic error: %s", e)
        return [], [], [], {"mode": "easyocr_error", "error": str(e)}


def release_gpu_memory():
    """
    Explicitly release all EasyOCR reader models from VRAM.
    Call this AFTER OCR is done, BEFORE loading the VLM.
    This is critical on 4GB GPUs to avoid OOM when VLM loads.
    """
    global _readers
    import gc
    _readers.clear()
    gc.collect()
    try:
        _fix_torch_dll_path()  # Ensure DLL path is registered before torch ops
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
            logger.info("[easyocr] GPU memory released. Free VRAM: %.0f MB", free_mb)
    except Exception as e:
        logger.debug("[easyocr] GPU release (torch unavailable, skipping): %s", e)
