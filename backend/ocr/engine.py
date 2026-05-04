"""
backend/ocr/engine.py
PaddleOCR multi-language engine with parallel language scoring.
Supports Indic script-aware OCR with dominant-language enforcement.
Source: ocr-invoice-system/src/ocr_engine.py
"""
import os
import sys
from pathlib import Path

# ── Windows DLL Fix ──────────────────────────────────────────────────────────
def _fix_dll_paths():
    if sys.platform == "win32":
        try:
            import torch
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

_fix_dll_paths()

# Paddle is now isolated via subprocess, so we don't need to force CPU mode 
# globally. Doing so would break the VLM subprocess which inherits os.environ.

import logging
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import cv2
import json
import subprocess
import tempfile

from backend.config import OCR_FORCE_DOMINANT_LANGUAGE, PADDLE_AUTO_LANGUAGES, INTERNAL_MODEL_API_KEY
from backend.vlm.gguf_engine import query_local_paddle_vl

logging.getLogger("ppocr").setLevel(logging.ERROR)

# PaddleOCR is now handled via subprocess to avoid pybind11 registration conflicts on Windows.
PADDLE_AVAILABLE = True 

def _get_paddleocr_class():
    return None # Not used in main process anymore

logger = logging.getLogger(__name__)

ocr_models = {}

MODEL_SUPPORTED_CODES = {'latin', 'devanagari', 'ta', 'te', 'ka', 'arabic', 'bengali', 'gujarati'}

SUPPORTED_LANGUAGES = {
    'latin': 'Latin/English',
    'devanagari': 'Hindi/Marathi/Nepali/Sanskrit/Devanagari',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ka': 'Kannada',
    'arabic': 'Urdu/Arabic-script',
}

SCRIPT_RANGES = {
    'latin': [(0x0041, 0x005A), (0x0061, 0x007A)],
    'devanagari': [(0x0900, 0x097F)],
    'bengali': [(0x0980, 0x09FF)],
    'bn': [(0x0980, 0x09FF)],
    'gurmukhi': [(0x0A00, 0x0A7F)],
    'gujarati': [(0x0A80, 0x0AFF)],
    'gu': [(0x0A80, 0x0AFF)],
    'ta': [(0x0B80, 0x0BFF)],
    'oriya': [(0x0B00, 0x0B7F)],
    'te': [(0x0C00, 0x0C7F)],
    'ka': [(0x0C80, 0x0CFF)],
    'ml': [(0x0D00, 0x0D7F)],
    'arabic': [(0x0600, 0x06FF)],
}

BENGALI_RANGE = (0x0980, 0x09FF)

LANGUAGE_CODES = {
    'english': 'latin', 'latin': 'latin', 'en': 'latin',
    'arabic': 'arabic',
    'tamil': 'ta', 'ta': 'ta',
    'hindi': 'devanagari', 'hi': 'devanagari', 'devanagari': 'devanagari',
    'marathi': 'devanagari', 'mr': 'devanagari',
    'nepali': 'devanagari', 'sanskrit': 'devanagari', 'konkani': 'devanagari',
    'maithili': 'devanagari', 'dogri': 'devanagari', 'bhojpuri': 'devanagari',
    'awadhi': 'devanagari', 'magahi': 'devanagari',
    'telugu': 'te', 'te': 'te',
    'kannada': 'ka', 'kn': 'ka', 'ka': 'ka',
    'urdu': 'arabic',

    # Script aliases for broader Indian language coverage in metadata.
    'bengali': 'bengali', 'bn': 'bengali', 'assamese': 'bengali', 'as': 'bengali',
    'punjabi': 'gurmukhi', 'pa': 'gurmukhi',
    'gujarati': 'gujarati', 'gu': 'gujarati',
    'odia': 'oriya', 'oriya': 'oriya', 'or': 'oriya',
    'malayalam': 'ml', 'ml': 'ml',
}

MODEL_LANGUAGE_CODES = {
    'latin': 'latin',
    'devanagari': 'devanagari',
    'ta': 'ta',
    'te': 'te',
    'ka': 'ka',
    'arabic': 'arabic',
    'bengali': 'devanagari', # Fallback to Devanagari if Bengali missing (better than crash)
    'bn': 'devanagari',
    'gujarati': 'devanagari',
    'gu': 'devanagari',

    # Indic scripts supported.
    'bengali': 'bn',
    'bn': 'bn',
    'gurmukhi': None,
    'gujarati': 'gu',
    'gu': 'gu',
    'oriya': None,
    'ml': None,
}

AUTO_LANGUAGES = list(PADDLE_AUTO_LANGUAGES or ['latin', 'devanagari', 'ta', 'te', 'ka'])
ACTIVE_AUTO_LANGUAGES = list(AUTO_LANGUAGES)

INDIAN_LANGUAGE_GROUPS = {
    'deva_family': ['hindi', 'marathi', 'nepali', 'sanskrit', 'konkani', 'maithili', 'dogri', 'bhojpuri'],
    'tamil': ['tamil'],
    'telugu': ['telugu'],
    'kannada': ['kannada'],
    'malayalam': ['malayalam'],
    'bengali_family': ['bengali', 'assamese'],
    'gurmukhi_family': ['punjabi'],
    'gujarati': ['gujarati'],
    'odia': ['odia'],
    'urdu': ['urdu'],
}


def _resolve_auto_languages(requested: list[str]) -> list[str]:
    resolved = []
    for item in requested:
        try:
            normalized = _normalize_lang(item)
        except Exception:
            logger.warning("Ignoring unsupported OCR language setting: %s", item)
            continue
        model_code = MODEL_LANGUAGE_CODES.get(normalized)
        if model_code and model_code not in resolved:
            resolved.append(model_code)
    return resolved or ['latin']


def _language_support_report() -> dict:
    directly_supported = []
    script_supported_only = []
    unsupported = []

    for name in sorted({k for k in LANGUAGE_CODES if len(k) > 2 and k.isalpha()}):
        normalized = LANGUAGE_CODES.get(name)
        model_code = MODEL_LANGUAGE_CODES.get(normalized)
        if model_code:
            if normalized in {'devanagari', 'bengali', 'gurmukhi', 'gujarati', 'oriya', 'ml'}:
                script_supported_only.append(name)
            else:
                directly_supported.append(name)
        else:
            unsupported.append(name)

    return {
        'direct_model_support': sorted(set(directly_supported)),
        'script_family_support': sorted(set(script_supported_only)),
        'unsupported_in_current_paddle_build': sorted(set(unsupported)),
    }


def _normalize_lang(lang):
    if not isinstance(lang, str):
        return str(lang)
    key = lang.strip().lower()
    if key in MODEL_SUPPORTED_CODES:
        return key
    if key not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language '{lang}'.")
    return LANGUAGE_CODES[key]


def _get_model(lang):
    lang = _normalize_lang(lang)
    model_lang = MODEL_LANGUAGE_CODES.get(lang, lang)
    if not model_lang:
        raise ValueError(f"No PaddleOCR model available for language/script '{lang}'.")
    if model_lang not in ocr_models:
        ocr_models[model_lang] = _init_model(model_lang)
    return ocr_models[model_lang]


def _init_model(lang):
    OCRClass = _get_paddleocr_class()
    if not OCRClass:
        raise ImportError("PaddleOCR not available")
    try:
        return OCRClass(lang=lang, use_gpu=False, use_angle_cls=True, use_space_char=True, show_log=False, enable_mkldnn=True)
    except Exception:
        # Retry with cache cleanup
        cache = Path.home() / ".paddleocr" / "whl" / "rec" / lang
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        return OCRClass(lang=lang, use_gpu=False, use_angle_cls=True, use_space_char=True, show_log=False, enable_mkldnn=True)


def _load_models():
    """Pre-load all supported OCR models at startup."""
    global ACTIVE_AUTO_LANGUAGES

    configured_auto = _resolve_auto_languages(AUTO_LANGUAGES)
    available = []
    for lang in configured_auto:
        try:
            _get_model(lang)
            logger.info("Model pre-loaded: %s", lang)
            available.append(lang)
        except Exception as exc:
            logger.warning("Could not pre-load model '%s': %s", lang, exc)

    if available:
        ACTIVE_AUTO_LANGUAGES = available
    else:
        ACTIVE_AUTO_LANGUAGES = ['latin']

    logger.info("Active OCR auto languages: %s", ", ".join(ACTIVE_AUTO_LANGUAGES))


def _count_script_chars(texts, ranges):
    count = 0
    for text in texts:
        for ch in text:
            code = ord(ch)
            for start, end in ranges:
                if start <= code <= end:
                    count += 1
                    break
    return count


def _has_bengali(text):
    for ch in text:
        if BENGALI_RANGE[0] <= ord(ch) <= BENGALI_RANGE[1]:
            return True
    return False


def _language_score(lang, texts, confidences):
    script_score = _count_script_chars(texts, SCRIPT_RANGES.get(lang, []))
    confidence_score = sum(confidences) if confidences else 0.0
    score = script_score + 0.18 * len(texts) + 0.03 * confidence_score
    
    # Massive boost for Indian languages. If an Indic model finds actual Indic characters, 
    # it must crush the 'latin' model which often hallucinates gibberish English letters.
    if lang != 'latin' and script_score > 5:
        score *= 10
        
    return score


def _select_best_language(results):
    best_lang, best_score = None, -1.0
    for lang, (texts, _, confidences) in results.items():
        score = _language_score(lang, texts, confidences)
        if score > best_score:
            best_score = score
            best_lang = lang
    return best_lang


def _rank_document_languages(results):
    ranked = []
    for lang, (texts, _, confidences) in results.items():
        non_empty = [t for t in texts if str(t).strip()]
        if not non_empty:
            continue
        score = _language_score(lang, non_empty, confidences)
        ranked.append((lang, score, len(non_empty)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _select_dominant_document_language(results):
    ranked = _rank_document_languages(results)
    if not ranked:
        return None, ranked

    best_lang, best_score, best_lines = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_lines < 2:
        return None, ranked

    if second_score <= 0:
        return best_lang, ranked

    if OCR_FORCE_DOMINANT_LANGUAGE and best_score >= second_score * 1.05:
        return best_lang, ranked

    if best_score >= second_score * 1.18:
        return best_lang, ranked

    return None, ranked


def _rect_from_box(box):
    x_coords = [p[0] for p in box]
    y_coords = [p[1] for p in box]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def _sort_by_reading_order(texts, boxes, confidences):
    rows = []
    for text, box, confidence in zip(texts, boxes, confidences):
        if not box:
            continue
        xmin, ymin, xmax, ymax = _rect_from_box(box)
        rows.append({
            'text': text,
            'box': box,
            'confidence': confidence,
            'ycenter': (ymin + ymax) / 2.0,
            'xmin': xmin,
            'xmax': xmax,
            'height': max(1.0, ymax - ymin),
        })

    if not rows:
        return texts, boxes, confidences

    rows.sort(key=lambda r: r['ycenter'])
    avg_height = sum(r['height'] for r in rows) / len(rows)
    line_threshold = max(18.0, avg_height * 0.7)

    clusters, current = [], [rows[0]]
    for row in rows[1:]:
        center = sum(item['ycenter'] for item in current) / len(current)
        if abs(row['ycenter'] - center) <= line_threshold:
            current.append(row)
        else:
            clusters.append(current)
            current = [row]
    clusters.append(current)

    ordered = []
    for cluster in clusters:
        # Sort current line by X-coordinate
        sorted_cluster = sorted(cluster, key=lambda r: r['xmin'])
        
        # Physical Spacing Reconstruction
        line_text = ""
        for i, item in enumerate(sorted_cluster):
            if i > 0:
                prev = sorted_cluster[i-1]
                gap = item['xmin'] - prev['xmax']
                # Estimate char width based on height (roughly 0.5 ratio)
                char_width = prev['height'] * 0.45
                
                if gap > char_width * 1.5:
                    num_spaces = min(20, int(gap / char_width))
                    ordered.append({
                        'text': " " * num_spaces,
                        'box': [], 'confidence': 1.0, 'xmin': prev['xmax'] + 1, 'ycenter': prev['ycenter']
                    })
            ordered.append(item)

    return (
        [r['text'] for r in ordered],
        [r['box'] for r in ordered],
        [r['confidence'] for r in ordered],
    )



def _script_char_count(text, lang):
    return _count_script_chars([text], SCRIPT_RANGES.get(lang, []))


def _letterlike_chars(text):
    return [ch for ch in text if ch.isalpha()]


def _digit_chars(text):
    return [ch for ch in text if ch.isdigit()]


def _noise_chars(text):
    allowed = set(" -:.,/()₹")
    return [ch for ch in text if not (ch.isalnum() or ch.isspace() or ch in allowed)]


def _detection_score(lang, text, confidence):
    letters = _letterlike_chars(text)
    digits = _digit_chars(text)
    letter_count = len(letters)
    digit_count = len(digits)
    script_count = _script_char_count(text, lang)
    noise_count = len(_noise_chars(text))

    ratio_bonus = 0.0
    if letter_count == 0:
        ratio_bonus = 0.8 if digit_count else 0.0
    else:
        ratio_bonus = script_count / max(letter_count, 1)

    other_penalty = sum(
        _script_char_count(text, sl) for sl in SCRIPT_RANGES if sl != lang
    )
    return (
        (confidence * 4.0)
        + (ratio_bonus * 5.0)
        + min(len(text.strip()), 24) * 0.06
        + min(digit_count, 8) * 0.08
        - other_penalty * 0.18
        - noise_count * 0.35
    )


def _merge_auto_results(results):
    detections = []
    lang_counts = {}
    for lang, (texts, boxes, confidences) in results.items():
        for text, box, confidence in zip(texts, boxes, confidences):
            text = text.strip()
            if not text:
                continue
            xmin, ymin, xmax, ymax = _rect_from_box(box)
            detections.append({
                'lang': lang, 'text': text, 'box': box,
                'confidence': float(confidence),
                'score': _detection_score(lang, text, float(confidence)),
                'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                'ycenter': (ymin + ymax) / 2.0,
                'height': max(1.0, ymax - ymin),
            })

    if not detections:
        return [], [], [], {'mode': 'merged_auto', 'language_counts': {}, 'languages_tried': list(results.keys())}

    detections.sort(key=lambda d: d['ycenter'])
    avg_height = sum(d['height'] for d in detections) / len(detections)
    line_threshold = max(26.0, avg_height * 0.7)

    clusters, current = [], [detections[0]]
    for item in detections[1:]:
        cluster_center = sum(e['ycenter'] for e in current) / len(current)
        if abs(item['ycenter'] - cluster_center) <= line_threshold:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)

    merged = []
    for cluster in clusters:
        lang_groups = {}
        for item in cluster:
            lang_groups.setdefault(item['lang'], []).append(item)
        best_lang = max(lang_groups, key=lambda l: sum(e['score'] for e in lang_groups[l]))
        selected = sorted(lang_groups[best_lang], key=lambda d: d['xmin'])
        for item in selected:
            lang_counts[item['lang']] = lang_counts.get(item['lang'], 0) + 1
        merged.extend(selected)

    merged.sort(key=lambda d: (d['ycenter'], d['xmin']))
    
    # Check for Bengali characters in any of the merged results
    has_bengali = any(any(0x0980 <= ord(ch) <= 0x09FF for ch in d['text']) for d in merged)
    
    return (
        [d['text'] for d in merged],
        [d['box'] for d in merged],
        [d['confidence'] for d in merged],
        {
            'mode': 'merged_auto', 
            'language_counts': lang_counts, 
            'languages_tried': list(results.keys()),
            'script_detected': 'bengali' if has_bengali else None
        },
    )


def _select_header_language(results, image_height):
    if image_height <= 0:
        return None
    header_limit = image_height * 0.45
    ranked = []
    for lang, (texts, boxes, confidences) in results.items():
        header_texts, header_confs = [], []
        for text, box, conf in zip(texts, boxes, confidences):
            xmin, ymin, xmax, ymax = _rect_from_box(box)
            if (ymin + ymax) / 2.0 <= header_limit:
                header_texts.append(text)
                header_confs.append(conf)
        if header_texts:
            ranked.append((lang, _language_score(lang, header_texts, header_confs)))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[1], reverse=True)
    best_lang, best_score = ranked[0]
    next_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score >= 2.5 and best_score >= next_score * 1.15:
        return best_lang
    return None


def _flatten_result(result):
    texts, boxes, confidences = [], [], []
    if result and result[0]:
        for line in result[0]:
            box, text_info = line[0], line[1]
            texts.append(text_info[0])
            boxes.append(box)
            confidences.append(float(text_info[1]))
    return texts, boxes, confidences


def _image_variant_for_lang(image, lang):
    if isinstance(image, dict):
        return image.get(lang, image.get('default'))
    return image


def run_ocr_gguf(image_bytes: bytes):
    """
    Fallback OCR using local Paddle-VL GGUF model.
    Since it's a VLM, it returns text but not precise bounding boxes.
    """
    logger.info("[ocr] Running fallback GGUF OCR (Paddle-VL)...")
    prompt = "Extract all text from this image exactly as it appears. Maintain reading order."
    text = query_local_paddle_vl(image_bytes, prompt)
    
    if "LOCAL_ERROR" in text:
        return [], [], [], {"mode": "error", "error": text}
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Mock boxes since VLM doesn't provide them easily
    return lines, [[] for _ in lines], [1.0 for _ in lines], {"mode": "gguf_fallback", "model": "Paddle-VL-GGUF"}


def run_ocr(image, lang=None, image_bytes: bytes = None, filename: str = ""):
    """
    Run PaddleOCR in an isolated subprocess to prevent VRAM fragmentation.
    Reverted back to PaddleOCR from Surya OCR due to Surya's severe hallucination 
    issues on Indic scripts (predicting Bengali/Math tags for Devanagari text).
    """
    try:
        import subprocess
        import tempfile
        import cv2
        import sys
        
        logger.info("[ocr] Running PaddleOCR in subprocess...")
        
        # Save image for subprocess
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            # If image is a dict, get default
            img_to_save = image if not isinstance(image, dict) else image.get('default', image.get('latin'))
            cv2.imwrite(tmp_path, img_to_save)
            
        # Default to a safe multilingual paddle engine (latin + devanagari if possible)
        # We will use 'hi' as the default for Indian invoices if no lang is specified,
        # since PaddleOCR's 'hi' model covers Hindi + English well.
        target_lang = _normalize_lang(lang) if lang else "hi"
            
        sub_script = Path(__file__).parent / "paddle_subprocess.py"
        cmd = [sys.executable, str(sub_script), tmp_path, target_lang]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        os.unlink(tmp_path)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if "error" in data:
                logger.error(f"[ocr] Paddle Subprocess Error: {data['error']}")
                raise Exception(data["error"])
                
            texts, boxes, confidences = data["texts"], data["boxes"], data["confidences"]
        else:
            logger.error(f"[ocr] Paddle Subprocess Crashed. Stderr: {result.stderr}")
            raise Exception("Subprocess crashed")
            
        texts, boxes, confidences = _sort_by_reading_order(texts, boxes, confidences)
        is_bengali = any(_has_bengali(t) for t in texts)
        
        return texts, boxes, confidences, {
            'mode': 'paddle_subprocess', 
            'selected_language': target_lang,
            'script_detected': 'bengali' if is_bengali else None,
            'indian_language_support': _language_support_report(),
            'indian_language_groups': INDIAN_LANGUAGE_GROUPS,
        }
    except Exception as exc:
        logger.warning("[ocr] PaddleOCR failed: %s. Trying GGUF fallback...", exc)
        if image_bytes:
            return run_ocr_gguf(image_bytes)
        raise exc
