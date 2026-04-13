"""
backend/ocr/engine.py
PaddleOCR multi-language engine with parallel language scoring.
Supports: Latin/English, Tamil, Hindi/Marathi, Telugu, Kannada.
Source: ocr-invoice-system/src/ocr_engine.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import logging
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.getLogger("ppocr").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

ocr_models = {}

SUPPORTED_LANGUAGES = {
    'latin': 'Latin/English',
    'ta': 'Tamil',
    'hi': 'Hindi/Marathi/Devanagari',
    'devanagari': 'Hindi/Marathi/Devanagari',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ka': 'Kannada',
}

SCRIPT_RANGES = {
    'latin': [(0x0041, 0x005A), (0x0061, 0x007A)],
    'ta': [(0x0B80, 0x0BFF)],
    'hi': [(0x0900, 0x097F)],
    'devanagari': [(0x0900, 0x097F)],
    'kn': [(0x0C80, 0x0CFF)],
    'ka': [(0x0C80, 0x0CFF)],
    'te': [(0x0C00, 0x0C7F)],
    'ml': [(0x0D00, 0x0D7F)],
}

LANGUAGE_CODES = {
    'english': 'latin', 'latin': 'latin', 'en': 'latin',
    'tamil': 'ta', 'ta': 'ta',
    'hindi': 'hi', 'hi': 'hi', 'devanagari': 'devanagari',
    'marathi': 'hi', 'mr': 'hi',
    'telugu': 'te', 'te': 'te',
    'kannada': 'kn', 'kn': 'kn', 'ka': 'ka',
}

MODEL_LANGUAGE_CODES = {
    'latin': 'latin', 'ta': 'ta',
    'hi': 'devanagari', 'devanagari': 'devanagari',
    'te': 'te', 'kn': 'ka', 'ka': 'ka',
}

AUTO_LANGUAGES = ['latin', 'ta', 'hi', 'te', 'kn']


def _normalize_lang(lang):
    key = lang.strip().lower()
    if key not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language '{lang}'.")
    return LANGUAGE_CODES[key]


def _get_model(lang):
    lang = _normalize_lang(lang)
    model_lang = MODEL_LANGUAGE_CODES.get(lang, lang)
    if model_lang not in ocr_models:
        ocr_models[model_lang] = _init_model(model_lang)
    return ocr_models[model_lang]


def _init_model(lang):
    from paddleocr import PaddleOCR
    try:
        return PaddleOCR(lang=lang, use_angle_cls=False, use_space_char=True, show_log=False, enable_mkldnn=True)
    except (tarfile.ReadError, EOFError):
        cache = Path.home() / ".paddleocr" / "whl" / "rec" / lang
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        return PaddleOCR(lang=lang, use_angle_cls=False, use_space_char=True, show_log=False, enable_mkldnn=True)


def _load_models():
    """Pre-load all supported OCR models at startup."""
    for lang in AUTO_LANGUAGES:
        try:
            _get_model(lang)
            logger.info("Model pre-loaded: %s", lang)
        except Exception as exc:
            logger.warning("Could not pre-load model '%s': %s", lang, exc)


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


def _language_score(lang, texts, confidences):
    script_score = _count_script_chars(texts, SCRIPT_RANGES.get(lang, []))
    confidence_score = sum(confidences) if confidences else 0.0
    return script_score + 0.18 * len(texts) + 0.03 * confidence_score


def _select_best_language(results):
    best_lang, best_score = None, -1.0
    for lang, (texts, _, confidences) in results.items():
        score = _language_score(lang, texts, confidences)
        if score > best_score:
            best_score = score
            best_lang = lang
    return best_lang


def _rect_from_box(box):
    x_coords = [p[0] for p in box]
    y_coords = [p[1] for p in box]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


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
    return (
        [d['text'] for d in merged],
        [d['box'] for d in merged],
        [d['confidence'] for d in merged],
        {'mode': 'merged_auto', 'language_counts': lang_counts, 'languages_tried': list(results.keys())},
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


def run_ocr(image, lang=None):
    """
    Run PaddleOCR. If lang is None, all supported languages run in parallel
    and the best result per region is selected.
    Returns: (texts, boxes, confidences, metadata)
    """
    if lang is not None:
        ocr = _get_model(lang)
        selected_lang = _normalize_lang(lang)
        selected_image = _image_variant_for_lang(image, selected_lang)
        texts, boxes, confidences = _flatten_result(ocr.ocr(selected_image, cls=False))
        return texts, boxes, confidences, {'mode': 'single_language', 'selected_language': selected_lang}

    # Auto mode: detect once, recognise in parallel
    from paddleocr.tools.infer.utility import get_rotate_crop_image

    main_ocr = _get_model('latin')
    base_image = _image_variant_for_lang(image, 'default')
    det_res = main_ocr.ocr(base_image, cls=False, rec=False)
    boxes = det_res[0] if det_res and det_res[0] else []

    if not boxes:
        return [], [], [], {'mode': 'merged_auto', 'language_counts': {}, 'languages_tried': list(AUTO_LANGUAGES)}

    results = {}

    def process_candidate(candidate):
        try:
            cand_image = _image_variant_for_lang(image, candidate)
            crops = [get_rotate_crop_image(cand_image, np.array(box, dtype=np.float32)) for box in boxes]
            ocr = _get_model(candidate)
            rec_res = ocr.ocr(crops, det=False, cls=False)
            texts, confidences = [], []
            if rec_res and rec_res[0]:
                for item in rec_res[0]:
                    texts.append(item[0] if isinstance(item, (list, tuple)) else "")
                    confidences.append(float(item[1]) if isinstance(item, (list, tuple)) else 0.0)
            else:
                texts = [""] * len(boxes)
                confidences = [0.0] * len(boxes)
            return candidate, (texts, list(boxes), confidences)
        except Exception:
            return candidate, None

    with ThreadPoolExecutor(max_workers=min(len(AUTO_LANGUAGES), os.cpu_count() or 4)) as executor:
        futures = {executor.submit(process_candidate, c): c for c in AUTO_LANGUAGES}
        for future in as_completed(futures):
            candidate, res = future.result()
            if res:
                results[candidate] = res

    if not results:
        raise RuntimeError("Unable to process image with any supported OCR model.")

    header_lang = _select_header_language(results, base_image.shape[0] if hasattr(base_image, 'shape') else 0)
    if header_lang:
        texts, boxes_list, confidences = results[header_lang]
        return texts, boxes_list, confidences, {
            'mode': 'header_guided_single_language',
            'selected_language': header_lang,
            'languages_tried': list(results.keys()),
        }

    texts, boxes_list, confidences, metadata = _merge_auto_results(results)
    if texts:
        return texts, boxes_list, confidences, metadata

    best_lang = _select_best_language(results)
    texts, boxes_list, confidences = results[best_lang]
    return texts, boxes_list, confidences, {
        'mode': 'single_best_language',
        'selected_language': best_lang,
        'languages_tried': list(results.keys()),
    }
