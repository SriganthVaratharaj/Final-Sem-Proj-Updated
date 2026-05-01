"""
backend/ocr/postprocessing.py
Text cleaning, Indic digit normalisation, language grouping, invoice field extraction.
Source: ocr-invoice-system/src/postprocessing.py
"""
# -*- coding: utf-8 -*-
import json
import os
import re
import time

# ── Indic digit → ASCII digit translation tables ──────────────────────────────
DEVANAGARI_DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')
TAMIL_DIGITS      = str.maketrans('௦௧௨௩௪௫௬௭௮௯', '0123456789')
TELUGU_DIGITS     = str.maketrans('౦౧౨౩౪౫౬౭౮౯', '0123456789')
BENGALI_DIGITS    = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
GUJARATI_DIGITS   = str.maketrans('૦૧૨૩૪૫૬૭૮૯', '0123456789')
KANNADA_DIGITS    = str.maketrans('೦೧೨೩೪೫೬೭೮೯', '0123456789')
MALAYALAM_DIGITS  = str.maketrans('൦൧൨൩൪൫൬൭൮൯', '0123456789')

ALL_DIGITS = {
    **DEVANAGARI_DIGITS, **TAMIL_DIGITS, **TELUGU_DIGITS,
    **BENGALI_DIGITS, **GUJARATI_DIGITS, **KANNADA_DIGITS, **MALAYALAM_DIGITS,
}

SCRIPT_KEEP_RANGES = {
    'latin': [(0x0041, 0x005A), (0x0061, 0x007A)],
    'devanagari': [(0x0900, 0x097F)],
    'bengali': [(0x0980, 0x09FF)],
    'gurmukhi': [(0x0A00, 0x0A7F)],
    'gujarati': [(0x0A80, 0x0AFF)],
    'oriya': [(0x0B00, 0x0B7F)],
    'ta': [(0x0B80, 0x0BFF)],
    'te': [(0x0C00, 0x0C7F)],
    'kn': [(0x0C80, 0x0CFF)],
    'ml': [(0x0D00, 0x0D7F)],
    'arabic': [(0x0600, 0x06FF)],
}

TAMIL_TOTAL_KEYWORDS   = ['மொத்தம்', 'கொடுத்தது', 'தொகை', 'கட்டு', 'total', 'subtotal', 'amount', 'grand total', 'ரூ']
TAMIL_INVOICE_KEYWORDS = ['invoice', 'bill', 'பில்', 'ரசீது', 'விலைப்பட்டி']
TAMIL_PHONE_KEYWORDS   = ['phone', 'தொலை', 'கைபேசி', 'அழைப்பு']
TELUGU_TOTAL_KEYWORDS  = ['మొత్తం', 'గ్రాండ్', 'మొత్తము సంఖ్య', 'బిల్లు', 'రూ']
TELUGU_INVOICE_KEYWORDS= ['బిల్లు', 'ఇన్వాయిస్', 'బిల్లు నం', 'రసీదు']
TELUGU_PHONE_KEYWORDS  = ['ఫోన్', 'మొబైల్', 'సంప్రదించండి', 'నంబర్']
TELUGU_DATE_KEYWORDS   = ['తేది', 'తారీఖు', 'తేదీగల', 'వారము', 'నెలలో', 'సం', 'రోజు', 'మాసం']

LANGUAGE_LABELS = {
    'latin': 'Latin/English',
    'devanagari': 'Hindi/Marathi/Nepali (Devanagari)',
    'bengali': 'Bengali/Assamese',
    'gurmukhi': 'Punjabi (Gurmukhi)',
    'gujarati': 'Gujarati',
    'oriya': 'Odia',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'arabic': 'Urdu/Arabic-script',
}

SUPPORTED_OCR_LANGS = {'latin', 'devanagari', 'ta', 'te', 'kn', 'arabic'}
TARGET_LANGS = set(LANGUAGE_LABELS.keys())

OCR_TO_SCRIPT = {
    'hi': 'devanagari',
    'devanagari': 'devanagari',
    'kn': 'kn',
    'ka': 'kn',
    'ta': 'ta',
    'te': 'te',
    'latin': 'latin',
    'arabic': 'arabic',
}


def _normalize_number(text):
    n = text.translate(ALL_DIGITS)
    n = n.replace('।', '').replace('Rs', '').replace('₹', '').replace('రూ', '').strip()
    return n.replace(',', '')


def _to_float(value):
    try:
        return float(value)
    except ValueError:
        cleaned = re.sub(r'[^0-9.\-]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return None


def _extract_invoice_fields(texts, boxes):
    fields = {'invoice_no': None, 'date': None, 'phone': None, 'sub_total': None, 'tax': None, 'grand_total': None}
    number_pattern = re.compile(r'[0-9\u0966-\u096F\u0BE6-\u0BEF]+(?:[.,][0-9\u0966-\u096F\u0BE6-\u0BEF]+)?')
    candidates = []

    y_centers = []
    for box in boxes:
        if box and len(box) == 4:
            y_coords = [p[1] for p in box]
            y_centers.append(sum(y_coords) / len(y_coords))
        else:
            y_centers.append(0)
    max_y = max(y_centers) if y_centers else 0

    for idx, t in enumerate(texts):
        t_stripped = t.strip()
        t_lower = t_stripped.lower()

        if fields['date'] is None:
            dmatch = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', t_stripped)
            if dmatch:
                fields['date'] = dmatch.group(1)
            elif any(k in t_stripped for k in TELUGU_DATE_KEYWORDS):
                fields['date'] = t_stripped

        invoice_keys = ['invoice', 'bill', 'bill no', 'inv.', 'no'] + TAMIL_INVOICE_KEYWORDS + TELUGU_INVOICE_KEYWORDS
        if fields['invoice_no'] is None and any(k in t_lower for k in invoice_keys):
            m = number_pattern.search(t_stripped)
            if m:
                fields['invoice_no'] = _normalize_number(m.group(0))

        phone_keys = ['phone', 'mob', 'tel'] + TAMIL_PHONE_KEYWORDS + TELUGU_PHONE_KEYWORDS
        if fields['phone'] is None and any(k in t_lower for k in phone_keys):
            m = number_pattern.search(t_stripped)
            if m and len(re.sub(r'[^0-9]', '', _normalize_number(m.group(0)))) >= 6:
                fields['phone'] = _normalize_number(m.group(0))

        total_keys = TAMIL_TOTAL_KEYWORDS + TELUGU_TOTAL_KEYWORDS + ['total', 'grand total', 'amount', 'subtotal', 'tax']
        for m in number_pattern.findall(t_stripped):
            n = _to_float(_normalize_number(m))
            if n is not None:
                candidates.append({
                    'value': n, 'text': t_stripped,
                    'y': y_centers[idx] if idx < len(y_centers) else 0,
                    'is_total_label': any(k in t_lower for k in total_keys),
                })

    for c in candidates:
        ln = c['text'].lower()
        val = c['value']
        if 'tax' in ln or 'gst' in ln or 'cgst' in ln or 'sgst' in ln:
            fields['tax'] = val
        elif 'grand' in ln or 'மொத்தம்' in ln or 'మొత్తం' in ln or ('total' in ln and 'sub' not in ln):
            fields['grand_total'] = val
        elif 'sub' in ln or 'தொகை' in ln or ('total' in ln):
            if fields['sub_total'] is None:
                fields['sub_total'] = val
        if fields['phone'] is None:
            digits = re.sub(r'[^0-9]', '', _normalize_number(c['text']))
            if len(digits) == 10:
                fields['phone'] = digits

    if fields['grand_total'] is None and candidates:
        bottom = [c for c in candidates if c['y'] >= max_y * 0.75 and c['value'] < 100000]
        if bottom:
            fields['grand_total'] = max(c['value'] for c in bottom)
    if fields['grand_total'] is None:
        decimal_cands = [c['value'] for c in candidates if c['value'] < 100000 and (not float(c['value']).is_integer() or 100 < c['value'] < 10000)]
        if decimal_cands:
            fields['grand_total'] = max(decimal_cands)
    if fields['grand_total'] is None:
        reasonable = [c['value'] for c in candidates if c['value'] < 100000]
        if reasonable:
            fields['grand_total'] = max(reasonable)

    # Remove fields that were not found so UI only shows what exists
    return {k: v for k, v in fields.items() if v is not None}


def _detect_language_for_text(text):
    counts = {}
    for lang in TARGET_LANGS:
        ranges = SCRIPT_KEEP_RANGES.get(lang, [])
        count = sum(1 for ch in text for s, e in ranges if s <= ord(ch) <= e)
        counts[lang] = count
    best_lang = max(counts, key=counts.get, default=None)
    if best_lang and counts[best_lang] >= 1:
        return best_lang

    if re.search(r'[A-Za-z]', text):
        return 'latin'
    return None


def _group_by_language(texts, boxes, confidences, ocr_metadata=None):
    grouped = {label: [] for label in LANGUAGE_LABELS.values()}
    layout, line_layout = [], []
    metadata = ocr_metadata or {}
    dominant_script = OCR_TO_SCRIPT.get(str(metadata.get('selected_language', '')).lower())

    items = []
    for text, box, confidence in zip(texts, boxes, confidences):
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        items.append({
            'text': text, 'box': box, 'confidence': float(confidence),
            'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
            'ycenter': (ymin + ymax) / 2.0, 'height': max(1.0, ymax - ymin),
        })

    avg_height = sum(i['height'] for i in items) / len(items) if items else 30.0
    line_threshold = max(26.0, avg_height * 0.7)

    items.sort(key=lambda i: i['ycenter'])
    clusters, current = [], []
    for item in items:
        if not current:
            current = [item]
            continue
        center = sum(e['ycenter'] for e in current) / len(current)
        if abs(item['ycenter'] - center) <= line_threshold:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    if current:
        clusters.append(current)

    line_id = 0
    for cluster in clusters:
        cluster.sort(key=lambda i: i['xmin'])
        line_text = ' '.join(i['text'] for i in cluster).strip()
        lang = _detect_language_for_text(line_text) or dominant_script
        label = LANGUAGE_LABELS.get(lang)
        avg_conf = sum(i['confidence'] for i in cluster) / len(cluster)
        entry = {'line_id': line_id, 'text': line_text, 'boxes': [i['box'] for i in cluster], 'confidence': round(avg_conf, 4)}
        if label:
            grouped[label].append(entry)
        line_layout.append({**entry, 'language': label or 'Unknown'})
        for item in cluster:
            layout.append({'line_id': line_id, 'text': item['text'], 'box': item['box'], 'confidence': item['confidence'], 'language': label or 'Unknown'})
        line_id += 1

    summary = []
    for lang, label in LANGUAGE_LABELS.items():
        lang_items = grouped[label]
        if lang_items:
            summary.append({
                'language': label,
                'char_count': sum(len(i['text']) for i in lang_items),
                'avg_confidence': round(sum(i['confidence'] for i in lang_items) / len(lang_items), 4),
            })

    return grouped, summary, layout, line_layout


def postprocess(texts, boxes, confidences, ocr_metadata=None):
    """Clean extracted text and format into structured JSON."""
    entries = [(t.strip(), b, c) for t, b, c in zip(texts, boxes, confidences) if t.strip()]
    cleaned_texts = [t for t, _, _ in entries]
    cleaned_boxes = [b for _, b, _ in entries]
    cleaned_confidences = [c for _, _, c in entries]

    invoice_fields = _extract_invoice_fields(cleaned_texts, cleaned_boxes)
    grouped, summary, layout, line_layout = _group_by_language(cleaned_texts, cleaned_boxes, cleaned_confidences, ocr_metadata)

    return {
        "text": cleaned_texts,
        "boxes": cleaned_boxes,
        "confidence": cleaned_confidences,
        "ocr_metadata": ocr_metadata or {},
        "language_summary": {
            "detected": summary,
            "unsupported_models": sorted(TARGET_LANGS - SUPPORTED_OCR_LANGS),
            "selected_language": (ocr_metadata or {}).get("selected_language"),
            "ocr_mode": (ocr_metadata or {}).get("mode"),
            "language_ranking": (ocr_metadata or {}).get("language_ranking", []),
            "indian_language_support": (ocr_metadata or {}).get("indian_language_support", {}),
            "indian_language_groups": (ocr_metadata or {}).get("indian_language_groups", {}),
            "notes": [
                "Document-level dominant language is preferred to avoid cross-language OCR drift.",
                "Indian language grouping is script-aware (Devanagari, Tamil, Telugu, Kannada, Bengali, Gurmukhi, Gujarati, Odia, Malayalam).",
            ],
        },
        "language_sections": grouped,
        "layout": layout,
        "line_layout": line_layout,
        "invoice_fields": invoice_fields,
    }


def save_json(data, path="output/result.json"):
    """Save structured data to a JSON file atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        try:
            os.replace(temp_path, path)
        except PermissionError:
            fallback = path.replace('.json', f'.{int(time.time())}.json')
            os.replace(temp_path, fallback)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
