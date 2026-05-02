from __future__ import annotations
"""
## SINGLE_PASS_V2_CHAIN_OF_THOUGHT
backend/vlm/vlm_model.py
"""
import json, logging, re
from pathlib import Path
from typing import Any
from backend.config import VLM_REQUIRED_FIELDS, INTERNAL_MODEL_API_KEY
from backend.vlm.gguf_engine import query_local_llava
from backend.utils.image_enhancer import split_for_extraction

logger = logging.getLogger(__name__)

ALL_IN_ONE_PROMPT_TEMPLATE = """
You are an invoice OCR engine. Your ONLY job is to read every piece of text visible in this image, from TOP to BOTTOM, LEFT to RIGHT.

The invoice may be in English, Hindi, Bengali, Tamil, Telugu, Kannada, Gujarati, or a mix.

{reference_alphabets}

{ocr_context}

CRITICAL RULES:
1. Read EVERY line you can see in the image, in the order it appears top-to-bottom.
2. Do NOT skip any line — even if you don't understand the label.
3. Copy each label AND its value EXACTLY as seen. Do NOT translate.
4. For numbers (IDs, amounts, dates): copy digit-by-digit. Do NOT guess or truncate.
5. If text is white on a dark/colored background, look carefully — it is still readable.
6. Do NOT repeat any line.
7. If you cannot read a specific word clearly, write [unclear] for that word only.

OUTPUT FORMAT — one line per visible text row:
```
LINE 1: <exact text of first line>
LINE 2: <exact text of second line>
LINE 3: <exact text of third line>
...
```

Also, at the end, output a separate TOTALS section:
```
TOTAL AMOUNT: <value>
```
"""

# Unicode ranges for scripts VLM cannot reliably read
_BENGALI_RANGE = (0x0980, 0x09FF)
_GUJARATI_RANGE = (0x0A80, 0x0AFF)
_UNSUPPORTED_RANGES = [_BENGALI_RANGE, _GUJARATI_RANGE]

def _is_unsupported_script(text: str) -> bool:
    """Return True if text contains Bengali/Gujarati chars that VLM can't read."""
    for ch in text:
        code = ord(ch)
        for start, end in _UNSUPPORTED_RANGES:
            if start <= code <= end:
                return True
    return False

def _clean_output(text: str) -> dict | None:
    if not text: return None
    # Strip markdown code fences
    text = text.replace("```json", "").replace("```", "").strip()
    # Handle escaped JSON inside a string (e.g. \"key\": \"val\")
    if text.startswith('"') or '\\"' in text:
        try:
            text = json.loads(f'"{text}"') if text.startswith('"') else text.replace('\\"', '"')
        except Exception:
            pass
    # We explicitly asked for Markdown, NOT JSON. 
    # Do not parse {}, just return the full text for the Digital Twin.
    return {
        "full_extraction": text.strip(),
        "is_comprehensive": True,
        "_source": "vlm_descriptive_list"
    }

# ── POINT 3: Quick Language Scan Prompt ──────────────────────────────────────
_LANG_SCAN_PROMPT = """
Look at this image. What is the MAIN language/script of the text?
Reply with ONLY one word from this list:
english, hindi, bengali, tamil, telugu, kannada, gujarati, marathi, odia, malayalam, punjabi, urdu, mixed

Do not explain. Do not add punctuation. Just one word.
"""

# Per-language prompt additions injected in the 2nd pass
_LANG_SPECIFIC_RULES: dict[str, str] = {
    "hindi":     "The text uses Devanagari script. Read matras (vowel marks) carefully.",
    "marathi":   "The text uses Devanagari script. Read matras (vowel marks) carefully.",
    "bengali":   "The text uses Bengali script. Watch for conjunct consonants.",
    "tamil":     "The text uses Tamil script. Letters have circular curves.",
    "telugu":    "The text uses Telugu script. Characters have rounded shapes.",
    "kannada":   "The text uses Kannada script.",
    "gujarati":  "The text uses Gujarati script. Similar to Devanagari but no top bar.",
    "malayalam": "The text uses Malayalam script.",
    "odia":      "The text uses Odia script.",
    "punjabi":   "The text uses Gurmukhi script.",
    "urdu":      "The text uses Urdu/Nastaliq script (right-to-left).",
    "english":   "",
    "mixed":     "The invoice contains multiple languages/scripts. Extract all visible text.",
}


def _quick_language_scan(image_bytes: bytes, model_type: str = "minicpm") -> str:
    """
    [POINT 3 - Prompt Feedback Loop Pass 1]
    Lightweight VLM call to detect invoice language before the full extraction.
    Returns a language key from _LANG_SPECIFIC_RULES, defaults to 'mixed' on failure.
    """
    try:
        raw = query_local_llava(image_bytes, _LANG_SCAN_PROMPT.strip(), model_type=model_type)
        if not raw:
            return "mixed"
        # Clean and normalize
        lang = raw.strip().lower().split()[0] if raw.strip() else "mixed"
        lang = re.sub(r'[^a-z]', '', lang)  # letters only
        if lang not in _LANG_SPECIFIC_RULES:
            lang = "mixed"
        logger.info("[vlm] Language scan result: '%s'", lang)
        return lang
    except Exception as e:
        logger.warning("[vlm] Language scan failed: %s", e)
        return "mixed"


def _strip_hallucinations(text: str) -> str:
    """
    Detect and remove hallucinated repeated patterns from VLM output.
    e.g. '33 33 33 33 33...' or 'word word word word...'
    Returns empty string if the whole value is junk.
    """
    if not text or len(text) < 10:
        return text
    words = text.split()
    if len(words) < 5:
        return text
    # Check if >60% of words are the same token
    from collections import Counter
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    if top_count / len(words) > 0.6:
        return ""
    # Repeating digit pattern: "33 33 33 33..."
    if re.match(r'^(\d+\s+){5,}', text.strip()):
        return ""
    return text


def _postprocess_fields(data: dict) -> dict:
    """Strip hallucinations and deduplicate repeated lines from all string fields."""
    if not isinstance(data, dict):
        return data
    
    result = {}
    for k, v in data.items():
        if isinstance(v, str):
            v = _strip_hallucinations(v)
            if v:
                v = _dedup_markdown(v)
        result[k] = v
    return result


def _dedup_markdown(text: str) -> str:
    """
    Remove duplicate rows/lines from markdown output.
    The VLM often repeats the same table row 3-5x for empty table cells.
    Strategy: Within a table block, keep only unique rows (preserve header + separator).
    For non-table lines, deduplicate consecutive identical lines.
    """
    if not text:
        return text
    
    lines = text.split('\n')
    out_lines = []
    in_table = False
    seen_table_rows: set = set()
    prev_line = None
    
    for line in lines:
        stripped = line.strip()
        
        # Detect table start/end
        if stripped.startswith('|') and '---' not in stripped:
            in_table = True
        elif in_table and not stripped.startswith('|'):
            in_table = False
            seen_table_rows.clear()  # reset for next table
        
        if in_table and stripped.startswith('|'):
            # Header row or separator — always keep
            if '---' in stripped or stripped == out_lines[-1].strip() if out_lines else False:
                out_lines.append(line)
                continue
            # Normalize row for dedup (strip spaces around pipes)
            norm = re.sub(r'\s*\|\s*', '|', stripped)
            if norm not in seen_table_rows:
                seen_table_rows.add(norm)
                out_lines.append(line)
            # else: skip duplicate row
        else:
            # Non-table: skip consecutive identical lines
            if stripped != prev_line:
                out_lines.append(line)
            prev_line = stripped
    
    return '\n'.join(out_lines)


# Unicode block → language file mapping for smart script detection
_SCRIPT_TO_LANG = [
    ((0x0900, 0x097F), ["hindi", "marathi", "maithili"]),      # Devanagari
    ((0x0980, 0x09FF), ["bengali", "assamese"]),                # Bengali-Assamese
    ((0x0A00, 0x0A7F), ["punjabi"]),                           # Gurmukhi
    ((0x0A80, 0x0AFF), ["gujarati"]),                          # Gujarati
    ((0x0B00, 0x0B7F), ["odia"]),                              # Odia
    ((0x0B80, 0x0BFF), ["tamil"]),                             # Tamil
    ((0x0C00, 0x0C7F), ["telugu"]),                            # Telugu
    ((0x0C80, 0x0CFF), ["kannada"]),                           # Kannada
    ((0x0D00, 0x0D7F), ["malayalam"]),                         # Malayalam
    ((0x0600, 0x06FF), ["urdu", "sindhi"]),                    # Arabic/Urdu
]

def _detect_scripts(text: str) -> list[str]:
    """Return list of language names whose scripts appear in the text."""
    detected = set()
    for ch in text:
        code = ord(ch)
        for (start, end), langs in _SCRIPT_TO_LANG:
            if start <= code <= end:
                detected.update(langs)
    return list(detected)


# Script family groups — scripts within same family are OK together
# Scripts from DIFFERENT families mixed = cross-script contamination = garbage OCR
_SCRIPT_FAMILIES = [
    {"devanagari", "hindi", "marathi", "maithili"},  # Devanagari family
    {"bengali", "assamese"},                          # Bengali family
    {"telugu"},
    {"kannada"},
    {"malayalam"},
    {"gujarati"},
    {"odia"},
    {"punjabi"},
    {"tamil"},
    {"urdu", "sindhi"},
]

def _count_distinct_script_families(detected_langs: list[str]) -> int:
    """Count how many distinct script families appear in the detected languages.
    If > 1 family detected, OCR has cross-script contamination (garbage).
    """
    families_hit = 0
    for family in _SCRIPT_FAMILIES:
        if any(lang in family for lang in detected_langs):
            families_hit += 1
    return families_hit

def _load_reference_alphabets(ocr_hint: str = "", image_langs: list = None, inject_confusion_map: bool = False) -> str:
    """
    Load reference alphabets for detected scripts.
    inject_confusion_map=True  → full confusion map + bill keywords (for stylized fonts)
    inject_confusion_map=False → compact alphabets only (normal case, saves ~200 tokens)
    VRAM-safe: hard cap enforced on all sections.
    """
    alphabets_dir = Path(__file__).parent.parent / "language_alphabets"
    if not alphabets_dir.exists():
        return ""

    # Detect from OCR hint or use explicitly passed langs
    if image_langs:
        detected_langs = image_langs
    else:
        detected_langs = _detect_scripts(ocr_hint) if ocr_hint else []

    if not detected_langs:
        logger.info("[vlm] No Indic script detected — skipping alphabet injection.")
        return ""

    # Priority order for injection
    priority_order = [
        "bengali", "assamese", "hindi", "marathi", "tamil",
        "telugu", "kannada", "gujarati", "malayalam", "odia", "punjabi", "maithili"
    ]
    selected = [l for l in priority_order if l in detected_langs][:2]  # max 2 scripts
    if not selected:
        selected = detected_langs[:1]

    logger.info("[vlm] Detected scripts: %s — injecting targeted alphabets.", selected)

    ref_sections = []
    for file in alphabets_dir.glob("*.txt"):
        if file.stem.lower() not in selected:
            continue
        content = file.read_text(encoding="utf-8")
        
        # Extract sections: standard alphabets + confusion map
        standard_chars = []
        confusion_lines = []
        keywords_lines = []
        current_section = None

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "=== STANDARD ALPHABETS ===" in stripped:
                current_section = "standard"
            elif "=== FONT VARIANT CONFUSION MAP ===" in stripped:
                current_section = "confusion"
            elif "=== COMMON BILL KEYWORDS" in stripped:
                current_section = "keywords"
            elif stripped.startswith("LANGUAGE:") or stripped.startswith("SCRIPT_RANGE:"):
                continue
            elif current_section == "standard" and not stripped.startswith("#"):
                standard_chars.append(stripped)
            elif current_section == "confusion" and not stripped.startswith("#") and "→" in stripped:
                confusion_lines.append(stripped)
            elif current_section == "keywords" and not stripped.startswith("#") and ":" in stripped:
                keywords_lines.append(stripped)

        # Build compact injection (hard token budget)
        lang_name = file.stem.capitalize()
        section = f"[{lang_name} Script Reference]\n"
        
        # Standard chars — always included, first 150 chars
        chars_text = " ".join(standard_chars)[:150]
        section += f"Alphabets: {chars_text}\n"
        
        # Confusion map + keywords — ONLY when stylized font detected (token budget mitigation)
        if inject_confusion_map:
            if confusion_lines:
                section += "Font Confusables (decorative→actual): "
                section += " | ".join(confusion_lines[:8]) + "\n"
            if keywords_lines:
                section += "Bill Keywords: "
                section += " | ".join(keywords_lines[:5]) + "\n"

        ref_sections.append(section)


    if not ref_sections:
        return ""

    full_ref = "[SCRIPT REFERENCE — Use to correct stylized font misreads]\n"
    full_ref += "\n".join(ref_sections)
    full_ref += "[/SCRIPT REFERENCE]\n"
    return full_ref



def vlm_extract_all(image_bytes: bytes, correction_rules: str = "", ocr_hint: str = "", filename: str = "") -> dict:
    try:
        image_id = Path(filename).stem if filename else "current_invoice"

        # ── POINT 1: Divide & Conquer 2.0 ─────────────────────────────────────
        # Split large/tall/dual images into segments, extract each, merge results.
        segments = split_for_extraction(image_bytes)
        if len(segments) > 1:
            logger.info("[vlm] D&C: %d segments detected. Running per-segment extraction.", len(segments))
            all_lines = []
            for i, seg_bytes in enumerate(segments):
                seg_result = _extract_single_segment(
                    seg_bytes, ocr_hint=ocr_hint,
                    filename=f"{image_id}_seg{i+1}"
                )
                seg_text = seg_result.get("fields", {}).get("full_extraction", "")
                if seg_text:
                    all_lines.append(f"--- SEGMENT {i+1} ---\n{seg_text}")
            merged_text = "\n".join(all_lines)
            return {
                "fields": {
                    "full_extraction": merged_text,
                    "is_comprehensive": True,
                    "_source": "divide_and_conquer"
                },
                "is_invoice": True,
                "_source": "divide_and_conquer"
            }

        # Single segment — standard path
        return _extract_single_segment(image_bytes, ocr_hint=ocr_hint, filename=filename)

    except Exception as e:
        logger.error(f"VLM Error: {e}")
        return _failed_result()


def _extract_single_segment(image_bytes: bytes, ocr_hint: str = "", filename: str = "") -> dict:
    """Core extraction logic for a single image segment."""
    try:
        image_id = Path(filename).stem if filename else "invoice"
        # 4096 ctx = 2880 image tokens + ~400 prompt tokens → ~816 left for OCR hint
        # Keep first 40 lines (header/items) + last 20 lines (totals/tax), cap at 1800 chars
        if ocr_hint:
            lines = [l for l in ocr_hint.splitlines() if l.strip()]
            if len(lines) > 60:
                lines = lines[:40] + ["..."] + lines[-20:]
            ocr_hint = "\n".join(lines)
            if len(ocr_hint) > 1800:
                ocr_hint = ocr_hint[:1800]

            # Check for actual Indic unicode chars (Devanagari to Malayalam range)
            has_indic = any(
                0x0900 <= ord(ch) <= 0x0DFF
                for ch in ocr_hint
            )

            latin_count = len(re.findall(r'[a-zA-Z]', ocr_hint))
            word_count = len(ocr_hint.split())
            is_latin_junk = not has_indic and latin_count > 20 and word_count > 10

            detected_langs_in_hint = _detect_scripts(ocr_hint)
            n_families = _count_distinct_script_families(detected_langs_in_hint)
            # Cross-script = EasyOCR found mixed scripts (e.g., Bengali+Telugu+Kannada) on same image
            # This means stylized font was misread. The OCR hint is GARBAGE.
            is_cross_script = has_indic and n_families > 1

            if is_cross_script:
                # *** THE KEY FIX ***
                # The OCR hint is contaminated/garbage from a stylized font.
                # Sending it to MiniCPM confuses the model badly.
                # Instead: discard the hint entirely and let MiniCPM read from the image directly.
                # MiniCPM-V 2.6 has native Indic language support and works better WITHOUT bad hints.
                logger.warning(
                    "[vlm] Cross-script contamination detected (%d families: %s). "
                    "Discarding garbage OCR hint. MiniCPM will read image directly.",
                    n_families, detected_langs_in_hint
                )
                ocr_context = "[READ DIRECTLY FROM IMAGE. Do not invent text. Extract exactly what you see.]"
                # has_indic stays True so we still route to MiniCPM
            elif is_latin_junk:
                ocr_context = "[NOTICE: Script detection uncertain. READ DIRECTLY FROM IMAGE.]"
            else:
                ocr_context = f"[OCR HINT - VERIFY AGAINST IMAGE]\n{ocr_hint}\n[/OCR HINT]"
        else:
            ocr_context = ""
            has_indic = False

        ref_alphabets = _load_reference_alphabets(
            ocr_hint=ocr_hint,
            inject_confusion_map=is_cross_script if ocr_hint else False
        )

        # ── POINT 3: Prompt Feedback Loop ────────────────────────────────────
        # Pass 1: Quick language scan (cheap single-word response).
        # Pass 2: Full extraction with language-specific rule injected into prompt.
        # Only run scan when no OCR hint (pure image case — we can't tell language otherwise).
        if not ocr_hint or not has_indic:
            # If no OCR hint, we don't know the language yet — run quick scan
            scan_model = "minicpm"  # Always use vision model for scan
            detected_lang = _quick_language_scan(image_bytes, model_type=scan_model)
        else:
            # OCR hint has Indic chars → detect from text (cheaper than another VLM call)
            detected_langs = _detect_scripts(ocr_hint)
            detected_lang = detected_langs[0] if detected_langs else "mixed"

        lang_rule = _LANG_SPECIFIC_RULES.get(detected_lang, "")
        if lang_rule:
            lang_rule_injection = f"\n[LANGUAGE CONTEXT]: {lang_rule}\n"
        else:
            lang_rule_injection = ""

        prompt = ALL_IN_ONE_PROMPT_TEMPLATE.format(
            ocr_context=ocr_context,
            reference_alphabets=ref_alphabets
        )
        # Inject language-specific rule right after the task header (max 1 sentence, ~20 tokens)
        prompt = f"### TASK: EXTRACT DATA FROM IMAGE: {image_id}{lang_rule_injection}\n" + prompt

        # Decide model: language scan overrides the has_indic heuristic for accuracy
        if detected_lang == "english":
            model_type = "qwen"
        else:
            model_type = "minicpm"
        logger.info("[vlm] Pass 2 extraction | lang=%s | model=%s", detected_lang, model_type)

        res = query_local_llava(image_bytes, prompt, api_key=INTERNAL_MODEL_API_KEY, model_type=model_type)
        parsed = _clean_output(res)

        if not parsed:
            # If parsing failed entirely, return the raw first 300 chars for visibility
            return {
                "fields": {"Raw Output": res[:300] if res else "No response from model"}, 
                "is_invoice": False, 
                "_source": "llava_parsing_failed"
            }

        # Strip hallucinated repeated values (e.g. "33 33 33 33...")
        cleaned_fields = _postprocess_fields(parsed)

        # EMPTY OUTPUT RECOVERY: If VLM returned empty (hallucination stripped or blank),
        # retry once with NO OCR hint — pure image-only read.
        full_text = cleaned_fields.get("full_extraction", "")
        all_empty = all(not v for v in cleaned_fields.values() if isinstance(v, str))
        
        if (not full_text or all_empty) and ocr_hint:
            logger.warning("[vlm] Empty extraction detected. Retrying without OCR hint (pure image read)...")
            retry_prompt = f"### TASK: EXTRACT DATA FROM IMAGE: {image_id}\n" + ALL_IN_ONE_PROMPT_TEMPLATE.format(
                ocr_context="[READ DIRECTLY FROM IMAGE. Ignore previous hints. Extract all visible text.]",
                reference_alphabets=""
            )
            retry_res = query_local_llava(image_bytes, retry_prompt, api_key=INTERNAL_MODEL_API_KEY)
            retry_parsed = _clean_output(retry_res)
            if retry_parsed:
                cleaned_fields = _postprocess_fields(retry_parsed)
                logger.info("[vlm] Retry successful. Source: image_only_fallback")
                return {
                    "fields": cleaned_fields,
                    "is_invoice": True,
                    "_source": "image_only_fallback"
                }

        # If still empty after all retries, preserve the raw result for debugging
        if not cleaned_fields.get("full_extraction") and res:
             cleaned_fields["debug_raw_vlm"] = res[:500]

        return {
            "fields": cleaned_fields,
            "is_invoice": True,
            "_source": "llava_local_with_ocr_hint" if ocr_hint else "llava_local_only"
        }
    except Exception as e:
        logger.error(f"VLM Error: {e}")
        return _failed_result()

def _failed_result():
    return {"fields":{}, "is_invoice":False, "_source":"failed"}


def extract_invoice_details(image, image_bytes, ocr_text="", layout_context=None, required_fields=None):
    res = vlm_extract_all(image_bytes)
    return {**res.get("fields", {}), "_source": res.get("_source", "unknown")}
