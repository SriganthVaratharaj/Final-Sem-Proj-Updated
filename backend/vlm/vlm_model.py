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

logger = logging.getLogger(__name__)

ALL_IN_ONE_PROMPT_TEMPLATE = """
You are a multilingual invoice data extractor. Extract ALL information from this invoice image.

The invoice may be in English, Hindi, Bengali, Tamil, Telugu, Kannada, Gujarati, or a mix.
Extract text EXACTLY as it appears. Do NOT translate. Do NOT guess.

{reference_alphabets}

{ocr_context}

OUTPUT FORMAT: Structured Markdown. No JSON.

STRICT RULES — READ CAREFULLY:
1. Each piece of text must appear EXACTLY ONCE in the output. Do NOT repeat it.
2. If text appears INSIDE a table/box in the image → put it ONLY in the Items Table section.
3. If text appears in the HEADER area → put it ONLY in the Header section.
4. Do NOT copy table cell contents into the Header or Bill Info sections.
5. If a table row is EMPTY in the image, write empty cells: `| | | | |`
6. If a table row has data, each DIFFERENT row must have DIFFERENT content. Do NOT repeat the same row multiple times.
7. If you cannot clearly read a field, write N/A. Do NOT guess or hallucinate.
8. Numbers, prices, and codes: copy digit-by-digit exactly as seen.

Extract in this order:
## Header
- Vendor/Shop Name:
- Address:
- Phone / GSTIN / Tax ID:

## Bill Info  
- Bill No:
- Date:
- Total Amount:

## Items Table
| Sr | Item Name | Qty | Unit Price | Total |
|---|---|---|---|---|
(one row per item — only real items visible in image)

## Other Details
(any remaining text not captured above)
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

def _load_reference_alphabets(ocr_hint: str = "") -> str:
    """Load ONLY the primary detected language alphabet.
    Strictly limited to prevent context overflow on 4GB VRAM (n_ctx=4096).
    """
    alphabets_dir = Path(__file__).parent.parent / "language_alphabets"
    if not alphabets_dir.exists(): return ""

    detected_langs = _detect_scripts(ocr_hint) if ocr_hint else []

    if not detected_langs:
        logger.info("[vlm] No Indic script in OCR hint — skipping alphabet injection to save tokens.")
        return ""

    # ── VRAM-safe: inject at most 2 scripts, 200 chars each ──────────────────
    # Priority: bengali > hindi > tamil > others
    priority_order = ["bengali", "assamese", "hindi", "marathi", "tamil", "telugu", "kannada", "gujarati", "malayalam", "odia", "punjabi", "maithili"]
    selected = [l for l in priority_order if l in detected_langs][:2]  # max 2 scripts

    if not selected:
        selected = detected_langs[:1]  # fallback: first detected

    logger.info("[vlm] Detected scripts: %s — injecting targeted alphabets.", selected)

    ref_text = "[REFERENCE ALPHABETS FOR LANGUAGE DETECTION]\n"
    ref_text += "Match these chars against image to identify the script. DO NOT copy these into your output.\n"
    for file in alphabets_dir.glob("*.txt"):
        if file.stem.lower() in selected:
            content = file.read_text(encoding="utf-8")
            char_lines = []
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("===") and not line.startswith("LANGUAGE") and not line.startswith("SCRIPT"):
                    char_lines.append(line)
            compact = " ".join(char_lines)[:200]  # Hard cap: 200 chars per script (was 600)
            ref_text += f"{file.stem.capitalize()}: {compact}\n"
    ref_text += "[/REFERENCE ALPHABETS]\n"
    return ref_text


def vlm_extract_all(image_bytes: bytes, correction_rules: str = "", ocr_hint: str = "", filename: str = "") -> dict:
    try:
        # Add image ID to the prompt to force context reset/isolation
        image_id = Path(filename).stem if filename else "current_invoice"
        
        # SMART OCR HINT TRUNCATION
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

        ref_alphabets = _load_reference_alphabets(ocr_hint=ocr_hint)

        prompt = ALL_IN_ONE_PROMPT_TEMPLATE.format(ocr_context=ocr_context, reference_alphabets=ref_alphabets)
        # Force strict isolation by mentioning the image ID
        prompt = f"### TASK: EXTRACT DATA FROM IMAGE: {image_id}\n" + prompt
        
        # Decide which model to use
        # If it's an English/Latin bill, use Qwen. If Indic script is detected, use the stronger MiniCPM model.
        model_type = "minicpm" if has_indic else "qwen"
        logger.info(f"[vlm] Using VLM Model Type: {model_type.upper()}")

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
