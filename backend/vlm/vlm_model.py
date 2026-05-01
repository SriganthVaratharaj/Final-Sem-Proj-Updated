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
Perform a FULL "A to Z" comprehensive data extraction of this invoice.
IMPORTANT: The invoice might contain multiple mixed languages (e.g., English, Tamil, Hindi, Bengali, etc.). Detect the languages automatically and extract all the text exactly as it appears without translating unless asked.
Extract EVERY single detail you can see, from the very top to the bottom.
Include:
- Header details (Vendor Name, Address, Contact, Email)
- Invoice Meta (Invoice Number, Date, GSTIN)
- Buyer/Customer Details
- ITEM LIST (Every single item, quantity, rate, and amount)
- Tax details (CGST, SGST, IGST)
- TOTAL AMOUNT (Check the bottom-right carefully)

{reference_alphabets}

{ocr_context}

Output the extracted details as a highly structured Markdown document. 
CRITICAL: You must visually recreate the original invoice layout as closely as possible.
- Use Markdown tables (`| Column A | Column B |`) to represent items, quantities, and prices exactly as they appear in boxes/tables.
- Use headers, bold text, and blockquotes to represent the structure (Vendor at top, totals at bottom).
- DO NOT output JSON. Just give me the full structured Markdown text (Digital Twin).
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
    # Try to extract JSON if it exists
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
            
    # If no JSON or parsing fails, return as a structured block
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
    """Strip hallucinations from all string fields recursively."""
    if not isinstance(data, dict):
        return data
    return {
        k: (_strip_hallucinations(v) if isinstance(v, str) else v)
        for k, v in data.items()
    }

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
    """Load ONLY the language alphabet files that match scripts found in OCR hint.
    Falls back to all languages if OCR hint is empty or no script detected.
    This keeps prompt size small and prevents context overflow.
    """
    alphabets_dir = Path(__file__).parent.parent / "language_alphabets"
    if not alphabets_dir.exists(): return ""

    # Detect which scripts appear in the OCR text
    detected_langs = _detect_scripts(ocr_hint) if ocr_hint else []
    
    # Always include English (it's in every invoice)
    # If nothing detected, try all (small invoice, no Indic text)
    if not detected_langs:
        logger.info("[vlm] No Indic script in OCR hint — skipping alphabet injection to save tokens.")
        return ""  # No injection needed for pure-English or unknown invoices

    logger.info("[vlm] Detected scripts: %s — injecting targeted alphabets.", detected_langs)

    ref_text = "[REFERENCE ALPHABETS FOR LANGUAGE DETECTION]\n"
    ref_text += "Match these chars against image to identify the script. DO NOT copy these into your output.\n"
    for file in alphabets_dir.glob("*.txt"):
        if file.stem.lower() in detected_langs:
            # Only keep lines with actual characters (skip section headers)
            content = file.read_text(encoding="utf-8")
            # Extract only the character lines (strip verbose headers)
            char_lines = []
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("===") and not line.startswith("LANGUAGE") and not line.startswith("SCRIPT"):
                    char_lines.append(line)
            compact = " ".join(char_lines)[:600]  # Hard cap: 600 chars per language
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
                # Take top 40 + bottom 20 (most useful for invoices)
                lines = lines[:40] + ["..."] + lines[-20:]
            ocr_hint = "\n".join(lines)
            # Hard char cap as safety net
            if len(ocr_hint) > 1800:
                ocr_hint = ocr_hint[:1800]

            # Check for Bengali/Gujarati/Tamil chars (valid Indic script)
            has_indic = any(
                0x0900 <= ord(ch) <= 0x0DFF  # Devanagari -> Malayalam range
                for ch in ocr_hint
            )

            # GIBBERISH DETECTION — two triggers:
            # 1. All Latin + no Indic = probably junk from English-only images
            # 2. Cross-script contamination: OCR detected 2+ unrelated script families
            #    (e.g., Hindi + Telugu + Kannada mixed) = artistic font being misread
            latin_count = len(re.findall(r'[a-zA-Z]', ocr_hint))
            word_count = len(ocr_hint.split())
            is_latin_junk = not has_indic and latin_count > 20 and word_count > 10

            detected_langs_in_hint = _detect_scripts(ocr_hint)
            n_families = _count_distinct_script_families(detected_langs_in_hint)
            is_cross_script = has_indic and n_families > 1

            if is_cross_script:
                logger.warning(
                    "[vlm] Cross-script contamination detected (%d script families: %s). "
                    "Passing to VLM anyway since it needs all the help it can get.",
                    n_families, detected_langs_in_hint
                )
                # DO NOT discard the hint. The new VLM prompt can handle noisy hints better.
                ocr_context = f"[OCR HINT - VERIFY AGAINST IMAGE (May contain noise)]\n{ocr_hint}\n[/OCR HINT]"
            elif is_latin_junk:
                ocr_context = "[NOTICE: Script detection uncertain. READ DIRECTLY FROM IMAGE.]"
            else:
                ocr_context = f"[OCR HINT - VERIFY AGAINST IMAGE]\n{ocr_hint}\n[/OCR HINT]"
        else:
            ocr_context = ""

        ref_alphabets = _load_reference_alphabets(ocr_hint=ocr_hint)

        prompt = ALL_IN_ONE_PROMPT_TEMPLATE.format(ocr_context=ocr_context, reference_alphabets=ref_alphabets)
        # Force strict isolation by mentioning the image ID
        prompt = f"### TASK: EXTRACT DATA FROM IMAGE: {image_id}\n" + prompt
        
        res = query_local_llava(image_bytes, prompt, api_key=INTERNAL_MODEL_API_KEY)
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
