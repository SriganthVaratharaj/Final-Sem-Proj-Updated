from __future__ import annotations
"""
## DISTRIBUTED_VLM_PIPELINE
backend/vlm/vlm_model.py
"""
import json, logging, re
from pathlib import Path
from typing import Any
from backend.config import VLM_REQUIRED_FIELDS, INTERNAL_MODEL_API_KEY
from backend.vlm.gguf_engine import query_local_llava
from backend.utils.image_enhancer import split_for_extraction

logger = logging.getLogger(__name__)

MASTER_PROMPT_TEMPLATE = """
You are an End-to-End Layout-Aware Visual OCR AI. Analyze the image and perform these strict tasks:

Step 1: Identify ALL documents in the image. If there are multiple separate bills or duplicates, extract all of them.
Step 2: Identify the primary native language(s).

{extraction_instruction}
Step 3.5 (Structure): For any table, grid, or box structure, you MUST use Markdown Table format (| Column |) to preserve the visual layout.

{reference_alphabets}

Step 4 (Format): You MUST return the final output STRICTLY as a single JSON object. Do NOT add conversational text. Do NOT wrap in markdown code blocks.
### CRITICAL: YOUR ENTIRE RESPONSE MUST BE A SINGLE VALID JSON OBJECT. NO MARKDOWN, NO EXPLANATION, NO PREFACE. ###
IMPORTANT: Use standard, flat English keys (e.g., "vendor_name", "invoice_number", "total_amount") for BOTH native_json and english_json. Do NOT use nested objects inside native_json. The keys must be English, only the values should be in the native language.
{{
  "metadata": {{
      "classification": "Identify all documents found",
      "detected_language": "Language Name",
      "document_count": "Number of docs found"
  }},
  "native_json": {{ "vendor_name": "...", "invoice_number": "...", "total_amount": "...", "items": "Markdown Table", "...": "..." }}, 
  "english_json": {{ "vendor_name": "Translated", "...": "..." }}, 
  "native_layout_text": "...", 
  "english_layout_text": "..." 
}}
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
    # Pre-clean: remove potential markdown garbage
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.replace('```', '').strip()
    
    try:
        # Strategy 1: Find the largest JSON-like block
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            candidate = text[start:end+1]
            # Heuristic repair: fix common trailing commas before closing braces
            candidate = re.sub(r',\s*([\]}])', r'\1', candidate)
            try:
                parsed = json.loads(candidate)
                return {
                    "fields": parsed,
                    "is_comprehensive": True,
                    "_source": "master_vlm_json"
                }
            except:
                pass
        
        # Strategy 2: Try parsing the whole thing if Strategy 1 failed
        parsed = json.loads(text)
        return {
            "fields": parsed,
            "is_comprehensive": True,
            "_source": "master_vlm_json"
        }
    except Exception as e:
        logger.error("[vlm] JSON parsing failed: %s", e)
        logger.debug("[vlm] Raw text that failed: %s", text[:500])
        return {
            "full_extraction": text.strip(),
            "is_comprehensive": False,
            "_source": "vlm_raw_fallback"
        }

def _get_dynamic_kaggle_url():
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    return os.getenv("KAGGLE_VLM_URL", "")

# ── OPTIMIZATION: Skip language scan for remote Kaggle to prevent browser timeouts ──
def _quick_language_scan(image_bytes: bytes, model_type: str = "minicpm") -> str:
    """
    Lightweight VLM call to detect invoice language before the full extraction.
    """
    kaggle_url = _get_dynamic_kaggle_url()
    if kaggle_url and kaggle_url.strip():
        logger.info("[vlm] Remote Kaggle detected. Skipping language scan pass for speed.")
        return "mixed"
    try:
        lang_scan_prompt = """Look at this image. What is the MAIN language/script of the text?
Reply with ONLY one word from this list:
english, hindi, bengali, tamil, telugu, kannada, gujarati, marathi, odia, malayalam, punjabi, urdu, mixed

Do not explain. Do not add punctuation. Just one word."""
        raw = query_local_llava(image_bytes, lang_scan_prompt, model_type="qwen")
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

def _strip_hallucinations(text: str) -> str:
    """Detect and remove hallucinated repeated patterns from VLM output."""
    if not text or len(text) < 10:
        return text
    words = text.split()
    if len(words) < 5:
        return text
    from collections import Counter
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    if top_count / len(words) > 0.6:
        return ""
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
    """Remove duplicate rows/lines from markdown output."""
    if not text:
        return text
    lines = text.split('\n')
    out_lines = []
    in_table = False
    seen_table_rows: set = set()
    prev_line = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and '---' not in stripped:
            in_table = True
        elif in_table and not stripped.startswith('|'):
            in_table = False
            seen_table_rows.clear()
        if in_table and stripped.startswith('|'):
            if '---' in stripped or (out_lines and stripped == out_lines[-1].strip()):
                out_lines.append(line)
                continue
            norm = re.sub(r'\s*\|\s*', '|', stripped)
            if norm not in seen_table_rows:
                seen_table_rows.add(norm)
                out_lines.append(line)
        else:
            if stripped != prev_line:
                out_lines.append(line)
            prev_line = stripped
    return '\n'.join(out_lines)


def _load_reference_alphabets(detected_lang: str) -> str:
    """Load reference alphabets for a specific script."""
    alphabets_dir = Path(__file__).parent.parent / "language_alphabets"
    if not alphabets_dir.exists():
        return ""
    file = alphabets_dir / f"{detected_lang}.txt"
    if not file.exists():
        return ""
    try:
        content = file.read_text(encoding="utf-8")
        standard_chars = []
        current_section = None
        for line in content.splitlines():
            stripped = line.strip()
            if "=== STANDARD ALPHABETS ===" in stripped:
                current_section = "standard"
            elif stripped.startswith("===") or not stripped:
                current_section = None
            elif current_section == "standard" and not stripped.startswith("#"):
                standard_chars.append(stripped)
        chars_text = " ".join(standard_chars)[:150]
        return f"[Script Reference: {detected_lang.capitalize()}]\nAlphabets: {chars_text}\n"
    except:
        return ""


def vlm_extract_all(image_bytes: bytes, correction_rules: str = "", ocr_hint: str = "", filename: str = "") -> dict:
    try:
        image_id = Path(filename).stem if filename else "current_doc"
        segments = split_for_extraction(image_bytes)
        if len(segments) > 1:
            logger.info("[vlm] D&C: %d segments detected.", len(segments))
            all_lines = []
            for i, seg_bytes in enumerate(segments):
                seg_result = _extract_single_segment(seg_bytes, filename=f"{image_id}_seg{i+1}")
                seg_text = seg_result.get("fields", {}).get("full_extraction", "")
                if seg_text:
                    all_lines.append(f"--- SEGMENT {i+1} ---\n{seg_text}")
            return {
                "fields": {"full_extraction": "\n".join(all_lines), "is_comprehensive": True, "_source": "divide_and_conquer"},
                "is_invoice": True, "_source": "divide_and_conquer"
            }
        return _extract_single_segment(image_bytes, filename=filename)
    except Exception as e:
        logger.error(f"VLM Error: {e}")
        return _failed_result()


def _extract_single_segment(image_bytes: bytes, filename: str = "") -> dict:
    """Pure VLM extraction logic."""
    try:
        image_id = Path(filename).stem if filename else "invoice"
        
        # Detect language first
        detected_lang = _quick_language_scan(image_bytes)
        lang_rule = _LANG_SPECIFIC_RULES.get(detected_lang, "")
        ref_alphabets = _load_reference_alphabets(detected_lang)

        ext_instr = (
            "Step 3 (Extraction): Perform a full structured extraction. Capture all key-value pairs, tables, and paragraphs. Preserve the logical layout.\n"
            "Step 3.1 (Translation & Transliteration): Translate all extracted values, text blocks, names, addresses, and line items from their native language to English in both `english_json` and `english_layout_text`. "
            "For example, if the vendor name is written in Tamil/Hindi/Telugu/etc., you must write the original script in `native_json` / `native_layout_text` and its English translation / transliteration in `english_json` / `english_layout_text`. "
            "Ensure that `english_json` contains the exact same keys as `native_json`, but with all values translated/transliterated to English. Do not leave `english_json` empty."
        )
        
        prompt = MASTER_PROMPT_TEMPLATE.format(
            extraction_instruction=ext_instr,
            reference_alphabets=ref_alphabets
        )
        if lang_rule:
            prompt += f"\n[LANGUAGE CONTEXT]: {lang_rule}"

        model_type = "qwen" if detected_lang == "english" else "minicpm"
        logger.info("[vlm] Master Pass | lang=%s | model=%s", detected_lang, model_type)

        res = query_local_llava(image_bytes, prompt, api_key=INTERNAL_MODEL_API_KEY, model_type=model_type)
        parsed_result = _clean_output(res)

        if not parsed_result or "_source" not in parsed_result or parsed_result["_source"] != "master_vlm_json":
            return {"fields": {"full_extraction": res[:1000] if res else "No response"}, "is_invoice": False, "_source": "vlm_parsing_failed"}

        master_data = parsed_result["fields"]
        
        # Save debug as early as possible
        try:
            from backend.config import OUTPUT_DIR
            debug_path = OUTPUT_DIR / "debug" / image_id
            debug_path.mkdir(parents=True, exist_ok=True)
            for k, suffix in [("native_json", ".json"), ("english_json", "_en.json"), ("native_layout_text", ".txt"), ("english_layout_text", "_en.txt")]:
                val = master_data.get(k, {})
                # If the key is missing but the master_data itself has keys that look like fields,
                # it means the model didn't wrap it in native_json.
                if not val and k == "native_json" and len(master_data) > 3:
                     val = {mk: mv for mk, mv in master_data.items() if mk not in ["metadata", "native_layout_text", "english_layout_text"]}
                
                with open(debug_path / (k + suffix), "w", encoding="utf-8") as f:
                    if suffix == ".json": json.dump(val, f, ensure_ascii=False, indent=4)
                    else: f.write(str(val))
        except Exception as de: 
            logger.debug("[vlm] Debug save failed: %s", de)

        # Map fields with fallback to root if wrapper keys are missing
        final_fields = master_data.get("native_json", {})
        if not final_fields and len(master_data) > 3:
             final_fields = {mk: mv for mk, mv in master_data.items() if mk not in ["metadata", "native_layout_text", "english_layout_text", "english_json"]}
        
        final_fields["english_json"] = master_data.get("english_json", {})
        final_fields["full_extraction"] = master_data.get("native_layout_text", "") or master_data.get("full_extraction", "")
        final_fields["english_extraction"] = master_data.get("english_layout_text", "")
        final_fields["metadata"] = master_data.get("metadata", {})

        return {
            "fields": final_fields,
            "is_invoice": True,
            "_source": "kaggle_remote_vlm" if "trycloudflare.com" in _get_dynamic_kaggle_url() else "local_vlm"
        }
    except Exception as e:
        logger.error(f"VLM Error: {e}")
        return _failed_result()

def _failed_result():
    return {"fields":{}, "is_invoice":False, "_source":"failed"}

def extract_invoice_details(image, image_bytes, ocr_text="", layout_context=None, required_fields=None):
    res = vlm_extract_all(image_bytes)
    return {**res.get("fields", {}), "_source": res.get("_source", "unknown")}
