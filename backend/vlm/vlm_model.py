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

Step 1: Identify if the image is an 'Invoice', 'Receipt', or a 'General Document'.
Step 2: Identify the primary native language of the text.

{extraction_instruction}

{reference_alphabets}

Step 4 (Format): You MUST return the final output STRICTLY as a single JSON object in the exact structure below. Do NOT add conversational text. Do NOT wrap in markdown code blocks.
{{
  "metadata": {{
      "classification": "Invoice, Receipt, or Document",
      "detected_language": "Language Name"
  }},
  "native_json": {{ ... }}, 
  "english_json": {{ ... }}, 
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
    # Strip markdown code fences if model hallucinated them
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        # Try to find the first '{' and last '}' to extract JSON
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        parsed = json.loads(text)
        return {
            "fields": parsed,
            "is_comprehensive": True,
            "_source": "master_vlm_json"
        }
    except Exception as e:
        logger.error("[vlm] JSON parsing failed: %s", e)
        return {
            "full_extraction": text.strip(),
            "is_comprehensive": False,
            "_source": "vlm_raw_fallback"
        }

# ── OPTIMIZATION: Skip language scan for remote Kaggle to prevent browser timeouts ──
def _quick_language_scan(image_bytes: bytes, model_type: str = "minicpm") -> str:
    """
    Lightweight VLM call to detect invoice language before the full extraction.
    """
    if KAGGLE_VLM_URL and KAGGLE_VLM_URL.strip():
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


def _quick_language_scan(image_bytes: bytes, model_type: str = "minicpm") -> str:
    """
    Lightweight VLM call to detect invoice language before the full extraction.
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

        ext_instr = "Step 3 (Extraction): Perform a full structured extraction. Capture all key-value pairs, tables, and paragraphs. Preserve the logical layout."
        
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
        
        # Save debug
        try:
            from backend.config import OUTPUT_DIR
            debug_path = OUTPUT_DIR / "debug" / image_id
            debug_path.mkdir(parents=True, exist_ok=True)
            for k, suffix in [("native_json", ".json"), ("english_json", "_en.json"), ("native_layout_text", ".txt"), ("english_layout_text", "_en.txt")]:
                val = master_data.get(k, {})
                with open(debug_path / (k + suffix), "w", encoding="utf-8") as f:
                    if suffix == ".json": json.dump(val, f, ensure_ascii=False, indent=4)
                    else: f.write(str(val))
        except: pass

        final_fields = master_data.get("native_json", {})
        final_fields["full_extraction"] = master_data.get("native_layout_text", "")
        final_fields["english_extraction"] = master_data.get("english_layout_text", "")
        final_fields["metadata"] = master_data.get("metadata", {})

        return {
            "fields": final_fields,
            "is_invoice": True,
            "_source": "kaggle_remote_vlm" if "trycloudflare.com" in __import__("backend.config").config.KAGGLE_VLM_URL else "local_vlm"
        }
    except Exception as e:
        logger.error(f"VLM Error: {e}")
        return _failed_result()

def _failed_result():
    return {"fields":{}, "is_invoice":False, "_source":"failed"}

def extract_invoice_details(image, image_bytes, ocr_text="", layout_context=None, required_fields=None):
    res = vlm_extract_all(image_bytes)
    return {**res.get("fields", {}), "_source": res.get("_source", "unknown")}
