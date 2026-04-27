from __future__ import annotations
"""
## SINGLE_PASS_V2_CHAIN_OF_THOUGHT
backend/vlm/vlm_model.py
"""
import json, logging, re
from typing import Any
from backend.config import VLM_REQUIRED_FIELDS, INTERNAL_MODEL_API_KEY
from backend.vlm.gguf_engine import query_local_llava

logger = logging.getLogger(__name__)

ALL_IN_ONE_PROMPT = """
You are a document extraction assistant. Analyze the image.
If the text is in Bengali, Tamil, Hindi, etc., you MUST use those scripts.

Return ONLY JSON:
{
  "ocr_text": "<full transcription in original script>",
  "fields": {<key-value pairs>}
}
"""

def _clean_output(text: str) -> dict | None:
    if not text: return None
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try: return json.loads(match.group(0).strip().replace("```json", "").replace("```", ""))
        except: pass
    return None

def vlm_extract_all(image_bytes: bytes, correction_rules: str = "") -> dict:
    try:
        res = query_local_llava(image_bytes, ALL_IN_ONE_PROMPT, api_key=INTERNAL_MODEL_API_KEY)
        parsed = _clean_output(res)
        if not parsed: return _failed_result()
        ocr_text = parsed.get("ocr_text", "")
        return {
            "ocr_text": ocr_text,
            "fields": parsed.get("fields", {}),
            "is_invoice": "total" in ocr_text.lower(),
            "_source": "llava_local"
        }
    except: return _failed_result()

def _failed_result():
    return {"ocr_text":"", "fields":{}, "is_invoice":False, "_source":"failed"}

def extract_invoice_details(image, image_bytes, ocr_text="", layout_context=None, required_fields=None):
    res = vlm_extract_all(image_bytes)
    return {**res.get("fields", {}), "_source": res.get("_source", "unknown")}
