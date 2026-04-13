"""
backend/vlm/vlm_model.py
VLM invoice field extraction — 3-tier fallback chain:
  Tier 1: LLaVA 1.6 Mistral-7B (HF Inference API)
  Tier 2: BLIP-2 Flan-T5-XL   (HF Inference API)
  Tier 3: BLIP-base            (local, requires requirements-ml.txt)
Source: vlm_llava_project/model/vlm_model.py
Updated: adds _source key to track which tier succeeded.
"""
from __future__ import annotations

import json
import logging
import re

from backend.vlm.hf_client import query_huggingface, query_llava
from backend.vlm.local_model import generate

logger = logging.getLogger(__name__)

MASTER_PROMPT = (
    "Extract the following details from this invoice and return ONLY a valid JSON object. "
    "Do not include any explanations, greetings, or markdown formatting outside the JSON block.\n\n"
    "Required fields:\n"
    '- "vendor_name": (The shop or company name. Example: \'Acme Corp\')\n'
    '- "invoice_number": (The invoice or bill number. Example: \'INV-1234\')\n'
    '- "date": (The invoice date. Example: \'YYYY-MM-DD\')\n'
    '- "total_amount": (The final total amount, including currency. Example: \'$150.00\')\n\n'
    "If a field is missing, use \"Not found\" as its value.\n\n"
    "JSON output structure:\n"
    "{\n"
    '  "vendor_name": "...",\n'
    '  "invoice_number": "...",\n'
    '  "date": "...",\n'
    '  "total_amount": "..."\n'
    "}"
)

_REQUIRED_KEYS = ["vendor_name", "invoice_number", "date", "total_amount"]


def _clean_output(text: str) -> dict | None:
    """Robust JSON extractor from LLM output."""
    if not text:
        return None

    text = re.sub(r"(USER|ASSISTANT|QUESTION|ANSWER):", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        json_str = match.group(0).replace("```json", "").replace("```", "")
        try:
            data = json.loads(json_str)
            for key in _REQUIRED_KEYS:
                if key not in data:
                    data[key] = "Not found"
            return data
        except json.JSONDecodeError as exc:
            logger.warning("[vlm_model] JSON parse failed: %s", exc)
    return None


def extract_invoice_details(image, image_bytes: bytes) -> dict:
    """
    Extract invoice fields from an image using 3-tier VLM fallback.
    Returns dict with keys: vendor_name, invoice_number, date, total_amount, _source.
    """
    # ── Tier 1: LLaVA ──────────────────────────────────────────────────────
    try:
        logger.info("[vlm] Trying LLaVA...")
        res = query_llava(image_bytes, MASTER_PROMPT)
        if "ERROR" not in res:
            parsed = _clean_output(res)
            if parsed:
                logger.info("[vlm] LLaVA succeeded")
                parsed["_source"] = "llava"
                return parsed
            raise ValueError("LLaVA returned invalid JSON")
        raise ValueError(res)
    except Exception as exc:
        logger.warning("[vlm] LLaVA failed: %s — trying BLIP-2...", exc)

    # ── Tier 2: BLIP-2 cloud ────────────────────────────────────────────────
    try:
        res = query_huggingface(image_bytes, MASTER_PROMPT)
        if "ERROR" not in res:
            parsed = _clean_output(res)
            if parsed:
                logger.info("[vlm] BLIP-2 succeeded")
                parsed["_source"] = "blip2"
                return parsed
            raise ValueError("BLIP-2 returned invalid JSON")
        raise ValueError(res)
    except Exception as exc:
        logger.warning("[vlm] BLIP-2 failed: %s — trying local BLIP...", exc)

    # ── Tier 3: Local BLIP-base (iterative per-field) ───────────────────────
    logger.info("[vlm] Using local BLIP-base (iterative)...")
    results: dict[str, str] = {k: "Not found" for k in _REQUIRED_KEYS}
    field_prompts = {
        "vendor_name": "What is the shop or vendor name?",
        "invoice_number": "What is the invoice number?",
        "date": "What is the invoice date?",
        "total_amount": "What is the total amount?",
    }
    for key, prompt in field_prompts.items():
        try:
            res = generate(image, prompt)
            res = re.sub(r"(USER|ASSISTANT|QUESTION|ANSWER):", "", res, flags=re.IGNORECASE).strip()
            results[key] = res if len(res) >= 3 else "Not found"
        except Exception as local_exc:
            logger.error("[vlm] Local model failed for %s: %s", key, local_exc)

    results["_source"] = "local"
    return results
