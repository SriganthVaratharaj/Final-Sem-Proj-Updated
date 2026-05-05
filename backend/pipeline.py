from __future__ import annotations
"""
## KAGGLE_VLM_ONLY_PIPELINE
backend/pipeline.py

Architecture: 100% Remote VLM Inference via Kaggle.
No local OCR (Paddle/EasyOCR) or LayoutLM needed.
"""
import asyncio, logging
from pathlib import Path
import cv2
import numpy as np

from backend.utils.image_enhancer import enhance_for_vlm, split_dual_invoice
from backend.vlm.vlm_model import vlm_extract_all
from backend.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

async def run_pipeline(image_path, image_bytes, original_filename, on_stage=None, user_email=None, session_id=None, correction_rules=""):
    stem = Path(original_filename).stem
    user_output_dir = OUTPUT_DIR / (user_email or "guest")
    user_output_dir.mkdir(parents=True, exist_ok=True)

    async def _emit(s):
        if on_stage: await on_stage(s)

    try:
        image_segments = split_dual_invoice(image_bytes)
        
        combined_vlm_parts = []
        combined_twin_parts = []
        final_dominant_lang = "latin"
        
        for seg_idx, seg_bytes in enumerate(image_segments):
            logger.info("[pipeline] Processing Image Segment %d/%d", seg_idx+1, len(image_segments))
            await _emit(f"Processing Bill {seg_idx+1} on Kaggle GPU...")

            # ── Image Enhancement ──────────────────────────────────────
            vlm_bytes = await asyncio.to_thread(enhance_for_vlm, seg_bytes)

            # ── VLM Extraction (Kaggle API) ────────────────────────────
            # No OCR hint passed, pure visual reasoning
            vlm_res = await asyncio.to_thread(
                vlm_extract_all, vlm_bytes, correction_rules, "", original_filename
            )

            vlm_text = vlm_res.get("fields", {}).get("full_extraction", "")
            if not vlm_text: vlm_text = f"Extraction failed for segment {seg_idx+1}"
            
            # The Kaggle VLM directly generates the spatial layout twin text
            twin_text = vlm_res.get("fields", {}).get("english_extraction", "")
            
            combined_vlm_parts.append(vlm_text)
            combined_twin_parts.append(twin_text)

        # ── Result Combination ────────────────────────────────────────────────
        final_vlm = "\n\n---\n\n### SECOND BILL ###\n\n".join(combined_vlm_parts) if len(combined_vlm_parts) > 1 else combined_vlm_parts[0]
        final_twin = "\n\n---\n\n".join(combined_twin_parts)

        # We use the FIRST segment's fields for structured output
        first_res = vlm_res if image_segments else {"fields": {}, "_source": "unavailable"}
        raw_fields = first_res.get("fields", {})
        source = first_res.get("_source", "unavailable")
        final_dominant_lang = raw_fields.get("metadata", {}).get("detected_language", "unknown")
        
        from backend.utils.layout_template import map_to_standard_template
        template_fields = map_to_standard_template(raw_fields)

        return {
            "status": "success",
            "vlm_output": final_vlm,
            "vlm_fields": raw_fields,
            "template_fields": template_fields,
            "vlm_source": source,
            "digital_twin_content": final_twin,
            "metadata": {
                "segments_processed": len(image_segments),
                "dominant_language": final_dominant_lang
            }
        }

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return {"status": "failed", "error": str(e)}
