"""
backend/pipeline.py
Master pipeline orchestrator — runs all three AI modules in sequence:
  Stage 1: PaddleOCR   (local)
  Stage 2: LayoutLMv3  (HF Inference API)
  Stage 3: VLM/LLaVA   (HF Inference API + fallback chain)
Then exports results and saves to MongoDB.
"""
from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import AsyncIterator

from PIL import Image

from backend.config import OUTPUT_DIR
from backend.layout.box_adapter import build_entries, paddle_boxes_to_layoutlm, paddle_boxes_to_pixel_rect
from backend.layout.layoutlm_service import analyze_document_layout
from backend.ocr.engine import run_ocr
from backend.ocr.postprocessing import postprocess
from backend.ocr.preprocessing import preprocess_image
from backend.ocr.validator import validate_image_clarity
from backend.utils.export import export_to_excel, save_layout_json
from backend.utils.image_optimizer import optimize_image
from backend.utils.pdf_converter import get_first_page_as_bytes, is_pdf_supported
from backend.utils.report_generator import generate_structured_report, save_structured_report
from backend.vlm.vlm_model import extract_invoice_details
from db.repository import save_result

logger = logging.getLogger(__name__)


def _empty_result(image_name: str, error: str | None = None) -> dict:
    return {
        "status": "failed",
        "image_name": image_name,
        "error": error,
        "ocr_texts": [],
        "ocr_language_summary": {},
        "ocr_invoice_fields": {},
        "ocr_layout": [],
        "document_type": "unknown",
        "layout_regions": [],
        "detected_blocks": [],
        "layoutlm_status": {},
        "layoutlm_embedding_preview": [],
        "vlm_fields": {},
        "vlm_source": "unavailable",
        "json_output_url": "",
        "excel_file_url": "",
        "text_report_url": "",
        "text_report_preview": "",
        "gsheets_synced": False,
        "db_id": None,
    }


async def run_pipeline(
    image_path: str,
    image_bytes: bytes,
    original_filename: str,
    on_stage: AsyncIterator | None = None,
    user_email: str | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Run the full 3-stage pipeline for a single image.

    Args:
        image_path:        Path to the saved temp file
        image_bytes:       Raw bytes of the uploaded image
        original_filename: Original file name (for export naming)
        on_stage:          Optional async callback(stage_name: str) for SSE progress
        user_email:        Email of logged in user (if any)
        session_id:        Temp session ID if guest (if no email)

    Returns:
        Combined result dict
    """
    stem = Path(original_filename).stem
    result = _empty_result(original_filename)
    
    # ── Folder Setup ──
    # Default to a subfolder based on user or session
    subfolder = user_email if user_email else f"tmp/{session_id}"
    user_output_dir = OUTPUT_DIR / subfolder
    user_output_dir.mkdir(parents=True, exist_ok=True)

    async def _emit(stage: str):
        if on_stage:
            await on_stage(stage)

    try:
        # ══════════════════════════════════════════════════════════════
        # PDF HANDLING
        # ══════════════════════════════════════════════════════════════
        if original_filename.lower().endswith(".pdf"):
            logger.info("[pipeline][%s] Converting PDF to image...", original_filename)
            if not is_pdf_supported():
                result["error"] = "PDF processing not supported (PyMuPDF missing)"
                return result
            
            pdf_png_bytes = await asyncio.to_thread(get_first_page_as_bytes, image_bytes)
            if not pdf_png_bytes:
                result["error"] = "Failed to convert PDF (might be empty or corrupted)"
                return result
                
            # Replace the image bytes and write to temp path so OpenCV/Paddle can read it
            image_bytes = pdf_png_bytes
            with open(image_path, "wb") as f:
                f.write(image_bytes)
                
            logger.info("[pipeline][%s] PDF converted to PNG successfully", original_filename)

        # ══════════════════════════════════════════════════════════════
        # STAGE 1 — PaddleOCR
        # ══════════════════════════════════════════════════════════════
        await _emit("ocr")
        logger.info("[pipeline][%s] Stage 1: Image validation + PaddleOCR", original_filename)

        is_clear, reason = await asyncio.to_thread(validate_image_clarity, image_path)
        if not is_clear:
            result["error"] = f"Image quality check failed: {reason}"
            return result

        preprocessed = await asyncio.to_thread(preprocess_image, image_path)
        texts, poly_boxes, confidences, ocr_metadata = await asyncio.to_thread(run_ocr, preprocessed)
        ocr_result = await asyncio.to_thread(postprocess, texts, poly_boxes, confidences, ocr_metadata)

        result["ocr_texts"]           = ocr_result.get("text", [])
        result["ocr_language_summary"]= ocr_result.get("language_summary", {})
        result["ocr_invoice_fields"]  = ocr_result.get("invoice_fields", {})
        result["ocr_layout"]          = ocr_result.get("layout", [])

        logger.info("[pipeline][%s] OCR done: %d words", original_filename, len(texts))

        # ══════════════════════════════════════════════════════════════
        # STAGE 2 — LayoutLMv3 (via HF API)
        # ══════════════════════════════════════════════════════════════
        await _emit("layout")
        logger.info("[pipeline][%s] Stage 2: LayoutLMv3", original_filename)

        pil_image    = Image.open(image_path).convert("RGB")
        pixel_rects  = paddle_boxes_to_pixel_rect(poly_boxes)
        lm_boxes     = paddle_boxes_to_layoutlm(poly_boxes, pil_image.size)
        entries      = build_entries(texts, pixel_rects, lm_boxes, confidences)
        joined_text  = " ".join(texts)

        layout_result = await asyncio.to_thread(
            analyze_document_layout,
            pil_image, texts, lm_boxes, entries, joined_text,
        )

        result["document_type"]              = layout_result.get("document_type", "unknown")
        result["layoutlm_status"]            = layout_result.get("layoutlmv3_status", {})
        result["layoutlm_embedding_preview"] = layout_result.get("embedding_preview", [])
        doc_layout                           = layout_result.get("document_layout_analysis", {})
        result["layout_regions"]             = doc_layout.get("layout_regions", [])
        result["detected_blocks"]            = doc_layout.get("detected_blocks", [])

        logger.info("[pipeline][%s] Layout done: %d regions", original_filename, len(result["layout_regions"]))

        # ══════════════════════════════════════════════════════════════
        # STAGE 3 — VLM / LLaVA (via HF API)
        # ══════════════════════════════════════════════════════════════
        await _emit("vlm")
        logger.info("[pipeline][%s] Stage 3: VLM extraction", original_filename)

        optimized_bytes = await asyncio.to_thread(optimize_image, image_bytes)
        pil_vlm         = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        vlm_data        = await asyncio.to_thread(extract_invoice_details, pil_vlm, optimized_bytes)

        vlm_source = vlm_data.pop("_source", "unknown")
        result["vlm_fields"] = vlm_data
        result["vlm_source"] = vlm_source

        logger.info("[pipeline][%s] VLM done via %s", original_filename, vlm_source)

        # ══════════════════════════════════════════════════════════════
        # EXPORT — Excel + JSON + TXT report
        # ══════════════════════════════════════════════════════════════
        await _emit("export")

        # Merge OCR rule-based fields + VLM fields (VLM takes priority if found)
        ocr_fields  = result["ocr_invoice_fields"]
        merged_fields = {
            "invoice_no":    ocr_fields.get("invoice_no",   "Not found"),
            "date":          vlm_data.get("date")          or ocr_fields.get("date",    "Not found"),
            "vendor_name":   vlm_data.get("vendor_name",   "Not found"),
            "invoice_number":vlm_data.get("invoice_number","Not found"),
            "total_amount":  vlm_data.get("total_amount")  or str(ocr_fields.get("grand_total", "Not found")),
            "tax":           str(ocr_fields.get("tax",     "Not found")),
            "phone":         ocr_fields.get("phone",       "Not found"),
        }

        full_payload = {
            "file_name":               original_filename,
            "document_type":           result["document_type"],
            "extracted_data":          merged_fields,
            "document_layout_analysis": doc_layout,
        }

        excel_path  = await asyncio.to_thread(export_to_excel,           merged_fields, user_output_dir, stem)
        json_path   = await asyncio.to_thread(save_layout_json,           full_payload,  user_output_dir, stem)
        report_text = await asyncio.to_thread(generate_structured_report, full_payload)
        report_path = await asyncio.to_thread(save_structured_report,     report_text,   user_output_dir, stem)

        result["json_output_url"]    = f"/outputs/{subfolder}/{json_path.name}"
        result["excel_file_url"]     = f"/outputs/{subfolder}/{excel_path.name}"
        result["text_report_url"]    = f"/outputs/{subfolder}/{report_path.name}"
        result["text_report_preview"]= report_text

        # ══════════════════════════════════════════════════════════════
        # DB SAVE — MongoDB
        # ══════════════════════════════════════════════════════════════
        if user_email:
            result["user_email"] = user_email
            try:
                db_id = await save_result(result)
                result["db_id"] = str(db_id) if db_id else None
            except Exception as db_exc:
                logger.warning("[pipeline] DB save failed: %s", db_exc)
        else:
            logger.info("[pipeline][%s] Guest user -> Skipping permanent DB save", original_filename)

        # Google sheets step removed

        result["status"] = "success"
        logger.info("[pipeline][%s] Pipeline complete", original_filename)

    except Exception as exc:
        logger.exception("[pipeline][%s] Pipeline failed", original_filename)
        result["error"] = str(exc)

    return result
