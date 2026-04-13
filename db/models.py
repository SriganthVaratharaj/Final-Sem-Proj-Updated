"""
db/models.py
MongoDB document schema definition for invoice extraction results.
These are plain dicts — no ORM. Structure documented for reference.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def make_invoice_document(
    file_name: str,
    document_type: str,
    # OCR
    ocr_texts: list[str],
    ocr_language_summary: dict[str, Any],
    ocr_invoice_fields: dict[str, Any],
    ocr_layout: list[dict],
    # Layout
    layout_regions: list[dict],
    detected_blocks: list[dict],
    layoutlm_status: dict[str, Any],
    layoutlm_embedding_preview: list[float],
    # VLM
    vlm_fields: dict[str, str],
    vlm_source: str,
    # Exports
    json_output_url: str = "",
    excel_file_url: str = "",
    text_report_url: str = "",
    gsheets_synced: bool = False,
    status: str = "success",
    error: str | None = None,
) -> dict[str, Any]:
    """
    Build a MongoDB document dict for the invoice_results collection.

    Collection: invoice_results
    Indexes recommended:
        - created_at (descending)
        - status
        - document_type
    """
    return {
        "created_at": datetime.now(timezone.utc),
        "file_name": file_name,
        "status": status,
        "error": error,
        "document_type": document_type,

        # ── OCR (PaddleOCR) ──────────────────────────────────────────
        "ocr": {
            "texts": ocr_texts,
            "language_summary": ocr_language_summary,
            "invoice_fields": ocr_invoice_fields,
            "layout": ocr_layout,
        },

        # ── Layout (LayoutLMv3 via HF API) ───────────────────────────
        "layout": {
            "regions": layout_regions,
            "detected_blocks": detected_blocks,
            "layoutlm_status": layoutlm_status,
            "embedding_preview": layoutlm_embedding_preview,
        },

        # ── VLM (LLaVA / BLIP-2 / local via HF API) ─────────────────
        "vlm": {
            "fields": vlm_fields,
            "source": vlm_source,
        },

        # ── Exports ──────────────────────────────────────────────────
        "exports": {
            "json_url": json_output_url,
            "excel_url": excel_file_url,
            "txt_report_url": text_report_url,
            "gsheets_synced": gsheets_synced,
        },
    }
