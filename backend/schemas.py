"""
backend/schemas.py
Pydantic models for all API request/response shapes.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    message: str
    db_connected: bool


class UploadResponse(BaseModel):
    job_id: str
    file_count: int


class OcrFields(BaseModel):
    invoice_no: str | None = None
    date: str | None = None
    phone: str | None = None
    sub_total: float | None = None
    tax: float | None = None
    grand_total: float | None = None


class VlmFields(BaseModel):
    vendor_name: str = "Not found"
    invoice_number: str = "Not found"
    date: str = "Not found"
    total_amount: str = "Not found"


class LayoutLMStatus(BaseModel):
    enabled: bool = False
    source: str = "Hugging Face API"
    mode: str = "layout-aware document understanding"
    model_name: str = "microsoft/layoutlmv3-base"
    executed: bool = False
    fallback_used: bool = False
    note: str = ""


class CombinedInvoiceResult(BaseModel):
    status: str                                   # "success" | "failed"
    image_name: str
    error: str | None = None

    # ── OCR (PaddleOCR) ──────────────────────────────────────────────────────
    ocr_texts: list[str] = Field(default_factory=list)
    ocr_language_summary: dict[str, Any] = Field(default_factory=dict)
    ocr_invoice_fields: dict[str, Any] = Field(default_factory=dict)

    # ── Layout (LayoutLMv3 via HF API) ───────────────────────────────────────
    document_type: str = "unknown"
    layout_regions: list[dict[str, Any]] = Field(default_factory=list)
    detected_blocks: list[dict[str, Any]] = Field(default_factory=list)
    layoutlm_status: dict[str, Any] = Field(default_factory=dict)
    layoutlm_embedding_preview: list[float] = Field(default_factory=list)

    # ── VLM (LLaVA / BLIP-2 / local) ────────────────────────────────────────
    vlm_fields: dict[str, str] = Field(default_factory=dict)
    vlm_source: str = "unavailable"

    # ── Exports ───────────────────────────────────────────────────────────────
    json_output_url: str = ""
    excel_file_url: str = ""
    text_report_url: str = ""
    text_report_preview: str = ""
    gsheets_synced: bool = False
    db_id: str | None = None


class ProcessResponse(BaseModel):
    results: list[CombinedInvoiceResult]
    total: int
