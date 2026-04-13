"""
db/repository.py
CRUD operations for the invoice_results MongoDB collection.
All operations are async (Motor). Each function silently returns a
safe fallback value if the DB connection is unavailable.
"""
from __future__ import annotations

import logging
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from db.connection import get_async_db
from db.models import make_invoice_document

logger = logging.getLogger(__name__)

COLLECTION = "invoice_results"


async def save_result(result: dict[str, Any]) -> str | None:
    """
    Persist a pipeline result dict to MongoDB.
    Returns the inserted document _id as a string, or None on failure.
    """
    db = get_async_db()
    if db is None:
        logger.warning("DB unavailable — result not saved to MongoDB")
        return None

    doc = make_invoice_document(
        file_name=result.get("image_name", "unknown"),
        document_type=result.get("document_type", "unknown"),
        ocr_texts=result.get("ocr_texts", []),
        ocr_language_summary=result.get("ocr_language_summary", {}),
        ocr_invoice_fields=result.get("ocr_invoice_fields", {}),
        ocr_layout=result.get("ocr_layout", []),
        layout_regions=result.get("layout_regions", []),
        detected_blocks=result.get("detected_blocks", []),
        layoutlm_status=result.get("layoutlm_status", {}),
        layoutlm_embedding_preview=result.get("layoutlm_embedding_preview", []),
        vlm_fields=result.get("vlm_fields", {}),
        vlm_source=result.get("vlm_source", "unavailable"),
        json_output_url=result.get("json_output_url", ""),
        excel_file_url=result.get("excel_file_url", ""),
        text_report_url=result.get("text_report_url", ""),
        gsheets_synced=result.get("gsheets_synced", False),
        status=result.get("status", "success"),
        error=result.get("error"),
    )

    try:
        inserted = await db[COLLECTION].insert_one(doc)
        logger.info("DB saved result _id=%s", inserted.inserted_id)
        return str(inserted.inserted_id)
    except Exception as exc:
        logger.warning("DB insert failed: %s", exc)
        return None


async def get_result(result_id: str) -> dict[str, Any] | None:
    """Fetch a single result by its MongoDB _id string."""
    db = get_async_db()
    if db is None:
        return None
    try:
        oid = ObjectId(result_id)
    except InvalidId:
        return None

    try:
        doc = await db[COLLECTION].find_one({"_id": oid})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as exc:
        logger.warning("DB get_result failed: %s", exc)
        return None


async def list_results(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent `limit` results, newest first."""
    db = get_async_db()
    if db is None:
        return []
    try:
        cursor = db[COLLECTION].find(
            {},
            {"ocr.texts": 0, "ocr.layout": 0, "layout.regions": 0},  # exclude heavy fields
        ).sort("created_at", -1).limit(limit)

        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            if doc.get("created_at"):
                doc["created_at"] = doc["created_at"].isoformat()
            results.append(doc)
        return results
    except Exception as exc:
        logger.warning("DB list_results failed: %s", exc)
        return []


async def delete_result(result_id: str) -> bool:
    """Delete a single result by its _id. Returns True on success."""
    db = get_async_db()
    if db is None:
        return False
    try:
        oid = ObjectId(result_id)
        res = await db[COLLECTION].delete_one({"_id": oid})
        return res.deleted_count == 1
    except Exception as exc:
        logger.warning("DB delete_result failed: %s", exc)
        return False
