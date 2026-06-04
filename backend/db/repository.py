"""
backend/db/repository.py
Mocked CRUD operations.
"""
from __future__ import annotations
import logging
from typing import Any
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# In-memory mock store for results
_mock_store = []

async def save_result(result: dict[str, Any]) -> str | None:
    doc_id = str(uuid.uuid4())
    doc = result.copy()
    doc["_id"] = doc_id
    doc["created_at"] = datetime.now()
    _mock_store.insert(0, doc)
    return doc_id

async def get_result(result_id: str) -> dict[str, Any] | None:
    for doc in _mock_store:
        if doc["_id"] == result_id:
            d = doc.copy()
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            return d
    return None

async def list_results(limit: int = 20, user_email: str | None = None) -> list[dict[str, Any]]:
    results = []
    for doc in _mock_store:
        if user_email and doc.get("user_email") != user_email:
            continue
        d = doc.copy()
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        results.append(d)
        if len(results) >= limit:
            break
    return results

async def search_results(q: str, user_email: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    results = []
    q_lower = q.lower()
    for doc in _mock_store:
        if user_email and doc.get("user_email") != user_email:
            continue
        # basic search mock
        match = False
        if q_lower in doc.get("file_name", "").lower():
            match = True
        elif q_lower in doc.get("document_type", "").lower():
            match = True
            
        if match:
            d = doc.copy()
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            results.append(d)
            if len(results) >= limit:
                break
    return results

async def delete_result(result_id: str) -> bool:
    global _mock_store
    initial_len = len(_mock_store)
    _mock_store = [d for d in _mock_store if d["_id"] != result_id]
    return len(_mock_store) < initial_len
