"""
backend/db/repository.py
Real MongoDB CRUD operations using Motor.
"""
from __future__ import annotations
import logging
from typing import Any
from datetime import datetime
from bson import ObjectId
from backend.db.connection import get_async_db

logger = logging.getLogger(__name__)

async def save_result(result: dict[str, Any]) -> str | None:
    db = get_async_db()
    doc = result.copy()
    doc["created_at"] = datetime.now()
    res = await db["results"].insert_one(doc)
    return str(res.inserted_id)

async def get_result(result_id: str) -> dict[str, Any] | None:
    db = get_async_db()
    try:
        doc = await db["results"].find_one({"_id": ObjectId(result_id)})
    except Exception:
        return None
    
    if doc:
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
    return doc

async def list_results(limit: int = 20, user_email: str | None = None) -> list[dict[str, Any]]:
    db = get_async_db()
    query = {}
    if user_email:
        query["user_email"] = user_email
        
    cursor = db["results"].find(query).sort("created_at", -1).limit(limit)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        results.append(doc)
    return results

async def search_results(q: str, user_email: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    db = get_async_db()
    query = {}
    if user_email:
        query["user_email"] = user_email
    
    # Simple regex search across file_name and document_type
    if q:
        query["$or"] = [
            {"file_name": {"$regex": q, "$options": "i"}},
            {"document_type": {"$regex": q, "$options": "i"}}
        ]
        
    cursor = db["results"].find(query).sort("created_at", -1).limit(limit)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        results.append(doc)
    return results

async def delete_result(result_id: str) -> bool:
    db = get_async_db()
    try:
        res = await db["results"].delete_one({"_id": ObjectId(result_id)})
        return res.deleted_count > 0
    except Exception:
        return False
