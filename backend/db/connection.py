"""
backend/db/connection.py
MongoDB connection management (async via Motor + optional sync via PyMongo).
Connection is lazy — the app does NOT fail to start if MongoDB is unavailable.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from backend.config import get_config_value

load_dotenv()

logger = logging.getLogger(__name__)

import urllib.parse

def sanitize_mongo_uri(uri: str) -> str:
    if not uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    if "@" not in rest:
        return uri
    # Split by the last '@' which separates credentials from the host
    parts = rest.rsplit("@", 1)
    creds, host_part = parts[0], parts[1]
    if ":" in creds:
        username, password = creds.split(":", 1)
        encoded_user = urllib.parse.quote_plus(urllib.parse.unquote(username))
        encoded_pass = urllib.parse.quote_plus(urllib.parse.unquote(password))
        return f"{scheme}://{encoded_user}:{encoded_pass}@{host_part}"
    else:
        encoded_user = urllib.parse.quote_plus(urllib.parse.unquote(creds))
        return f"{scheme}://{encoded_user}@{host_part}"

MONGO_URI: str = sanitize_mongo_uri(get_config_value("MONGO_URI", "mongodb://localhost:27017"))
MONGO_DB_NAME: str = get_config_value("MONGO_DB_NAME", "invoice_ai")

# ── Async client (Motor) — used inside FastAPI async routes ───────────────────
_async_client = None
_async_db = None


def get_async_db():
    global _async_client, _async_db
    if _async_db is None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            _async_client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            _async_db = _async_client[MONGO_DB_NAME]
            logger.info("Motor async MongoDB client initialised → %s / %s", MONGO_URI, MONGO_DB_NAME)
        except Exception as exc:
            logger.warning("Motor client init failed (MongoDB unavailable?): %s", exc)
            _async_db = None
    return _async_db


# Alias for backwards compatibility or common naming
get_db = get_async_db



async def ping_db() -> bool:
    """Returns True if MongoDB is reachable, False otherwise."""
    db = get_async_db()
    if db is None:
        return False
    try:
        await db.client.admin.command("ping")
        return True
    except Exception:
        return False
