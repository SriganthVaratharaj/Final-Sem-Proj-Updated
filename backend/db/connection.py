"""
backend/db/connection.py
Mocked connection management.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

async def ping_db() -> bool:
    """Mock ping returning True."""
    return True

def get_async_db():
    return None

get_db = get_async_db
