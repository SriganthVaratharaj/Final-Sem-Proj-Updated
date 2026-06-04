"""
backend/db/connection.py
Real MongoDB connection using Motor.
"""
from __future__ import annotations
import logging
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "invoice_ai")

client = None

def get_async_db():
    global client
    if client is None:
        client = AsyncIOMotorClient(MONGO_URI)
    return client[DB_NAME]

async def ping_db() -> bool:
    try:
        db = get_async_db()
        await db.command("ping")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

get_db = get_async_db
