"""
db/auth_repository.py
Repository for user authentication and management.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorCollection
from passlib.context import CryptContext
from db.connection import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

async def _users_collection() -> AsyncIOMotorCollection:
    db = await get_db()
    return db["users"]

async def create_user(email: str, password: str) -> Dict[str, Any]:
    """Create a new user, returns user dict or raises ValueError if exists."""
    collection = await _users_collection()
    existing = await collection.find_one({"email": email})
    if existing:
        raise ValueError("User already exists")
        
    hashed_pw = get_password_hash(password)
    user_doc = {
        "email": email,
        "hashed_password": hashed_pw
    }
    result = await collection.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    return user_doc

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    collection = await _users_collection()
    return await collection.find_one({"email": email})
