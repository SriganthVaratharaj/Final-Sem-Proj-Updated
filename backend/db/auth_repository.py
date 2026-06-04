"""
backend/db/auth_repository.py
Real repository for user authentication using MongoDB.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from backend.db.connection import get_async_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

async def create_user(email: str, password: str) -> Dict[str, Any]:
    db = get_async_db()
    clean_email = email.strip().lower()
    
    # Check if user already exists
    existing = await db["users"].find_one({"email": clean_email})
    if existing:
        raise ValueError("User already exists")
        
    hashed_password = get_password_hash(password)
    user_doc = {
        "email": clean_email,
        "hashed_password": hashed_password
    }
    result = await db["users"].insert_one(user_doc)
    user_doc["_id"] = str(result.inserted_id)
    return user_doc

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    db = get_async_db()
    clean_email = email.strip().lower()
    user_doc = await db["users"].find_one({"email": clean_email})
    if user_doc:
        user_doc["_id"] = str(user_doc["_id"])
    return user_doc

async def update_password(email: str, new_hash: str) -> None:
    db = get_async_db()
    clean_email = email.strip().lower()
    await db["users"].update_one(
        {"email": clean_email},
        {"$set": {"hashed_password": new_hash}}
    )
