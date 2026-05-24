"""
backend/db/auth_repository.py
Repository for user authentication and management.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorCollection
from passlib.context import CryptContext
from fastapi import HTTPException
from backend.db.connection import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

async def _users_collection() -> AsyncIOMotorCollection:
    db = get_db()
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection is currently unavailable. Please check backend config or try again later."
        )
    return db["users"]

async def create_user(email: str, password: str) -> Dict[str, Any]:
    """Create a new user, returns user dict or raises ValueError if exists."""
    try:
        collection = await _users_collection()
        existing = await collection.find_one({"email": email})
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Database error checking existing user: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database connection error. Please verify MongoDB connectivity or try again later."
        )
        
    if existing:
        raise ValueError("User already exists")
        
    hashed_pw = get_password_hash(password)
    user_doc = {
        "email": email,
        "hashed_password": hashed_pw
    }
    
    try:
        result = await collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        return user_doc
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Database error creating user: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database connection error. Please verify MongoDB connectivity or try again later."
        )

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    try:
        collection = await _users_collection()
        return await collection.find_one({"email": email})
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Database error during get_user_by_email: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database connection error. Please verify MongoDB connectivity or try again later."
        )
