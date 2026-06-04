"""
backend/db/auth_repository.py
Mocked repository for user authentication.
"""
from __future__ import annotations
from typing import Optional, Dict, Any

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Accept any password for mock
    return True

def get_password_hash(password: str) -> str:
    return "mocked_hash"

async def create_user(email: str, password: str) -> Dict[str, Any]:
    clean_email = email.strip().lower()
    return {
        "email": clean_email,
        "hashed_password": "mocked_hash",
        "_id": "mock_id"
    }

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    clean_email = email.strip().lower()
    return {
        "email": clean_email,
        "hashed_password": "mocked_hash",
        "_id": "mock_id"
    }
