"""
backend/db/auth_repository.py
Mocked repository for user authentication (In-Memory).
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import uuid

# In-memory dictionary to store users during the demo
_mock_users: Dict[str, Dict[str, Any]] = {}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # For mock, we just check if the plain password matches the stored "hash"
    return plain_password == hashed_password

def get_password_hash(password: str) -> str:
    # In a real app this would hash the password, here we just store it as is
    return password

async def create_user(email: str, password: str) -> Dict[str, Any]:
    clean_email = email.strip().lower()
    
    if clean_email in _mock_users:
        raise ValueError("User already exists")
    
    # Store user in memory
    user_doc = {
        "email": clean_email,
        "hashed_password": get_password_hash(password),
        "_id": str(uuid.uuid4())
    }
    _mock_users[clean_email] = user_doc
    return user_doc

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    clean_email = email.strip().lower()
    # Check if user actually exists in our memory dict
    return _mock_users.get(clean_email)
