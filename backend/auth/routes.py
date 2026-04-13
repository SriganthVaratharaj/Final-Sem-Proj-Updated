"""
backend/auth/routes.py
API endpoints for authentication (login and register).
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends, Header
from pydantic import BaseModel
import jwt

from db.auth_repository import create_user, get_user_by_email, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])

SECRET_KEY = os.getenv("JWT_SECRET", "super_secret_dev_key_123")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

async def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        return email
    except jwt.PyJWTError:
        return None

class AuthRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    message: str
    token: str
    email: str

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@router.post("/register", response_model=AuthResponse)
async def register(request: AuthRequest):
    try:
        await create_user(request.email, request.password)
        # Auto login after register
        access_token = create_access_token(
            data={"sub": request.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return AuthResponse(message="User created and logged in", token=access_token, email=request.email)
    except ValueError as e:
        # Instead of generic 400, providing clear info that the account exists
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account already exists. Please login instead."
        )

@router.post("/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    user = await get_user_by_email(request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account does not exist. Please sign up."
        )
    if not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(
        data={"sub": request.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return AuthResponse(message="Logged in successfully", token=access_token, email=request.email)
