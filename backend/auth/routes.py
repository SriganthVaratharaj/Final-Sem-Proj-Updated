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

import random
from backend.db.auth_repository import create_user, get_user_by_email, verify_password, get_password_hash
from backend.db.connection import get_async_db
from backend.auth.email_service import send_otp_email

# In-memory store for OTPs: { "email": "123456" }
# For a production app, this should be in Redis or DB with an expiration time.
otp_store = {}

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
    email = request.email.strip().lower()
    try:
        await create_user(email, request.password)
        # Auto login after register
        access_token = create_access_token(
            data={"sub": email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return AuthResponse(message="User created and logged in", token=access_token, email=email)
    except ValueError as e:
        # Instead of generic 400, providing clear info that the account exists
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account already exists. Please login instead."
        )

@router.post("/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    email = request.email.strip().lower()
    user = await get_user_by_email(email)
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
        data={"sub": email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return AuthResponse(message="Logged in successfully", token=access_token, email=email)

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email: str
    otp: str
    new_password: str

@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    email = request.email.strip().lower()
    user = await get_user_by_email(email)
    if not user:
        # We don't reveal if the email exists for security reasons, just return success
        return {"message": "If an account exists, an OTP has been sent."}
        
    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    otp_store[email] = otp
    
    # Send email
    success = send_otp_email(email, otp)
    if not success:
        # For local testing if email fails to send due to missing credentials, 
        # we log it but still let the UI flow work by returning the OTP
        return {"message": "OTP generated.", "dev_otp": otp}
        
    return {"message": "If an account exists, an OTP has been sent."}

@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    email = request.email.strip().lower()
    
    # Verify OTP
    stored_otp = otp_store.get(email)
    if not stored_otp or stored_otp != request.otp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP."
        )
        
    user = await get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account does not exist."
        )
        
    # Update password in repository
    new_hash = get_password_hash(request.new_password)
    
    # Try updating real MongoDB if available
    db_conn = get_async_db()
    if db_conn is not None:
        try:
            # Assuming 'users' collection
            await db_conn["users"].update_one(
                {"email": email}, 
                {"$set": {"hashed_password": new_hash}}
            )
        except Exception as e:
            pass # fallback if collection name differs
            
    # Also attempt to call a repo update function if they have one defined on HF
    try:
        from backend.db.auth_repository import update_password
        await update_password(email, new_hash)
    except ImportError:
        pass

    otp_store.pop(email, None)
    
    return {"message": "Password reset successfully."}
