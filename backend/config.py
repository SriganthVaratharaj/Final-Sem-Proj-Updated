"""
backend/config.py
Central configuration for all three AI modules.
All sensitive values are loaded from .env — never hardcoded.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
# The root is the parent of the backend/ folder
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "frontend" / "uploads"
OUTPUT_DIR = BASE_DIR / "db" / "outputs"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

# ── API metadata ──────────────────────────────────────────────────────────────
API_TITLE = "Invoice AI — Combined OCR + Layout + VLM System"
API_DESCRIPTION = (
    "Unified pipeline: PaddleOCR (multi-language) → "
    "LayoutLMv3 (spatial layout, via HF API) → "
    "LLaVA/BLIP-2 (VLM field extraction, via HF API)."
)
API_VERSION = "2.0.0"

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"}

# ── Hugging Face token (used for all three HF API calls) ─────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ── Module 1: PaddleOCR (runs locally) ───────────────────────────────────────
PADDLE_AUTO_LANGUAGES: list[str] = ["latin", "ta", "hi", "te", "kn"]

# ── Module 2: LayoutLMv3 (via HF Inference API) ──────────────────────────────
LAYOUTLM_API_URL: str = (
    "https://api-inference.huggingface.co/models/microsoft/layoutlmv3-base"
)
LAYOUTLM_MAX_TOKENS: int = 512
# Keep for report_generator compatibility
MODEL_NAME: str = "microsoft/layoutlmv3-base"

# ── Module 3: VLM — LLaVA primary (via HF Router) ────────────────────────────
LLAVA_URL: str = (
    "https://router.huggingface.co/hf-inference/models/llava-hf/llava-v1.6-mistral-7b-hf"
)
# BLIP-2 cloud fallback
BLIP2_URL: str = (
    "https://api-inference.huggingface.co/models/Salesforce/blip2-flan-t5-xl"
)

# ── image preprocessing constants ─────────────────────────────────────────────
IMAGE_SIZE = (1000, 1000)   # used by layoutlm_service normalisation
