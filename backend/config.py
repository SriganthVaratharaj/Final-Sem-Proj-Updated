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


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# ── Paths ─────────────────────────────────────────────────────────────────────
# The root is the parent of the backend/ folder
BASE_DIR = Path(__file__).resolve().parent.parent
HF_MODELS_DIR = BASE_DIR / "hf_models"
LAYOUTLM_LOCAL_DIR = HF_MODELS_DIR / "Layoutlmv3"
VLM_LOCAL_DIR = HF_MODELS_DIR / "llava"
PADDLE_VL_LOCAL_DIR = HF_MODELS_DIR / "PaadleOCR"
UPLOAD_DIR = BASE_DIR / "db" / "uploads"
OUTPUT_DIR = BASE_DIR / "db" / "outputs"
GUEST_DIR = BASE_DIR / "db" / "tmp"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

# ── API metadata ──────────────────────────────────────────────────────────────
API_TITLE = "Invoice AI — Combined OCR + Layout + VLM System"
API_DESCRIPTION = (
    "Unified pipeline: PaddleOCR (multi-language) → "
    "LayoutLMv3 (spatial layout, HF API with local transformers fallback) → "
    "LLaVA/BLIP-2 (VLM field extraction, HF API with local transformers fallback)."
)
API_VERSION = "2.0.0"

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"}

# ── Hugging Face & Internal Keys ──────────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
INTERNAL_MODEL_API_KEY: str = os.getenv("INTERNAL_MODEL_API_KEY", "inv_ai_local_default_key")
KAGGLE_VLM_URL: str = os.getenv("KAGGLE_VLM_URL", "")


# ── Module 1: PaddleOCR (runs locally) ───────────────────────────────────────
PADDLE_AUTO_LANGUAGES: list[str] = _parse_csv_env(
    "PADDLE_AUTO_LANGUAGES",
    ["latin", "devanagari", "ta", "te", "ka"],
)
OCR_FORCE_DOMINANT_LANGUAGE: bool = os.getenv("OCR_FORCE_DOMINANT_LANGUAGE", "true").lower() in {"1", "true", "yes", "on"}

# ── Module 2: LayoutLMv3 (via HF Inference API) ──────────────────────────────
LAYOUTLM_API_URL: str = (
    "https://router.huggingface.co/hf-inference/models/microsoft/layoutlmv3-base"
)
LAYOUTLM_MAX_TOKENS: int = 512
# Keep for report_generator compatibility
MODEL_NAME: str = "microsoft/layoutlmv3-base"
LAYOUTLM_ENABLE_LOCAL_FALLBACK: bool = _parse_bool_env("LAYOUTLM_ENABLE_LOCAL_FALLBACK", True)
LAYOUTLM_LOCAL_MODEL_ID: str = os.getenv(
    "LAYOUTLM_LOCAL_MODEL_ID",
    str(LAYOUTLM_LOCAL_DIR) if LAYOUTLM_LOCAL_DIR.exists() else MODEL_NAME,
)

# ── Module 3: VLM — LLaVA primary (via HF Router) ────────────────────────────
LLAVA_URL: str = (
    "https://router.huggingface.co/hf-inference/models/llava-hf/llava-v1.6-mistral-7b-hf"
)
# BLIP-2 cloud fallback
BLIP2_URL: str = (
    "https://router.huggingface.co/hf-inference/models/Salesforce/blip2-flan-t5-xl"
)
VLM_REQUIRED_FIELDS: list[str] = _parse_csv_env(
    "VLM_REQUIRED_FIELDS",
    ["vendor_name", "invoice_no", "date", "phone", "sub_total", "tax", "grand_total"],
)
VLM_ENABLE_LOCAL_TRANSFORMERS: bool = _parse_bool_env("VLM_ENABLE_LOCAL_TRANSFORMERS", True)
VLM_LOCAL_MODEL_ID: str = os.getenv(
    "VLM_LOCAL_MODEL_ID",
    str(VLM_LOCAL_DIR) if VLM_LOCAL_DIR.exists() else "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
)
VLM_LOCAL_MAX_NEW_TOKENS: int = int(os.getenv("VLM_LOCAL_MAX_NEW_TOKENS", "256"))

# ── GGUF Paths (LEGACY LOCAL FALLBACK) ────────────────────────────────────────
# English/Latin invoices → Qwen3 VL 4B Q4_K_M (fast, accurate for Latin scripts)
LLAVA_GGUF_PATH = HF_MODELS_DIR / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
LLAVA_MMPROJ_PATH = HF_MODELS_DIR / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"

# Indic language invoices → MiniCPM-V 2.6 Q4_K_M (multilingual, stronger for Indian scripts)
MINICPM_GGUF_PATH = HF_MODELS_DIR / "MiniCPM-V-2_6-Q4_K_M.gguf"
MINICPM_MMPROJ_PATH = HF_MODELS_DIR / "mmproj-MiniCPM-V-2_6-f16.gguf"

PADDLE_VL_GGUF_PATH = PADDLE_VL_LOCAL_DIR / "PaddleOCR-VL-1.5.gguf"
PADDLE_VL_MMPROJ_PATH = PADDLE_VL_LOCAL_DIR / "PaddleOCR-VL-1.5-mmproj.gguf"

# ── image preprocessing constants ─────────────────────────────────────────────
IMAGE_SIZE = (1000, 1000)   # used by layoutlm_service normalisation

VLM_LOCAL_N_CTX = 8192           # Increased for Q4_K_M — more room for image + prompt
VLM_LOCAL_MAX_NEW_TOKENS = 1024  # Full invoice output fits comfortably
