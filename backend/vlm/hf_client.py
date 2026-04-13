"""
backend/vlm/hf_client.py
HTTP client for Hugging Face Inference API.
Handles LLaVA (primary VLM) and BLIP-2 (cloud fallback).
Source: vlm_llava_project/model/hf_client.py (token now from config)
"""
from __future__ import annotations

import base64
import time

import requests

from backend.config import BLIP2_URL, HF_TOKEN, LLAVA_URL


def _get_headers() -> dict:
    return {"Authorization": f"Bearer {HF_TOKEN}"}


def _query_model(url: str, image_bytes: bytes, prompt: str) -> str:
    """
    Generic HF Inference API caller with 3-attempt retry.
    Returns generated text or an error sentinel string.
    """
    for attempt in range(3):
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            if "llava" in url.lower():
                final_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            else:
                final_prompt = prompt

            payload = {"inputs": final_prompt, "image": image_b64}
            response = requests.post(url, headers=_get_headers(), json=payload, timeout=120)
            data = response.json()

            if isinstance(data, dict) and "error" in data:
                if "loading" in str(data["error"]).lower():
                    time.sleep(5)
                    continue
                return f"HF_ERROR_{data['error']}"

            if response.status_code == 200:
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                return str(data)

        except Exception as exc:
            print(f"[hf_client] Attempt {attempt + 1} failed: {exc}")

        time.sleep(5)

    return "HF_ERROR"


def query_llava(image_bytes: bytes, prompt: str) -> str:
    res = _query_model(LLAVA_URL, image_bytes, prompt)
    return res.replace("HF_ERROR", "LLAVA_ERROR") if res.startswith("HF_ERROR") else res


def query_huggingface(image_bytes: bytes, prompt: str) -> str:
    return _query_model(BLIP2_URL, image_bytes, prompt)
