"""
backend/vlm/hf_client.py
HTTP client for Hugging Face Inference API.
Handles LLaVA (primary VLM) and BLIP-2 (cloud fallback).
Source: vlm_llava_project/model/hf_client.py (token now from config)
"""
from __future__ import annotations

import base64
import time

import logging
import requests

from backend.config import BLIP2_URL, HF_TOKEN, LLAVA_URL

logger = logging.getLogger(__name__)
_DISABLED_VLM_ENDPOINTS: set[str] = set()


def _get_headers() -> dict:
    return {"Authorization": f"Bearer {HF_TOKEN}"}


def _candidate_hf_urls(primary_url: str) -> list[str]:
    urls = [primary_url]

    router_marker = "/hf-inference/models/"
    legacy_marker = "/models/"

    if router_marker in primary_url:
        model_id = primary_url.split(router_marker, 1)[1].strip("/")
        urls.append(f"https://api-inference.huggingface.co/models/{model_id}")
    elif "api-inference.huggingface.co" in primary_url and legacy_marker in primary_url:
        model_id = primary_url.split(legacy_marker, 1)[1].strip("/")
        urls.append(f"https://router.huggingface.co/hf-inference/models/{model_id}")

    return list(dict.fromkeys(urls))


def _is_unsupported_endpoint(status_code: int, err_msg: str) -> bool:
    lower = (err_msg or "").lower()
    if status_code in {400, 404}:
        return (
            "model not supported by provider" in lower
            or "cannot post /models/" in lower
            or "not found" in lower
        )
    return False


def _query_model(url: str, image_bytes: bytes, prompt: str) -> str:
    """
    Generic HF Inference API caller with 3-attempt retry.
    Returns generated text or an error sentinel string.
    """
    if not HF_TOKEN:
        return "HF_ERROR_NO_TOKEN"

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    final_prompt = f"USER: <image>\n{prompt}\nASSISTANT:" if "llava" in url.lower() else prompt
    payload = {"inputs": final_prompt, "image": image_b64}

    candidate_urls = _candidate_hf_urls(url)
    last_error = "HF_ERROR"

    for attempt in range(3):
        should_retry = False
        active_endpoint_seen = False

        for endpoint in candidate_urls:
            if endpoint in _DISABLED_VLM_ENDPOINTS:
                continue

            active_endpoint_seen = True
            try:
                response = requests.post(endpoint, headers=_get_headers(), json=payload, timeout=120)

                if response.status_code != 200:
                    try:
                        data = response.json()
                        err_msg = str(data.get("error", "Unknown error"))
                    except Exception:
                        err_msg = response.text[:200]

                    if "loading" in err_msg.lower():
                        logger.info("[hf_client] Attempt %d via %s: model loading, retrying in 5s", attempt + 1, endpoint)
                        should_retry = True
                        continue

                    if _is_unsupported_endpoint(response.status_code, err_msg):
                        _DISABLED_VLM_ENDPOINTS.add(endpoint)
                        logger.warning("[hf_client] Disabling unsupported endpoint %s: %s", endpoint, err_msg)
                        last_error = f"HF_ERROR_{response.status_code}_{err_msg}"
                        continue

                    logger.error("[hf_client] API Failure via %s (Status %d): %s", endpoint, response.status_code, err_msg)
                    last_error = f"HF_ERROR_{response.status_code}_{err_msg}"
                    continue

                data = response.json()
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                return str(data)

            except Exception as exc:
                logger.error("[hf_client] Attempt %d via %s failed: %s", attempt + 1, endpoint, exc)
                last_error = f"HF_ERROR_{exc}"

        if not active_endpoint_seen:
            return "HF_ERROR_UNSUPPORTED_ENDPOINT"

        if should_retry or attempt < 2:
            time.sleep(5)

    return last_error


def query_llava(image_bytes: bytes, prompt: str) -> str:
    res = _query_model(LLAVA_URL, image_bytes, prompt)
    return res.replace("HF_ERROR", "LLAVA_ERROR") if res.startswith("HF_ERROR") else res


def query_huggingface(image_bytes: bytes, prompt: str) -> str:
    return _query_model(BLIP2_URL, image_bytes, prompt)
