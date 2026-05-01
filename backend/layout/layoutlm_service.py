"""
backend/layout/layoutlm_service.py
LayoutLMv3 spatial layout analysis with hybrid embedding strategy:
    - Primary: HF Inference API
    - Fallback: local transformers AutoProcessor + AutoModel
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

from PIL import Image

from backend.config import (
    HF_TOKEN,
    LAYOUTLM_API_URL,
    LAYOUTLM_ENABLE_LOCAL_FALLBACK,
    LAYOUTLM_LOCAL_MODEL_ID,
    LAYOUTLM_MAX_TOKENS,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)
_DISABLED_LAYOUT_ENDPOINTS: set[str] = set()
_LOCAL_LAYOUT_PROCESSOR = None
_LOCAL_LAYOUT_MODEL = None
_LOCAL_LAYOUT_DEVICE = "cpu"
_LOCAL_LAYOUT_INIT_ATTEMPTED = False


def _load_local_layout_model() -> bool:
    global _LOCAL_LAYOUT_PROCESSOR, _LOCAL_LAYOUT_MODEL, _LOCAL_LAYOUT_DEVICE, _LOCAL_LAYOUT_INIT_ATTEMPTED
    if _LOCAL_LAYOUT_PROCESSOR is not None and _LOCAL_LAYOUT_MODEL is not None:
        return True
    if _LOCAL_LAYOUT_INIT_ATTEMPTED:
        return False

    _LOCAL_LAYOUT_INIT_ATTEMPTED = True
    try:
        import torch
        from transformers import AutoModel, AutoProcessor

        _LOCAL_LAYOUT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[layoutlm] Loading local transformers model: %s (%s)", LAYOUTLM_LOCAL_MODEL_ID, _LOCAL_LAYOUT_DEVICE)

        _LOCAL_LAYOUT_PROCESSOR = AutoProcessor.from_pretrained(LAYOUTLM_LOCAL_MODEL_ID)
        _LOCAL_LAYOUT_MODEL = AutoModel.from_pretrained(LAYOUTLM_LOCAL_MODEL_ID).to(_LOCAL_LAYOUT_DEVICE)
        _LOCAL_LAYOUT_MODEL.eval()
        return True
    except Exception as exc:
        _LOCAL_LAYOUT_PROCESSOR = None
        _LOCAL_LAYOUT_MODEL = None
        logger.warning("[layoutlm] Local transformers model unavailable: %s", exc)
        return False


def _build_layout_payload(image: Image.Image, words: list[str], boxes: list[list[int]], max_items: int, max_dim: int, quality: int) -> dict:
    img = image.copy()
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    if img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "inputs": {
            "image": img_b64,
            "words": words[:max_items],
            "boxes": boxes[:max_items],
        }
    }

# ── Invoice keyword sets (unchanged from original) ────────────────────────────
HEADER_KEYWORDS = ("invoice", "bill to", "ship to", "seller", "vendor", "supplier", "gst", "tax invoice", "invoice no", "invoice number", "date")
TABLE_KEYWORDS  = ("item", "description", "qty", "quantity", "rate", "price", "amount", "unit", "hsn", "sku")
TOTALS_KEYWORDS = ("subtotal", "tax", "gst", "cgst", "sgst", "igst", "grand total", "total", "amount due", "balance due", "net amount")


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

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(urls))


def _call_layoutlm_api(image: Image.Image, words: list[str], boxes: list[list[int]]) -> list[float]:
    """
    Call LayoutLMv3 via HF Inference API for document embeddings.
    Returns a list of 8 float values (CLS embedding preview) or [] on failure.
    """
    if not HF_TOKEN:
        logger.info("[layoutlm] HF_TOKEN not set — skipping embedding call")
        return []
    if not words or not boxes:
        return []
    try:
        import requests

        payload_profiles = [
            _build_layout_payload(image, words, boxes, max_items=128, max_dim=900, quality=60),
            _build_layout_payload(image, words, boxes, max_items=64, max_dim=640, quality=45),
        ]

        for api_url in _candidate_hf_urls(LAYOUTLM_API_URL):
            if api_url in _DISABLED_LAYOUT_ENDPOINTS:
                continue

            for payload in payload_profiles:
                try:
                    resp = requests.post(
                        api_url,
                        headers={"Authorization": f"Bearer {HF_TOKEN}"},
                        json=payload,
                        timeout=60,
                    )
                except Exception as endpoint_exc:
                    logger.warning("[layoutlm] HF endpoint failed (%s): %s", api_url, endpoint_exc)
                    break

                status = resp.status_code
                text_preview = resp.text[:200] if resp.text else ""

                if status == 200:
                    data = resp.json()
                    # Feature-extraction returns [[token_embeds...]] shape
                    if isinstance(data, list) and data:
                        flat = data[0] if isinstance(data[0], list) else data
                        cls_embed = flat[0] if isinstance(flat[0], list) else flat
                        return [round(float(v), 4) for v in cls_embed[:8]]
                    logger.warning("[layoutlm] HF API response had no embedding payload via %s", api_url)
                    break

                if status in {400, 404} and ("Model not supported by provider" in text_preview or "Cannot POST /models/" in text_preview):
                    _DISABLED_LAYOUT_ENDPOINTS.add(api_url)
                    logger.warning("[layoutlm] Disabling unsupported endpoint %s: %s", api_url, text_preview)
                    break

                if status == 413:
                    logger.warning("[layoutlm] HF API payload too large via %s; trying smaller payload", api_url)
                    continue

                logger.warning("[layoutlm] HF API returned %s via %s: %s", status, api_url, text_preview)
                break

        if _DISABLED_LAYOUT_ENDPOINTS:
            logger.info("[layoutlm] Disabled endpoints: %s", ", ".join(sorted(_DISABLED_LAYOUT_ENDPOINTS)))
    except Exception as exc:
        logger.warning("[layoutlm] HF API call failed: %s", exc)
    return []


def _call_layoutlm_local_transformers(image: Image.Image, words: list[str], boxes: list[list[int]]) -> list[float]:
    """
    Local fallback using transformers AutoProcessor + AutoModel.
    Returns a short embedding preview compatible with API path.
    """
    if not LAYOUTLM_ENABLE_LOCAL_FALLBACK:
        return []
    if not words or not boxes:
        return []
    if not _load_local_layout_model():
        return []

    try:
        import torch

        token_words = words[:LAYOUTLM_MAX_TOKENS]
        token_boxes = boxes[:LAYOUTLM_MAX_TOKENS]

        encoding = _LOCAL_LAYOUT_PROCESSOR(
            image,
            token_words,
            boxes=token_boxes,
            truncation=True,
            padding="max_length",
            max_length=LAYOUTLM_MAX_TOKENS,
            return_tensors="pt",
        )
        for key, value in encoding.items():
            if hasattr(value, "to"):
                encoding[key] = value.to(_LOCAL_LAYOUT_DEVICE)

        with torch.no_grad():
            outputs = _LOCAL_LAYOUT_MODEL(**encoding)

        hidden_state = getattr(outputs, "last_hidden_state", None)
        if hidden_state is None or hidden_state.numel() == 0:
            return []

        cls_embedding = hidden_state[0, 0, :].detach().cpu().tolist()
        return [round(float(v), 4) for v in cls_embedding[:8]]
    except Exception as exc:
        logger.warning("[layoutlm] Local transformers inference failed: %s", exc)
        return []


# ── Pure-Python spatial analysis helpers (unchanged from original) ─────────────
def _guess_document_type(raw_text: str) -> str:
    lower = raw_text.lower()
    if "invoice" in lower: return "invoice"
    if "receipt" in lower: return "receipt"
    if lower.strip(): return "document"
    return "unknown"


def _clamp_box(box, width, height):
    left, top, right, bottom = box
    return [max(0, min(int(left), width)), max(0, min(int(top), height)),
            max(left, min(int(right), width)), max(top, min(int(bottom), height))]


def _merge_boxes(boxes, width, height):
    if not boxes: return [0, 0, width, height]
    return _clamp_box([min(b[0] for b in boxes), min(b[1] for b in boxes),
                       max(b[2] for b in boxes), max(b[3] for b in boxes)], width, height)


def _intersection_area(a, b):
    left, top = max(int(a[0]), int(b[0])), max(int(a[1]), int(b[1]))
    right, bottom = min(int(a[2]), int(b[2])), min(int(a[3]), int(b[3]))
    if right <= left or bottom <= top: return 0
    return (right - left) * (bottom - top)


def _box_area(box):
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _entry_belongs_to_region(entry_box, region_box):
    intersection = _intersection_area(entry_box, region_box)
    if intersection <= 0: return False
    entry_area = _box_area(entry_box)
    if entry_area <= 0: return False
    cx = (int(entry_box[0]) + int(entry_box[2])) / 2
    cy = (int(entry_box[1]) + int(entry_box[3])) / 2
    center_inside = (int(region_box[0]) <= cx <= int(region_box[2]) and
                     int(region_box[1]) <= cy <= int(region_box[3]))
    return center_inside or (intersection / entry_area) >= 0.5


def _find_keyword_boxes(entries, keywords, min_top=None):
    matches = []
    for entry in entries:
        text = str(entry.get("text", "")).strip().lower()
        box  = entry.get("bounding_box")
        if not text or not isinstance(box, list) or len(box) != 4: continue
        if min_top is not None and int(box[1]) < min_top: continue
        if any(kw in text for kw in keywords):
            matches.append([int(v) for v in box])
    return matches


def _build_layout_regions(width, height, entries):
    if height <= 0 or width <= 0: return []
    top_pos    = sorted(int(e["bounding_box"][1]) for e in entries if e.get("bounding_box"))
    bottom_pos = sorted(int(e["bounding_box"][3]) for e in entries if e.get("bounding_box"))

    header_limit = max(int(height * 0.18), top_pos[min(len(top_pos) - 1, max(len(top_pos) // 4, 0))] if top_pos else 0)
    footer_start = min(int(height * 0.8),  bottom_pos[max(0, len(bottom_pos) - max(len(bottom_pos) // 4, 1))] if bottom_pos else int(height * 0.8))

    if footer_start <= header_limit:
        header_limit, footer_start = int(height * 0.2), int(height * 0.8)

    header_bottom = max(0, min(header_limit, height))
    body_top      = min(header_bottom + 1, height)
    body_bottom   = max(min(footer_start, height), body_top)
    footer_top    = min(footer_start + 1, height)

    table_boxes = _find_keyword_boxes(entries, TABLE_KEYWORDS)
    totals_boxes = _find_keyword_boxes(entries, TOTALS_KEYWORDS, min_top=int(height * 0.35))

    table_top = min((int(b[1]) for b in table_boxes), default=body_top)
    table_bottom = max((int(b[3]) for b in table_boxes), default=body_bottom)
    totals_top = min((int(b[1]) for b in totals_boxes), default=footer_top)

    table_top = max(body_top, min(table_top, body_bottom))
    table_bottom = max(table_top, min(table_bottom, body_bottom))
    totals_top = max(body_top, min(totals_top, height))

    regions = [
        {"section": "header", "description": "Top region: invoice title, seller details, metadata",  "bounding_box": [0, 0, width, header_bottom]},
        {"section": "body",   "description": "Main content: billing details and item descriptions",   "bounding_box": [0, body_top, width, body_bottom]},
        {"section": "footer", "description": "Bottom region: totals, bank details, signature",        "bounding_box": [0, footer_top, width, height]},
    ]

    if table_boxes:
        regions.append({
            "section": "table",
            "description": "Detected line-item table region",
            "bounding_box": [0, table_top, width, table_bottom],
        })

    if totals_boxes:
        regions.append({
            "section": "totals",
            "description": "Detected totals/amount due region",
            "bounding_box": [0, totals_top, width, height],
        })

    return regions


def _build_structured_sections(layout_regions):
    structured = {}
    for region in layout_regions:
        section = str(region.get("section", "")).lower()
        lines = region.get("content_lines", []) or []
        structured[section] = {
            "line_count": len(lines),
            "preview": lines[:6],
        }
    return structured


def _group_entries_into_lines(entries):
    if not entries: return []
    sorted_entries = sorted(entries, key=lambda e: (int(e["bounding_box"][1]), int(e["bounding_box"][0])))
    avg_h = max(1, int(sum(max(1, int(e["bounding_box"][3]) - int(e["bounding_box"][1])) for e in sorted_entries) / len(sorted_entries)))
    threshold = max(12, avg_h // 2)
    grouped, current, current_cy = [], [], None
    for entry in sorted_entries:
        top, bottom = int(entry["bounding_box"][1]), int(entry["bounding_box"][3])
        cy = (top + bottom) / 2
        if current_cy is None or abs(cy - current_cy) <= threshold:
            current.append(entry)
            current_cy = sum((int(e["bounding_box"][1]) + int(e["bounding_box"][3])) / 2 for e in current) / len(current)
        else:
            grouped.append(current)
            current, current_cy = [entry], cy
    if current: grouped.append(current)
    return [
        " ".join(str(item.get("text", "")).strip() for item in sorted(line, key=lambda i: int(i["bounding_box"][0]))).strip()
        for line in grouped if any(str(item.get("text", "")).strip() for item in line)
    ]


def _attach_region_content(regions, entries):
    enriched = []
    for region in regions:
        rbox = region.get("bounding_box")
        if not isinstance(rbox, list) or len(rbox) != 4:
            enriched.append(region)
            continue
        region_entries = sorted(
            [e for e in entries if str(e.get("text", "")).strip() and isinstance(e.get("bounding_box"), list) and len(e["bounding_box"]) == 4 and _entry_belongs_to_region(e["bounding_box"], rbox)],
            key=lambda e: (int(e["bounding_box"][1]), int(e["bounding_box"][0])),
        )
        enriched.append({
            **region,
            "content_words": [str(e.get("text", "")).strip() for e in region_entries],
            "content_text":  " ".join(str(e.get("text", "")).strip() for e in region_entries).strip(),
            "content_lines": _group_entries_into_lines(region_entries),
        })
    return enriched


def _detect_layout_blocks(width, height, entries):
    blocks = []
    table_boxes  = _find_keyword_boxes(entries, TABLE_KEYWORDS)
    totals_boxes = _find_keyword_boxes(entries, TOTALS_KEYWORDS, min_top=int(height * 0.45))
    header_boxes = _find_keyword_boxes(entries, HEADER_KEYWORDS)

    if len(table_boxes) >= 2:
        blocks.append({"block_type": "table_region",  "bounding_box": _merge_boxes(table_boxes, width, height)})
    if totals_boxes:
        blocks.append({"block_type": "totals_region", "bounding_box": _merge_boxes(totals_boxes, width, height)})
    if header_boxes:
        blocks.append({"block_type": "key_text_block", "label": "invoice_header_block", "bounding_box": _merge_boxes(header_boxes, width, height)})

    seen, unique = set(), []
    for b in blocks:
        key = (b.get("block_type"), b.get("label"), tuple(b["bounding_box"]))
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def _build_document_layout_analysis(image, entries, layoutlm_status):
    width, height = image.size
    layout_regions  = _attach_region_content(_build_layout_regions(width, height, entries), entries)
    detected_blocks = _attach_region_content(_detect_layout_blocks(width, height, entries), entries)
    structured_sections = _build_structured_sections(layout_regions)
    layout_executed = bool(layout_regions)
    model_executed  = bool(layoutlm_status.get("executed"))
    fallback_used   = bool(layoutlm_status.get("fallback_used"))

    model_source = str(layoutlm_status.get("source", "Hugging Face API"))

    if model_executed:
        note = "Layout structure derived using OCR positions, bounding boxes, and LayoutLMv3 embeddings."
    elif layout_executed:
        note = "Layout structure derived using OCR positions and bounding box spatial analysis (LayoutLMv3 embedding fallback)."
    else:
        note = "Layout analysis could not derive reliable regions from available OCR output."

    return {
        "model_source": model_source,
        "model_name": MODEL_NAME,
        "execution_status": "success" if layout_executed else "failed",
        "model_execution_status": "success" if model_executed else ("fallback" if fallback_used else "skipped"),
        "layout_regions": layout_regions,
        "detected_blocks": detected_blocks,
        "structured_sections": structured_sections,
        "note": note,
    }


def analyze_document_layout(
    image: Image.Image,
    words: list[str],
    boxes: list[list[int]],
    entries: list[dict[str, Any]],
    raw_text: str,
) -> dict[str, Any]:
    """
        Analyse document layout using:
            1. HF Inference API call for LayoutLMv3 embeddings
            2. Local transformers embedding fallback (optional)
            3. Pure-Python spatial region + block detection (always runs)

    Args:
        image:    PIL Image of the document
        words:    list of text strings from OCR
        boxes:    list of [left,top,right,bottom] in 0–1000 scale (for LayoutLMv3)
        entries:  list of dicts with 'text' + 'bounding_box' (pixel coords) for spatial analysis
        raw_text: full concatenated OCR text
    """
    if not words or not boxes:
        note = "LayoutLMv3 skipped — OCR produced no usable words/boxes."
        layoutlm_status = {
            "enabled": True, "source": "Hugging Face API / Local Transformers",
            "mode": "layout-aware document understanding", "model_name": MODEL_NAME,
            "executed": False, "fallback_used": True, "note": note,
        }
        return {
            "document_type": _guess_document_type(raw_text),
            "token_count": 0, "model_name": MODEL_NAME,
            "embedding_preview": [], "note": note,
            "layoutlmv3_status": layoutlm_status,
            "document_layout_analysis": _build_document_layout_analysis(image, entries, layoutlm_status),
        }

    embedding_preview = _call_layoutlm_api(image, words, boxes)
    source = "Hugging Face API"
    if not embedding_preview:
        local_preview = _call_layoutlm_local_transformers(image, words, boxes)
        if local_preview:
            embedding_preview = local_preview
            source = "Local Transformers"

    executed = bool(embedding_preview)

    note = (
        f"LayoutLMv3 embedding retrieved via {source}. "
        "Spatial layout regions built from OCR bounding boxes."
    ) if executed else (
        "LayoutLMv3 embedding call skipped or failed — running spatial analysis using OCR bounding boxes only."
    )

    layoutlm_status = {
        "enabled": True, "source": source,
        "mode": "layout-aware document understanding", "model_name": MODEL_NAME,
        "executed": executed, "fallback_used": not executed, "note": note,
    }

    logger.info("[layoutlm] executed=%s, embedding_preview=%s", executed, embedding_preview)

    return {
        "document_type": _guess_document_type(raw_text),
        "token_count": len(words), "model_name": MODEL_NAME,
        "embedding_preview": embedding_preview, "note": note,
        "layoutlmv3_status": layoutlm_status,
        "document_layout_analysis": _build_document_layout_analysis(image, entries, layoutlm_status),
    }
