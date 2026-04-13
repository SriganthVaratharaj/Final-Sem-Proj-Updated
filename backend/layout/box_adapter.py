"""
backend/layout/box_adapter.py
Converts PaddleOCR polygon bounding boxes to the formats required by
LayoutLMv3 (normalized 0–1000) and spatial analysis (pixel axis-aligned rect).

PaddleOCR returns polygon points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
LayoutLMv3 requires axis-aligned:  [left, top, right, bottom]  (0–1000 scale)
Spatial analysis requires:          [left, top, right, bottom]  (pixel coords)
"""
from __future__ import annotations


def paddle_boxes_to_layoutlm(paddle_boxes: list, image_size: tuple[int, int]) -> list[list[int]]:
    """
    Convert PaddleOCR polygon boxes → LayoutLMv3 normalized boxes (0–1000 scale).

    Args:
        paddle_boxes: list of polygon boxes [[x,y],[x,y],[x,y],[x,y]]
        image_size:   (width, height) of the PIL image

    Returns:
        list of [left, top, right, bottom] in 0–1000 scale
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return [[0, 0, 0, 0]] * len(paddle_boxes)

    result = []
    for box in paddle_boxes:
        try:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            result.append([
                max(0, min(int(min(xs) / width  * 1000), 1000)),
                max(0, min(int(min(ys) / height * 1000), 1000)),
                max(0, min(int(max(xs) / width  * 1000), 1000)),
                max(0, min(int(max(ys) / height * 1000), 1000)),
            ])
        except (TypeError, ZeroDivisionError):
            result.append([0, 0, 0, 0])
    return result


def paddle_boxes_to_pixel_rect(paddle_boxes: list) -> list[list[int]]:
    """
    Convert PaddleOCR polygon boxes → axis-aligned pixel coordinate rects.

    Returns:
        list of [left, top, right, bottom] in pixel coordinates
    """
    result = []
    for box in paddle_boxes:
        try:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            result.append([
                int(min(xs)),
                int(min(ys)),
                int(max(xs)),
                int(max(ys)),
            ])
        except (TypeError, ValueError):
            result.append([0, 0, 0, 0])
    return result


def build_entries(
    texts: list[str],
    pixel_rects: list[list[int]],
    lm_boxes: list[list[int]],
    confidences: list[float],
) -> list[dict]:
    """
    Build the entries list expected by layoutlm_service spatial analysis functions.
    Each entry: { text, bounding_box (pixel), normalized_box (0-1000), confidence }
    """
    entries = []
    for text, pbox, nbox, conf in zip(texts, pixel_rects, lm_boxes, confidences):
        if not str(text).strip():
            continue
        entries.append({
            "text": str(text).strip(),
            "bounding_box": pbox,
            "normalized_box": nbox,
            "confidence": round(float(conf), 4),
        })
    return entries
