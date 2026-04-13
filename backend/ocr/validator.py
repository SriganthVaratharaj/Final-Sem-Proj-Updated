"""
backend/ocr/validator.py
Per-image clarity validation before OCR. Checks: blur, brightness, contrast,
minimum size, and edge density (proxy for readable text presence).
Source: ocr-invoice-system/src/image_validator.py
"""
import cv2
import numpy as np

BLUR_THRESHOLD = 30.0
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 245.0
MIN_CONTRAST_STD = 10.0
MIN_WIDTH = 80
MIN_HEIGHT = 80
MIN_EDGE_DENSITY = 0.005


def _load_gray(image_input):
    if isinstance(image_input, dict):
        img = image_input.get("default")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        img = cv2.imread(str(image_input))

    if img is None:
        raise ValueError(f"Could not read image: {image_input}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img


def validate_image_clarity(image_input):
    """
    Returns (is_clear: bool, reason: str | None).
    is_clear=True means proceed to OCR.
    """
    try:
        gray = _load_gray(image_input)
    except ValueError as exc:
        return False, str(exc)

    h, w = gray.shape

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False, f"Image too small ({w}×{h} px); minimum is {MIN_WIDTH}×{MIN_HEIGHT} px."

    mean_brightness = float(np.mean(gray))
    if mean_brightness < MIN_BRIGHTNESS:
        return False, f"Image too dark (brightness {mean_brightness:.1f}/{MIN_BRIGHTNESS})."
    if mean_brightness > MAX_BRIGHTNESS:
        return False, f"Image overexposed (brightness {mean_brightness:.1f}/{MAX_BRIGHTNESS})."

    std_brightness = float(np.std(gray))
    if std_brightness < MIN_CONTRAST_STD:
        return False, f"Insufficient contrast (std-dev {std_brightness:.1f}/{MIN_CONTRAST_STD})."

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < BLUR_THRESHOLD:
        return False, f"Image too blurry (Laplacian {laplacian_var:.2f}/{BLUR_THRESHOLD})."

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = float(np.count_nonzero(edges)) / (w * h)
    if edge_density < MIN_EDGE_DENSITY:
        return False, f"No readable text regions detected (edge density {edge_density:.4f})."

    return True, None
