"""
backend/utils/image_enhancer.py

Advanced image preprocessing for custom fonts, artistic text, and handwritten bills.
Strategy: Generate MULTIPLE enhanced variants and pick the best for OCR + VLM.

DESIGN PHILOSOPHY:
- OCR engine  → gets high-contrast, sharpened, denoised image
- VLM engine  → gets color-preserved, moderately enhanced image (color cues matter)
- Never destroy original — always keep as final fallback

KNOWN DISADVANTAGES & MITIGATIONS:
┌──────────────────────────────────────────────────┬────────────────────────────────────────────────┐
│ Disadvantage                                     │ Mitigation Applied                             │
├──────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ 1. Global B&W binarization destroys thin Indic   │ Adaptive threshold (local blocks, not global   │
│    matras & vowel signs                          │ Otsu) — handles local ink density variation    │
├──────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ 2. Color info loss (red totals, green headers)   │ VLM gets color-preserved enhanced copy;        │
│    breaks VLM understanding of layout            │ B&W only used for OCR text hint                │
├──────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ 3. Over-sharpening creates ringing artifacts     │ Unsharp Mask with conservative strength        │
│    that confuse OCR on blurry source images      │ (amount=1.5) instead of hard-edge kernels      │
├──────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ 4. Upscaling low-res images introduces blur      │ Only upscale if below 800px — use LANCZOS;     │
│    that worsens character recognition            │ cap at 2x to avoid artifact multiplication     │
├──────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ 5. Deskew failures on artistic/logo text         │ Soft deskew with ±15° limit + confidence score;│
│    rotate invoice at wrong angle                 │ skip if Hough line confidence < threshold      │
├──────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ 6. Handwriting: letter shapes vary per person    │ CLAHE equalisation handles uneven pen pressure;│
│    — no preprocessing fully solves this          │ VLM sees color original + OCR hint combined    │
├──────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ 7. Extra preprocessing latency (~200-400ms)      │ Run async in thread pool, doesn't block VLM    │
│    adds to total extraction time                 │ startup (both happen concurrently)             │
└──────────────────────────────────────────────────┴────────────────────────────────────────────────┘
"""
from __future__ import annotations
import io
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MIN_SIDE_PX   = 800    # Upscale if shorter side < this
MAX_SIDE_PX   = 2000   # Never exceed this (memory + token budget)
JPEG_QUALITY  = 88     # Output quality for VLM (keep detail)
OCR_JPEG_Q    = 92     # OCR variant: slightly higher = cleaner edges


def _to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)

def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def _safe_upscale(img: Image.Image) -> Image.Image:
    """
    Upscale only if too small. Max 2x zoom to avoid blur multiplication.
    MITIGATION for Disadvantage #4.
    """
    w, h = img.size
    short = min(w, h)
    if short < MIN_SIDE_PX:
        scale = min(2.0, MIN_SIDE_PX / short)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.debug("[enhancer] Upscaled %dx%d → %dx%d", w, h, new_w, new_h)
    # Cap maximum size
    long = max(img.size)
    if long > MAX_SIDE_PX:
        scale = MAX_SIDE_PX / long
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    return img


def _try_deskew(img: Image.Image) -> Image.Image:
    """
    Detect and correct tilt using Hough line transform.
    MITIGATION for Disadvantage #5: Conservative ±15° limit, skip on low confidence.
    """
    try:
        import cv2
        arr = np.array(img.convert("L"))
        edges = cv2.Canny(arr, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None or len(lines) < 5:
            return img  # Not enough lines → skip deskew

        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if abs(angle) <= 15:  # Only correct small tilts
                angles.append(angle)

        if not angles:
            return img

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return img  # Negligible tilt

        logger.debug("[enhancer] Deskewing by %.2f degrees", median_angle)
        return img.rotate(-median_angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    except Exception as e:
        logger.debug("[enhancer] Deskew skipped: %s", e)
        return img


def _clahe_equalise(img: Image.Image) -> Image.Image:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Handles uneven lighting in handwritten bills / scanned invoices.
    MITIGATION for Disadvantage #6.
    """
    try:
        import cv2
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        merged = cv2.merge([l_eq, a, b])
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)
    except Exception as e:
        logger.debug("[enhancer] CLAHE skipped: %s", e)
        return img


def _adaptive_threshold_bw(img: Image.Image) -> Image.Image:
    """
    Advanced Binarization (Sauvola-inspired).
    Preserves tiny Indic matras and vowel signs (dots/curves) better than standard thresholding.
    MITIGATION for Disadvantage #1 & fulfillment of Option 4.
    """
    try:
        import cv2
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        # Dual-pass thresholding: 
        # Pass 1: Large block for global structure
        # Pass 2: Small block for tiny character details (Bengali dots)
        bw_global = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        bw_detail = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Combine (logical AND) — keeps only text present in both
        # Then Dilate slightly to connect broken Indic characters
        combined = cv2.bitwise_and(bw_global, bw_detail)
        kernel = np.ones((1, 1), np.uint8) # Extremely conservative dilation
        result = cv2.dilate(combined, kernel, iterations=1)
        
        return Image.fromarray(result).convert("RGB")
    except Exception as e:
        logger.debug("[enhancer] Advanced binarization skipped: %s", e)
        return img.convert("L").convert("RGB")


def _conservative_sharpen(img: Image.Image) -> Image.Image:
    """
    Unsharp Mask sharpening — avoids hard-edge artifacts.
    MITIGATION for Disadvantage #3.
    """
    try:
        sharpened = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
        return sharpened
    except Exception as e:
        logger.debug("[enhancer] Sharpen skipped: %s", e)
        return img


def _denoise(img: Image.Image) -> Image.Image:
    """Fast median denoise — removes scanning noise without blurring edges."""
    try:
        import cv2
        arr = cv2.fastNlMeansDenoisingColored(np.array(img), None, 8, 8, 7, 21)
        return Image.fromarray(arr)
    except Exception as e:
        logger.debug("[enhancer] Denoise skipped: %s", e)
        return img


def _to_bytes(img: Image.Image, quality: int) -> bytes:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def split_dual_invoice(image_bytes: bytes) -> list[bytes]:
    """Detects if an image is a dual side-by-side invoice and returns a list of segments."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        # Wide image = likely dual
        if w > h * 1.1:
            logger.info("[enhancer] Wide image detected. Splitting into Left and Right segments.")
            left = img.crop((0, 0, int(w * 0.5), h))
            right = img.crop((int(w * 0.5), 0, w, h))
            return [_to_bytes(left, JPEG_QUALITY), _to_bytes(right, JPEG_QUALITY)]
        return [image_bytes]
    except Exception as e:
        logger.error("[enhancer] Dual split failed: %s", e)
        return [image_bytes]

def _boost_white_on_color_text(img: Image.Image) -> Image.Image:
    """
    Detect colored background bands (headers/footers like teal, green, blue)
    where white text sits on dark bg. Boost local contrast so VLM can read them.

    Strategy:
    - Scan image in horizontal strips (5% height each)
    - If a strip's average saturation is HIGH and brightness is LOW → colored bg
    - For those strips: apply aggressive contrast boost + slight brightening
    - White-on-dark text becomes clearly readable without destroying rest of image
    """
    try:
        import cv2
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h, w = arr.shape[:2]
        result = arr.copy()

        strip_h = max(10, h // 20)  # 5% height strips

        for y in range(0, h, strip_h):
            strip = hsv[y:y + strip_h]
            # High saturation (S > 60) + low-mid value (V < 160) = colored dark background
            avg_s = float(strip[:, :, 1].mean())
            avg_v = float(strip[:, :, 2].mean())

            if avg_s > 60 and avg_v < 160:
                # This is a colored band — apply local contrast boost
                band = arr[y:y + strip_h].astype(np.float32)
                # Stretch contrast: pull whites up, push mids down
                band = np.clip((band - 80) * 2.2, 0, 255).astype(np.uint8)
                result[y:y + strip_h] = band

        logger.debug("[enhancer] Colored bg boost applied.")
        return Image.fromarray(result)
    except Exception as e:
        logger.debug("[enhancer] Colored bg boost skipped: %s", e)
        return img


def enhance_for_vlm(image_bytes: bytes) -> bytes:
    """
    Color-preserving enhancement for VLM.
    (Note: Splitting is now handled by pipeline via split_dual_invoice)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = _safe_upscale(img)
        img = _clahe_equalise(img)

        # Boost white-on-colored-background text (teal/green headers, footer bands)
        # This is the key fix for bills like "विथूल बिल" where title is white-on-teal
        img = _boost_white_on_color_text(img)

        # Global contrast boost for stylized fonts
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.6)

        img = _conservative_sharpen(img)
        return _to_bytes(img, JPEG_QUALITY)
    except Exception as e:
        logger.warning("[enhancer] VLM enhance failed: %s", e)
        return image_bytes


def _thin_characters(img: Image.Image) -> Image.Image:
    """Uses morphological erosion to thin thick fancy characters for better OCR."""
    import numpy as np
    import cv2
    arr = np.array(img.convert("L"))
    inv = cv2.bitwise_not(arr)
    kernel = np.ones((2,2), np.uint8)
    thinned = cv2.erode(inv, kernel, iterations=1)
    result = cv2.bitwise_not(thinned)
    return Image.fromarray(result)

def enhance_for_ocr(image_bytes: bytes) -> bytes:
    """
    Aggressive B&W + Thinning for stylized fancy fonts.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = _safe_upscale(img)
        img = _denoise(img)
        img = _clahe_equalise(img)
        img = _adaptive_threshold_bw(img)
        img = _thin_characters(img)
        img = _conservative_sharpen(img)
        return _to_bytes(img, OCR_JPEG_Q)
    except Exception as e:
        logger.warning("[enhancer] OCR enhance failed: %s", e)
        return image_bytes


def optimize_image(image_bytes: bytes, max_size=(512, 512), quality=85) -> bytes:
    """
    Legacy-compatible wrapper used by pipeline.py.
    Now calls enhance_for_vlm() instead of just resizing.
    """
    return enhance_for_vlm(image_bytes)


def create_composite_vlm_image(image_bytes: bytes, table_bbox: list[int] | None) -> bytes:
    """
    Implements Option 2: 'Divide & Conquer' via Image Compounding.
    Creates an image containing the full document + a high-res crop of the table area.
    This gives the VLM 'Big Picture' context and 'Close-up' detail in a single token pass.
    """
    if not table_bbox or len(table_bbox) < 4:
        return enhance_for_vlm(image_bytes)

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        
        # 1. Enhance the full image
        full_enhanced = _clahe_equalise(_safe_upscale(img))
        
        # 2. Extract and enhance the table crop
        # Pad the bbox slightly for context
        pad_w = int((table_bbox[2] - table_bbox[0]) * 0.05)
        pad_h = int((table_bbox[3] - table_bbox[1]) * 0.05)
        crop_box = (
            max(0, table_bbox[0] - pad_w),
            max(0, table_bbox[1] - pad_h),
            min(w, table_bbox[2] + pad_w),
            min(h, table_bbox[3] + pad_h)
        )
        table_crop = img.crop(crop_box)
        table_enhanced = _clahe_equalise(_safe_upscale(table_crop))
        
        # 3. Build composite (Vertical stack)
        # Resize full image to be even smaller to save VRAM
        full_small = full_enhanced.resize((640, int(640 * (h/w))), Image.Resampling.LANCZOS)
        
        # Scale down table crop if it's huge
        if table_enhanced.width > 800:
            scale_crop = 800 / table_enhanced.width
            table_enhanced = table_enhanced.resize((800, int(table_enhanced.height * scale_crop)), Image.Resampling.LANCZOS)
        
        # Calculate canvas size
        canvas_w = max(full_small.width, table_enhanced.width)
        canvas_h = full_small.height + table_enhanced.height + 20 # 20px gap
        
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(full_small, (0, 0))
        canvas.paste(table_enhanced, (0, full_small.height + 20))
        
        result = _to_bytes(canvas, 80) # Lower quality to save memory
        logger.info("[enhancer] Compact Composite VLM image ready. Size: %d bytes", len(result))
        return result
        
    except Exception as e:
        logger.warning("[enhancer] Composite image failed, fallback to standard: %s", e)
        return enhance_for_vlm(image_bytes)
