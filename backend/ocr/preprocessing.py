"""
backend/ocr/preprocessing.py
Image preprocessing: resize, optional Telugu-enhanced variant for better OCR accuracy.
Source: ocr-invoice-system/src/preprocessing.py
"""
import cv2


def _resize_for_ocr(image, target_width=1200):
    height, width = image.shape[:2]
    if width < target_width:
        scale = target_width / width
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif width > 1600:
        scale = 1600 / width
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def _telugu_enhanced_variant(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    enhanced = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    return enhanced


def _script_enhanced_variant(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrast, None, 7, 7, 15)
    sharpen_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    sharpened = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, sharpen_kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def preprocess_image(image_path):
    """
    Image preprocessing: resize, grayscale, noise removal, thresholding.
    Returns script-aware variants to improve Indic OCR consistency.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}. Ensure it exists.")
    image = _resize_for_ocr(image)
    script_enhanced = _script_enhanced_variant(image)
    return {
        "default": image,
        "devanagari": image,
        "ta": script_enhanced,
        "te": _telugu_enhanced_variant(image),
        "ka": script_enhanced,
        "arabic": image,
    }
