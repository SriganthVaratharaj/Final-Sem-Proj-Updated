import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_tables_opencv(image_bytes: bytes) -> list[dict]:
    """
    Detects table-like grid structures and boxes in an invoice using OpenCV contours.
    Returns a list of detected rectangular regions that likely correspond to tables or boxed items.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to binarize the image
        thresh = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
        )
        
        # Scale for morphological operations based on image size
        scale = max(img.shape[0], img.shape[1]) // 150
        
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale * 2, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale * 2))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines to form a grid
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        table_mask = cv2.erode(cv2.bitwise_not(table_mask), np.ones((2, 2)))
        thresh_table = cv2.bitwise_not(table_mask)
        
        # Find contours of the grid intersections/boxes
        contours, _ = cv2.findContours(thresh_table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        table_regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter out lines and very small/large boxes
            if w > 50 and h > 20 and w < img.shape[1]*0.95 and h < img.shape[0]*0.95:
                # Calculate bounding box area ratio to filter non-rectangular shapes
                area = cv2.contourArea(c)
                if area / (w * h) > 0.6:  # Reasonably rectangular
                    table_regions.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "bounding_box": [x, y, x + w, y + h]
                    })
                    
        # Group overlapping or adjacent boxes into larger table regions
        # (Simplified: just return the raw boxes for LayoutLMv3 or VLM to use)
        logger.info("[opencv] Detected %d table/box cells", len(table_regions))
        return table_regions
        
    except Exception as e:
        logger.error("[opencv] Table detection failed: %s", e)
        return []

def draw_table_boxes(image_bytes: bytes, boxes: list[dict]) -> bytes:
    """Draws the detected boxes onto the image for debugging or VLM processing."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        for b in boxes:
            x, y, w, h = b['x'], b['y'], b['w'], b['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
        is_success, buffer = cv2.imencode(".jpg", img)
        if is_success:
            return buffer.tobytes()
        return image_bytes
    except Exception as e:
        logger.error("[opencv] Draw boxes failed: %s", e)
        return image_bytes
