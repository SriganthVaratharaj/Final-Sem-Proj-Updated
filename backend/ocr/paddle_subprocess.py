import os
import sys
import json
import numpy as np
import cv2
import logging
import base64
from pathlib import Path

# Force CPU for subprocess to avoid any CUDA conflict
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['FLAGS_selected_gpus'] = ''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Silence logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

def run_isolated_ocr(image_path, lang):
    try:
        from paddleocr import PaddleOCR
        # Initialize
        ocr = PaddleOCR(lang=lang, use_gpu=False, use_angle_cls=True, show_log=False)
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Image load failed"}
            
        # Run OCR
        result = ocr.ocr(img, cls=False)
        
        texts = []
        boxes = []
        confidences = []
        
        if result and result[0]:
            for line in result[0]:
                boxes.append(line[0])
                texts.append(line[1][0])
                confidences.append(float(line[1][1]))
        
        return {
            "texts": texts,
            "boxes": boxes,
            "confidences": confidences
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python paddle_subprocess.py <image_path> <lang>"}))
        sys.exit(1)
        
    img_path = sys.argv[1]
    language = sys.argv[2]
    
    output = run_isolated_ocr(img_path, language)
    print(json.dumps(output))
