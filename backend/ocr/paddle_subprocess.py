import os
import sys
import json
import numpy as np
import cv2
import logging
import base64
from pathlib import Path

# Environment cleanup for Windows/CPU stability
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TRANSFORMERS_NO_TENSORFLOW"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    del os.environ['CUDA_VISIBLE_DEVICES']


# Silence logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

def run_isolated_ocr(image_path, lang):
    try:
        from paddleocr import PaddleOCR
        # Initialize
        ocr = PaddleOCR(lang=lang, use_gpu=False, use_angle_cls=True, show_log=False, enable_mkldnn=True)
        
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
