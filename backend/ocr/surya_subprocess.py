import os
import sys
import json
import logging
from PIL import Image

# Disable Tensorflow to avoid protobuf version conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TRANSFORMERS_NO_TENSORFLOW"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_isolated_ocr(image_path):
    try:
        from surya.common.surya.schema import TaskNames
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        
        image = Image.open(image_path).convert("RGB")
        
        # Load heavy transformer models directly on GPU
        foundation_predictor = FoundationPredictor()
        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor(foundation_predictor)
        
        predictions = rec_predictor(
            [image],
            task_names=[TaskNames.ocr_with_boxes],
            det_predictor=det_predictor,
        )
        
        texts = []
        boxes = []
        confidences = []
        
        # Surya prediction format conversion to match PaddleOCR format
        for text_line in predictions[0].text_lines:
            texts.append(text_line.text)
            boxes.append(text_line.polygon)
            confidences.append(float(getattr(text_line, 'confidence', 1.0)))
            
        # Free GPU Memory aggressively before returning (OS will also free on exit)
        import torch
        import gc
        del foundation_predictor
        del det_predictor
        del rec_predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            "texts": texts,
            "boxes": boxes,
            "confidences": confidences
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python surya_subprocess.py <image_path>"}))
        sys.exit(1)
        
    img_path = sys.argv[1]
    
    output = run_isolated_ocr(img_path)
    print(json.dumps(output))
