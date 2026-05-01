from __future__ import annotations
"""
## SINGLE_PASS_PIPELINE_V3
backend/pipeline.py

OCR Strategy: PaddleOCR + EasyOCR run in PARALLEL.
Both results are merged into a single OCR hint fed to VLM.
No fallback chain — both engines always run independently.
"""
import asyncio, logging, io
from pathlib import Path
import cv2
import numpy as np

from backend.utils.export import export_to_excel, save_layout_json
from backend.utils.image_optimizer import optimize_image
from backend.utils.image_enhancer import enhance_for_ocr, enhance_for_vlm
from backend.utils.layout_template import map_to_standard_template
from backend.utils.layout_reconstructor import reconstruct_spatial_text, save_digital_twin
from backend.utils.report_generator import generate_structured_report, save_structured_report
from backend.vlm.vlm_model import vlm_extract_all
from backend.vlm.gguf_engine import release_vlm_memory
from backend.ocr.engine import run_ocr
from backend.ocr.easyocr_engine import run_easyocr_all_indic, is_easyocr_available, release_gpu_memory
from backend.layout.layoutlm_service import analyze_document_layout, release_layoutlm_memory
from backend.utils.table_detector import detect_tables_opencv, draw_table_boxes
from backend.config import OUTPUT_DIR
from PIL import Image

logger = logging.getLogger(__name__)


def _merge_ocr_results(paddle_res: tuple, easy_res: tuple) -> tuple[list[str], list[list[list[float]]]]:
    """
    Merge PaddleOCR and EasyOCR results (texts + boxes).
    Deduplicates based on spatial overlap and text similarity.
    """
    p_texts, p_boxes = paddle_res[0], paddle_res[1]
    e_texts, e_boxes = easy_res[0] if easy_res else [], easy_res[1] if easy_res else []

    merged_texts = list(p_texts)
    merged_boxes = list(p_boxes)
    
    p_lower = {t.lower().strip() for t in p_texts if t.strip()}

    for text, box in zip(e_texts, e_boxes):
        clean = text.strip()
        if not clean: continue
        # Basic dedup: if text is already found in Paddle, skip
        if clean.lower() not in p_lower and not any(clean.lower() in p for p in p_lower):
            merged_texts.append(clean)
            merged_boxes.append(box)

    return merged_texts, merged_boxes


async def run_pipeline(image_path, image_bytes, original_filename, on_stage=None, user_email=None, session_id=None, correction_rules=""):
    stem = Path(original_filename).stem
    user_output_dir = OUTPUT_DIR / (user_email or "guest")
    user_output_dir.mkdir(parents=True, exist_ok=True)

    async def _emit(s):
        if on_stage: await on_stage(s)

    try:
        # ── NEW: Dual-Document Splitting ──────────────────────────────────────────
        from backend.utils.image_enhancer import split_dual_invoice
        image_segments = split_dual_invoice(image_bytes)
        
        combined_vlm_parts = []
        combined_twin_parts = []
        final_dominant_lang = "latin"
        
        for seg_idx, seg_bytes in enumerate(image_segments):
            logger.info("[pipeline] Processing Image Segment %d/%d", seg_idx+1, len(image_segments))
            await _emit(f"Processing Bill {seg_idx+1}...")

            # ── Dual-path image enhancement ──────────────────────────────────────
            ocr_bytes, vlm_bytes = await asyncio.gather(
                asyncio.to_thread(enhance_for_ocr, seg_bytes),
                asyncio.to_thread(enhance_for_vlm, seg_bytes)
            )
            ocr_nparr = np.frombuffer(ocr_bytes, np.uint8)
            img_np_enhanced = cv2.imdecode(ocr_nparr, cv2.IMREAD_COLOR)

            # ── Step 1: PaddleOCR ────────────────────────────────────────────────
            p_res = ([], [], [], {})
            if img_np_enhanced is not None:
                texts, boxes, confs, meta = await asyncio.to_thread(
                    run_ocr, img_np_enhanced, None, ocr_bytes, original_filename
                )
                p_res = (texts, boxes, confs, meta)
            
            dominant_lang = p_res[3].get("dominant_language", "latin")
            final_dominant_lang = dominant_lang

            # ── Step 2: EasyOCR ──────────────────────────────────────────────────
            e_res = ([], [], [], {})
            if is_easyocr_available():
                texts, boxes, confs, meta = await asyncio.to_thread(
                    run_easyocr_all_indic, ocr_bytes, dominant_lang
                )
                e_res = (texts, boxes, confs, meta)

            await asyncio.to_thread(release_gpu_memory)
            merged_texts, merged_boxes = _merge_ocr_results(p_res, e_res)
            # Trim OCR hint to prevent context overflow on 4GB VRAM
            ocr_hint_full = "\n".join(merged_texts)
            ocr_hint = ocr_hint_full[:600] if len(ocr_hint_full) > 600 else ocr_hint_full

            # ── Step 3: OpenCV Table Detection ───────────────────────────────────
            table_boxes = await asyncio.to_thread(detect_tables_opencv, seg_bytes)
            main_table_bbox = None
            if table_boxes:
                xmin = min(b['bounding_box'][0] for b in table_boxes)
                ymin = min(b['bounding_box'][1] for b in table_boxes)
                xmax = max(b['bounding_box'][2] for b in table_boxes)
                ymax = max(b['bounding_box'][3] for b in table_boxes)
                main_table_bbox = [xmin, ymin, xmax, ymax]
                from backend.utils.image_enhancer import create_composite_vlm_image
                vlm_bytes = await asyncio.to_thread(create_composite_vlm_image, vlm_bytes, main_table_bbox)

            # ── Step 4: VLM Extraction ───────────────────────────────────────────
            vlm_res = await asyncio.to_thread(
                vlm_extract_all, vlm_bytes, correction_rules, ocr_hint, original_filename
            )
            await asyncio.to_thread(release_vlm_memory)

            # ── Step 5: Digital Twin ─────────────────────────────────────────────
            from backend.utils.layout_reconstructor import reconstruct_spatial_text
            sh, sw = img_np_enhanced.shape[:2]  # Use enhanced dimensions!
            twin_text = reconstruct_spatial_text(merged_texts, merged_boxes, sw, sh)
            
            vlm_text = vlm_res.get("fields", {}).get("full_extraction", "")
            if not vlm_text: vlm_text = f"Extraction failed for segment {seg_idx+1}"
            
            combined_vlm_parts.append(vlm_text)
            combined_twin_parts.append(twin_text)

        # ── Result Combination ────────────────────────────────────────────────
        final_vlm = "\n\n---\n\n### SECOND BILL ###\n\n".join(combined_vlm_parts) if len(combined_vlm_parts) > 1 else combined_vlm_parts[0]
        final_twin = "\n\n---\n\n".join(combined_twin_parts)

        # We use the FIRST segment's fields for structured output
        first_res = vlm_res if image_segments else {"fields": {}, "_source": "unavailable"}
        raw_fields = first_res.get("fields", {})
        source = first_res.get("_source", "unavailable")
        
        from backend.utils.layout_template import map_to_standard_template
        template_fields = map_to_standard_template(raw_fields)

        return {
            "status": "success",
            "vlm_output": final_vlm,
            "vlm_fields": raw_fields,
            "template_fields": template_fields,
            "vlm_source": source,
            "digital_twin_content": final_twin,
            "metadata": {
                "segments_processed": len(image_segments),
                "dominant_language": final_dominant_lang
            }
        }

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return {"status": "failed", "error": str(e)}
