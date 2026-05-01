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
        await _emit("ocr")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ── Dual-path image enhancement ──────────────────────────────────────
        # OCR → high-contrast B&W (adaptive threshold + CLAHE + denoise)
        # VLM → color-preserved enhanced (CLAHE + sharpen, layout colors intact)
        # Both run async so they don't add serial latency
        ocr_bytes, vlm_bytes = await asyncio.gather(
            asyncio.to_thread(enhance_for_ocr, image_bytes),
            asyncio.to_thread(enhance_for_vlm, image_bytes)
        )
        # Re-decode the OCR-enhanced image for PaddleOCR
        ocr_nparr = np.frombuffer(ocr_bytes, np.uint8)
        img_np_enhanced = cv2.imdecode(ocr_nparr, cv2.IMREAD_COLOR)

        # ── Step 1: PaddleOCR (uses high-contrast B&W enhanced image) ─────────
        p_res = ([], [], [], {})
        try:
            if img_np_enhanced is not None:
                texts, boxes, confs, meta = await asyncio.to_thread(
                    run_ocr, img_np_enhanced, None, ocr_bytes, original_filename
                )
                p_res = (texts, boxes, confs, meta)
                logger.info("[pipeline] PaddleOCR: %d lines", len(texts))
        except Exception as e:
            logger.warning("[pipeline] PaddleOCR failed: %s", e)

        # ── Step 2: EasyOCR (Sequential handoff) ─────────────────────────────
        e_res = ([], [], [], {})
        try:
            if is_easyocr_available():
                texts, boxes, confs, meta = await asyncio.to_thread(
                    run_easyocr_all_indic, ocr_bytes
                )
                e_res = (texts, boxes, confs, meta)
                logger.info("[pipeline] EasyOCR: %d lines", len(texts))
            else:
                logger.debug("[pipeline] EasyOCR not available, skipping")
        except Exception as e:
            logger.warning("[pipeline] EasyOCR failed: %s", e)

        # ── PHASE 1 → 2 HANDOFF: Release GPU VRAM before VLM/Layout loads ───
        logger.info("[pipeline] OCR done. Releasing GPU VRAM before Layout/VLM...")
        await asyncio.to_thread(release_gpu_memory)

        # ── Merge OCR results ─────────────────────────────────────────────────
        merged_texts, merged_boxes = _merge_ocr_results(p_res, e_res)
        ocr_hint = "\n".join(merged_texts)

        logger.info(
            "[pipeline] OCR merged: paddle=%d, easyocr=%d, total=%d lines",
            len(p_res[0]), len(e_res[0]), len(merged_texts)
        )
        
        # ── OpenCV Table Detection (Enhances VLM Image) ──────────────────────
        await _emit("layout")
        table_boxes = await asyncio.to_thread(detect_tables_opencv, image_bytes)
        
        # Calculate main table bounding box for Option 2 (Focus Crop)
        main_table_bbox = None
        if table_boxes:
            xmin = min(b['bounding_box'][0] for b in table_boxes)
            ymin = min(b['bounding_box'][1] for b in table_boxes)
            xmax = max(b['bounding_box'][2] for b in table_boxes)
            ymax = max(b['bounding_box'][3] for b in table_boxes)
            main_table_bbox = [xmin, ymin, xmax, ymax]
            
            # Draw boxes on the VLM image so the VLM can "see" the table boundaries explicitly!
            vlm_bytes = await asyncio.to_thread(draw_table_boxes, vlm_bytes, table_boxes)
            logger.info("[pipeline] Drew %d OpenCV table boxes on VLM input image.", len(table_boxes))

        # Apply Option 2: Create composite image (Full + Zoomed Table)
        if main_table_bbox:
            from backend.utils.image_enhancer import create_composite_vlm_image
            vlm_bytes = await asyncio.to_thread(create_composite_vlm_image, vlm_bytes, main_table_bbox)
            logger.info("[pipeline] Option 2 Applied: Composite 'Full + Zoomed Table' image sent to VLM.")

        # ── LayoutLMv3 Spatial Analysis ─────────────────────────────────────
        # Convert boxes to LayoutLM format (0-1000 scale)
        img_h, img_w = img_np.shape[:2]
        layout_words = []
        layout_boxes = []
        entries = []
        for txt, box in zip(merged_texts, merged_boxes):
            if len(box) >= 4:
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                # Normalize to 1000x1000
                nx0 = int(1000 * (xmin / img_w))
                ny0 = int(1000 * (ymin / img_h))
                nx1 = int(1000 * (xmax / img_w))
                ny1 = int(1000 * (ymax / img_h))
                
                layout_words.append(txt)
                layout_boxes.append([nx0, ny0, nx1, ny1])
                entries.append({
                    "text": txt,
                    "bounding_box": [xmin, ymin, xmax, ymax]
                })
        
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            layout_data = await asyncio.to_thread(analyze_document_layout, pil_img, layout_words, layout_boxes, entries, ocr_hint)
            logger.info("[pipeline] LayoutLMv3 Analysis completed.")
            # Release LayoutLM VRAM before loading VLM
            await asyncio.to_thread(release_layoutlm_memory)
        except Exception as e:
            logger.warning("[pipeline] LayoutLMv3 Analysis failed: %s", e)
            layout_data = {"layout_regions": []}
            await asyncio.to_thread(release_layoutlm_memory)

        await _emit("vlm")

        # ── PHASE 2: VLM on GPU (VRAM now free from OCR models) ──────────────
        # VLM gets color-preserved enhanced image (NOT B&W — colors help extract totals/headers)
        vlm_res = await asyncio.to_thread(
            vlm_extract_all, vlm_bytes, correction_rules, ocr_hint, original_filename
        )

        # ── PHASE 2 → DONE: Release VLM from VRAM ────────────────────────────
        logger.info("[pipeline] VLM done. Releasing GPU VRAM for next request...")
        await asyncio.to_thread(release_vlm_memory)

        await _emit("format")

        raw_fields = vlm_res.get("fields", {})

        # ── SPATIAL RECONSTRUCTION (Digital Twin) ────────────────────────────
        # User request: VLM creates the final Markdown layout representing tables/boxes natively
        digital_twin_content = raw_fields.get("full_extraction", "")
        if not digital_twin_content:
            # Fallback if VLM completely failed
            img_h, img_w = img_np.shape[:2]
            digital_twin_content = reconstruct_spatial_text(merged_texts, merged_boxes, img_w, img_h)

        # ── LAYOUT TEMPLATE MAPPING ─────────────────────────────────────────
        # Maps ANY raw VLM output → standard positional template.
        # Same field names, same order, every invoice — regardless of language/quality.
        # Handwritten/unclear images benefit most: template fills gaps with “—”
        # instead of silently dropping fields.
        template_fields = map_to_standard_template(raw_fields)
        logger.info("[pipeline] Layout template applied. Fields mapped: %d", 
                    sum(1 for v in template_fields.values() if v and v != "—"))

        result = {
            "status": "success",
            "ocr_texts": merged_texts,
            "ocr_sources": {
                "paddle_lines": len(p_res[0]),
                "easyocr_lines": len(e_res[0]),
                "merged_lines": len(merged_texts),
            },
            "vlm_fields": raw_fields,            # Raw VLM output (original)
            "template_fields": template_fields,   # Standardized positional template
            "vlm_source": vlm_res.get("_source", "failed"),
            "document_type": layout_data.get("document_type", "invoice" if vlm_res.get("is_invoice") else "general"),
            "layout_regions": layout_data.get("layout_regions", []),
            "layoutlmv3_status": layout_data.get("layoutlmv3_status", {}),
            "opencv_tables_detected": len(table_boxes),
        }

        # ── EXPORTS ─────────────────────────────────────────────────────────
        # Excel uses template_fields for consistent column order
        full_payload = {
            "file_name": original_filename,
            "extracted_data": template_fields,
            "raw_vlm_data": raw_fields,
            "document_type": result["document_type"]
        }
        excel_path = await asyncio.to_thread(export_to_excel, template_fields, user_output_dir, stem)
        json_path = await asyncio.to_thread(save_layout_json, full_payload, user_output_dir, stem)

        # DIGITAL TWIN EXPORTS (.txt and .docx)
        dt_txt_path = await asyncio.to_thread(save_digital_twin, digital_twin_content, user_output_dir, stem, "txt")
        dt_docx_path = await asyncio.to_thread(save_digital_twin, digital_twin_content, user_output_dir, stem, "docx")

        result.update({
            "excel_file_url": f"/outputs/{user_email or 'guest'}/{Path(excel_path).name}",
            "json_output_url": f"/outputs/{user_email or 'guest'}/{Path(json_path).name}",
            "digital_twin_txt_url": f"/outputs/{user_email or 'guest'}/{Path(dt_txt_path).name}",
            "digital_twin_docx_url": f"/outputs/{user_email or 'guest'}/{Path(dt_docx_path).name}",
            "digital_twin_content": digital_twin_content
        })
        return result

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"status": "failed", "error": str(e)}
