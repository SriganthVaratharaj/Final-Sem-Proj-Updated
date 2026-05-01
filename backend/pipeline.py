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
from backend.utils.report_generator import generate_structured_report, save_structured_report
from backend.vlm.vlm_model import vlm_extract_all
from backend.vlm.gguf_engine import release_vlm_memory
from backend.ocr.engine import run_ocr
from backend.ocr.easyocr_engine import run_easyocr_all_indic, is_easyocr_available, release_gpu_memory

logger = logging.getLogger(__name__)


def _merge_ocr_results(paddle_texts: list[str], easy_texts: list[str]) -> list[str]:
    """
    Merge PaddleOCR and EasyOCR outputs into a single deduplicated list.
    Basic strategy:
    - Keep all PaddleOCR lines
    - Append EasyOCR lines that are NOT already covered (dedup by rough match)
    """
    merged = list(paddle_texts)
    paddle_lower = {t.lower().strip() for t in paddle_texts if t.strip()}

    for line in easy_texts:
        clean = line.strip()
        if not clean:
            continue
        # Add if not already present (rough dedup — check substring match too)
        if clean.lower() not in paddle_lower and not any(clean.lower() in p for p in paddle_lower):
            merged.append(clean)

    return merged


async def run_pipeline(image_path, image_bytes, original_filename, on_stage=None, user_email=None, session_id=None, correction_rules=""):
    stem = Path(original_filename).stem
    user_output_dir = (Path("db/outputs") / (user_email or "guest"))
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

        # ── Run PaddleOCR first, then EasyOCR SEQUENTIALLY ──────────────────
        # REASON: PaddleOCR (Paddle C++ CUDA) and EasyOCR (PyTorch CUDA) both
        # register _CudaDeviceProperties on the same process. Running them in
        # parallel causes a fatal CUDA type collision. Sequential avoids this.
        paddle_texts, easy_texts = [], []

        # Step 1: PaddleOCR (uses high-contrast B&W enhanced image)
        try:
            if img_np_enhanced is not None:
                texts, _, _, _ = await asyncio.to_thread(
                    run_ocr, img_np_enhanced, None, ocr_bytes, original_filename
                )
                paddle_texts = texts
                logger.info("[pipeline] PaddleOCR: %d lines", len(texts))
        except Exception as e:
            logger.warning("[pipeline] PaddleOCR failed: %s", e)

        # Step 2: EasyOCR (also uses the B&W enhanced image for better contrast)
        try:
            if is_easyocr_available():
                texts, _, _, _ = await asyncio.to_thread(
                    run_easyocr_all_indic, ocr_bytes
                )
                easy_texts = texts
                logger.info("[pipeline] EasyOCR: %d lines", len(texts))
            else:
                logger.debug("[pipeline] EasyOCR not available, skipping")
        except Exception as e:
            logger.warning("[pipeline] EasyOCR failed: %s", e)

        # ── PHASE 1 → 2 HANDOFF: Release GPU VRAM before VLM loads ───────────
        # EasyOCR holds ~800MB per language model in VRAM.
        # Must free this BEFORE LLaVA loads or it will OOM on 4GB GPU.
        logger.info("[pipeline] OCR done. Releasing GPU VRAM before VLM...")
        await asyncio.to_thread(release_gpu_memory)

        # ── Merge OCR results ─────────────────────────────────────────────────
        merged_texts = _merge_ocr_results(paddle_texts, easy_texts)
        ocr_hint = "\n".join(merged_texts)

        logger.info(
            "[pipeline] OCR merged: paddle=%d, easyocr=%d, total=%d lines",
            len(paddle_texts), len(easy_texts), len(merged_texts)
        )

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
                "paddle_lines": len(paddle_texts),
                "easyocr_lines": len(easy_texts),
                "merged_lines": len(merged_texts),
            },
            "vlm_fields": raw_fields,            # Raw VLM output (original)
            "template_fields": template_fields,   # Standardized positional template
            "vlm_source": vlm_res.get("_source", "failed"),
            "document_type": "invoice" if vlm_res.get("is_invoice") else "general",
            "layout_regions": [],
        }

        # Export uses template_fields for consistent column order across all invoices
        full_payload = {
            "file_name": original_filename,
            "extracted_data": template_fields,
            "raw_vlm_data": raw_fields,
            "document_type": result["document_type"]
        }
        excel_path = await asyncio.to_thread(export_to_excel, template_fields, user_output_dir, stem)
        json_path = await asyncio.to_thread(save_layout_json, full_payload, user_output_dir, stem)

        result.update({
            "excel_file_url": f"/outputs/{user_email or 'guest'}/{excel_path.name}",
            "json_output_url": f"/outputs/{user_email or 'guest'}/{json_path.name}"
        })
        return result

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"status": "failed", "error": str(e)}
