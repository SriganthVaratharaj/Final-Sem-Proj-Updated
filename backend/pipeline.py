from __future__ import annotations
"""
## SINGLE_PASS_PIPELINE_V2
backend/pipeline.py
"""
import asyncio, logging, io
from pathlib import Path
from backend.utils.export import export_to_excel, save_layout_json
from backend.utils.image_optimizer import optimize_image
from backend.utils.report_generator import generate_structured_report, save_structured_report
from backend.vlm.vlm_model import vlm_extract_all

logger = logging.getLogger(__name__)

async def run_pipeline(image_path, image_bytes, original_filename, on_stage=None, user_email=None, session_id=None, correction_rules=""):
    stem = Path(original_filename).stem
    user_output_dir = (Path("db/outputs") / (user_email or "guest"))
    user_output_dir.mkdir(parents=True, exist_ok=True)

    async def _emit(s): 
        if on_stage: await on_stage(s)

    try:
        await _emit("ocr"); await _emit("layout"); await _emit("vlm")
        opt_bytes = await asyncio.to_thread(optimize_image, image_bytes)
        vlm_res = await asyncio.to_thread(vlm_extract_all, opt_bytes, correction_rules)

        ocr_lines = vlm_res.get("ocr_text", "").split("\n")
        result = {
            "status": "success",
            "ocr_texts": [l for l in ocr_lines if l.strip()],
            "vlm_fields": vlm_res.get("fields", {}),
            "vlm_source": vlm_res.get("_source", "failed"),
            "document_type": "invoice" if vlm_res.get("is_invoice") else "general",
            "layout_regions": [{"section": s.get("section"), "content_lines": s.get("content_lines")} for s in vlm_res.get("layout_sections", [])],
        }

        # Export logic simplified
        merged = vlm_res.get("fields", {})
        full_payload = {"file_name": original_filename, "extracted_data": merged, "document_type": result["document_type"]}
        excel_path = await asyncio.to_thread(export_to_excel, merged, user_output_dir, stem)
        json_path = await asyncio.to_thread(save_layout_json, full_payload, user_output_dir, stem)
        
        result.update({
            "excel_file_url": f"/outputs/{user_email or 'guest'}/{excel_path.name}",
            "json_output_url": f"/outputs/{user_email or 'guest'}/{json_path.name}"
        })
        return result
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"status": "failed", "error": str(e)}
