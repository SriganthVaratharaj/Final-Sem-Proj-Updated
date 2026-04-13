"""
backend/main.py
FastAPI application — unified entry point for the combined Invoice AI system.

Run:
    cd backend
    python main.py

Endpoints:
  POST /api/upload            → save files, return job_id
  GET  /api/stream/{job_id}   → SSE — real-time stage-by-stage progress
  POST /api/process           → single-call (no streaming)
  GET  /api/results           → list past results from MongoDB
  GET  /api/results/{id}      → single result from MongoDB
  DELETE /api/results/{id}    → delete a result
  GET  /api/health            → health check
  GET  /outputs/{file}        → download exported files
  GET  /                      → serve React frontend (production build)
"""
# ── `from __future__` MUST be first import (Python rule) ──────────────────────
from __future__ import annotations

# ── Path bootstrap: add project root so `backend.*` and `db.*` resolve ────────
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Updated Final Year Project/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.auth.routes import router as auth_router, get_current_user_optional

from backend.config import (
    ALLOWED_EXTENSIONS,
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    FRONTEND_DIST_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    OUTPUT_DIR,
    UPLOAD_DIR,
)
from backend.pipeline import run_pipeline
from db.connection import ping_db
from db.repository import delete_result, get_result, list_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Startup directories ────────────────────────────────────────────────────────
for _dir in (UPLOAD_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── In-memory job store: job_id → dict{files, user_email} ─────────────────────
_job_store: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load PaddleOCR models at startup to eliminate cold-start delay."""
    from backend.ocr.engine import _load_models
    logger.info("[startup] Pre-loading PaddleOCR language models...")
    await asyncio.to_thread(_load_models)
    logger.info("[startup] PaddleOCR models ready.")
    yield


app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan)
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:5174", "http://127.0.0.1:5174", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Cache-Control", "X-Accel-Buffering"],
)

# Serve exported files (JSON, Excel, TXT)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ── Validation helper ──────────────────────────────────────────────────────────
def _validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(400, "No file selected.")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Invalid file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["system"])
async def health():
    db_ok = await ping_db()
    return {"status": "ok", "db_connected": db_ok, "version": API_VERSION}


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING MODE — Step 1: Upload → get job_id
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload", tags=["processing"])
async def upload_for_streaming(
    files: List[UploadFile] = File(...),
    user_email: str | None = Depends(get_current_user_optional)
):
    """
    Save uploaded files to temp storage and return a job_id.
    Then open GET /api/stream/{job_id} for real-time SSE results.
    """
    job_id = str(uuid.uuid4())
    file_infos = []

    # Map target dir
    target_dir = (UPLOAD_DIR / user_email) if user_email else (UPLOAD_DIR / "tmp" / job_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, file in enumerate(files, start=1):
        _validate_file(file)
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(400, f"File '{file.filename}' exceeds {MAX_UPLOAD_SIZE_MB} MB limit.")

        safe_name = f"{idx:04d}_{file.filename}"
        file_path = target_dir / safe_name
        file_path.write_bytes(content)
        file_infos.append({"path": str(file_path), "bytes": content, "name": file.filename, "idx": idx})

    _job_store[job_id] = {"files": file_infos, "user_email": user_email}
    logger.info("[upload] job_id=%s, email=%s, files=%d", job_id, user_email or "guest", len(file_infos))
    return {"job_id": job_id, "file_count": len(file_infos)}


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING MODE — Step 2: SSE stream per stage
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/stream/{job_id}", tags=["processing"])
async def stream_results(job_id: str):
    """
    SSE endpoint — processes images sequentially and pushes events:
      { "event": "stage",  "stage": "ocr|layout|vlm|export", "image": N, "total": M }
      { "event": "result", "image": N, "data": { ...combined result... } }
      { "event": "done",   "total": M }
    """
    if job_id not in _job_store:
        return JSONResponse({"error": "Job not found or already consumed."}, status_code=404)

    job_data = _job_store.pop(job_id)
    file_infos = job_data["files"]
    user_email = job_data["user_email"]

    async def event_generator():
        total = len(file_infos)
        all_results = []

        try:
            for info in file_infos:
                image_number = info["idx"]

                # Stage progress callback → push SSE event
                async def emit_stage(stage_name: str, img_no=image_number):
                    event = json.dumps({"event": "stage", "stage": stage_name, "image": img_no, "total": total})
                    yield f"data: {event}\n\n"

                # We can't use a generator inside asyncio.to_thread, so we collect
                # stage events via a queue
                q: asyncio.Queue = asyncio.Queue()

                async def put_stage(stage_name: str):
                    await q.put(stage_name)

                async def run_and_collect():
                    return await run_pipeline(
                        image_path=info["path"],
                        image_bytes=info["bytes"],
                        original_filename=info["name"],
                        on_stage=put_stage,
                        user_email=user_email,
                        session_id=job_id
                    )

                pipeline_task = asyncio.create_task(run_and_collect())

                # Drain stage events while pipeline runs
                while not pipeline_task.done():
                    try:
                        stage = await asyncio.wait_for(q.get(), timeout=0.1)
                        event = json.dumps({"event": "stage", "stage": stage, "image": image_number, "total": total})
                        yield f"data: {event}\n\n"
                    except asyncio.TimeoutError:
                        pass

                result = await pipeline_task

                # Drain any remaining stage events
                while not q.empty():
                    stage = q.get_nowait()
                    event = json.dumps({"event": "stage", "stage": stage, "image": image_number, "total": total})
                    yield f"data: {event}\n\n"

                all_results.append(result)
                result_event = json.dumps({"event": "result", "image": image_number, "data": result}, ensure_ascii=False, default=str)
                yield f"data: {result_event}\n\n"

        except asyncio.CancelledError:
            logger.warning("[stream] Client disconnected — job %s", job_id)
        finally:
            # Clean up temp upload files when done (or if crashed)
            for info in file_infos:
                try:
                    Path(info["path"]).unlink(missing_ok=True)
                except OSError:
                    pass

        done_event = json.dumps({"event": "done", "total": len(all_results)})
        yield f"data: {done_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )

@app.delete("/api/cleanup/{session_id}", tags=["cleanup"])
async def cleanup_guest_session(session_id: str):
    """Deletes temporary outputs for guest sessions when they close the browser."""
    try:
        shutil.rmtree(OUTPUT_DIR / "tmp" / session_id, ignore_errors=True)
        shutil.rmtree(UPLOAD_DIR / "tmp" / session_id, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Failed to cleanup session {session_id}: {e}")
    return {"status": "cleaned"}


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY — Single-call (no streaming)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/process", tags=["processing"])
async def process_files(files: List[UploadFile] = File(...)):
    """Process one or more images synchronously and return all results at once."""
    all_results = []

    for idx, file in enumerate(files, start=1):
        _validate_file(file)
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(400, f"File '{file.filename}' exceeds {MAX_UPLOAD_SIZE_MB} MB limit.")

        safe_name = f"{idx:04d}_{file.filename}"
        file_path = TEMP_DIR / safe_name
        file_path.write_bytes(content)

        try:
            result = await run_pipeline(str(file_path), content, file.filename)
        finally:
            file_path.unlink(missing_ok=True)

        all_results.append(result)

    return JSONResponse(content={"results": all_results, "total": len(all_results)})


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY — MongoDB results
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/results", tags=["history"])
async def get_results(limit: int = 20):
    """List the most recent extraction results from MongoDB."""
    results = await list_results(limit=limit)
    return {"results": results, "total": len(results)}


@app.get("/api/results/{result_id}", tags=["history"])
async def get_single_result(result_id: str):
    """Fetch a single result by its MongoDB _id."""
    doc = await get_result(result_id)
    if not doc:
        raise HTTPException(404, "Result not found.")
    return doc


@app.delete("/api/results/{result_id}", tags=["history"])
async def remove_result(result_id: str):
    """Delete a result from MongoDB."""
    ok = await delete_result(result_id)
    if not ok:
        raise HTTPException(404, "Result not found.")
    return {"deleted": result_id}


# ═══════════════════════════════════════════════════════════════════════════════
# FRONTEND — Serve React build (production)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(path: str = ""):
    """Serve the React frontend for all non-API routes."""
    index = FRONTEND_DIST_DIR / "index.html"
    if FRONTEND_DIST_DIR.exists() and index.exists():
        # Serve static assets
        asset = FRONTEND_DIST_DIR / path
        if path and asset.exists() and asset.is_file():
            return FileResponse(asset)
        return FileResponse(index)
    return JSONResponse({"message": "Frontend not built. Run: cd frontend && npm run build"}, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(_PROJECT_ROOT)],
    )
