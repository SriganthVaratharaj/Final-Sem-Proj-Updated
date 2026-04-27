from __future__ import annotations
import sys, os, asyncio, json, logging, shutil, uuid
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))

from backend.auth.routes import router as auth_router, get_current_user_optional
from backend.config import *
from backend.pipeline import run_pipeline
from db.connection import ping_db
from db.repository import delete_result, get_result, list_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

_job_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Safe Mode Startup
    logger.info("[startup] Backend ready.")
    yield

app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan)
app.include_router(auth_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/tmp", StaticFiles(directory=GUEST_DIR), name="tmp")

@app.get("/api/health")
async def health(): return {"status": "ok", "db": await ping_db()}

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...), user_email: str = Depends(get_current_user_optional)):
    job_id = str(uuid.uuid4())
    target_dir = (UPLOAD_DIR / user_email) if user_email else (UPLOAD_DIR / "tmp" / job_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_infos = []
    for idx, file in enumerate(files, start=1):
        content = await file.read()
        file_path = target_dir / f"{idx:04d}_{file.filename}"
        file_path.write_bytes(content)
        file_infos.append({"path": str(file_path), "bytes": content, "name": file.filename, "idx": idx})
    _job_store[job_id] = {"files": file_infos, "user_email": user_email}
    return {"job_id": job_id}

@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in _job_store: raise HTTPException(404)
    job = _job_store.pop(job_id)
    async def gen():
        for info in job["files"]:
            res = await run_pipeline(info["path"], info["bytes"], info["name"], user_email=job["user_email"], session_id=job_id)
            yield f"data: {json.dumps({'event': 'result', 'data': res}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/api/filter")
async def filter_api(payload: dict):
    from backend.vlm.gguf_engine import query_local_llava
    from backend.vlm.vlm_model import _clean_output
    prompt, data = payload.get("prompt"), payload.get("data")
    res = await asyncio.to_thread(query_local_llava, b"", f"Filter this JSON based on '{prompt}': {json.dumps(data)}", INTERNAL_MODEL_API_KEY)
    return _clean_output(res) or data

@app.get("/api/results")
async def history(): return {"results": await list_results()}

@app.get("/")
@app.get("/{path:path}")
async def serve(path: str = ""):
    index = FRONTEND_DIST_DIR / "index.html"
    if path and (FRONTEND_DIST_DIR / path).exists(): return FileResponse(FRONTEND_DIST_DIR / path)
    return FileResponse(index) if index.exists() else {"error": "Frontend missing"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
