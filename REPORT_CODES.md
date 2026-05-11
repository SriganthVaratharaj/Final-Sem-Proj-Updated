# Project Report: Core Source Codes

This document contains the most important code snippets for the project report. These files represent the core logic of the Multimodal Invoice Extraction System.

---

## 1. Backend Entry Point (`backend/main.py`)
This file defines the FastAPI server, handles file uploads, and manages the SSE (Server-Sent Events) stream for real-time extraction results.

```python
@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...), user_email: str = Depends(get_current_user_optional)):
    job_id = str(uuid.uuid4())
    # ... logic to save files ...
    return {"job_id": job_id}

@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    # Streaming extraction results to frontend
    async def gen():
        for info in job["files"]:
            res = await run_pipeline(info["path"], info["bytes"], info["name"], user_email=job["user_email"], session_id=job_id)
            yield f"data: {json.dumps({'event': 'result', 'data': res}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

---

## 2. Core Pipeline Orchestration (`backend/pipeline.py`)
This is the "Brain" of the system. It handles image preprocessing (splitting dual invoices) and calls the VLM extraction.

```python
async def run_pipeline(image_path, image_bytes, original_filename, ...):
    # 1. Split images if they contain multiple invoices
    image_segments = split_dual_invoice(image_bytes)
    
    for seg_idx, seg_bytes in enumerate(image_segments):
        # 2. Enhance image for VLM
        vlm_bytes = await asyncio.to_thread(enhance_for_vlm, seg_bytes)

        # 3. VLM Extraction (Remote Kaggle or Local)
        vlm_res = await asyncio.to_thread(
            vlm_extract_all, vlm_bytes, correction_rules, "", original_filename
        )
        # ... process results ...
```

---

## 3. VLM Model Logic & Prompting (`backend/vlm/vlm_model.py`)
Contains the `MASTER_PROMPT_TEMPLATE` which instructs the AI how to extract structured data from images.

```python
MASTER_PROMPT_TEMPLATE = """
You are an End-to-End Layout-Aware Visual OCR AI. 
Step 1: Identify ALL documents in the image.
Step 2: Identify the primary native language(s).
...
Step 4 (Format): Return output STRICTLY as a single JSON object.
"""

def vlm_extract_all(image_bytes: bytes, ...):
    # Detect language and select appropriate prompt
    detected_lang = _quick_language_scan(image_bytes)
    # ... call GGUF engine ...
```

---

## 4. GGUF Engine & Remote Routing (`backend/vlm/gguf_engine.py`)
Handles the low-level communication with the local LLM (llama-server) or routes the request to a remote Kaggle GPU worker if configured.

```python
def query_local_llava(image_bytes: bytes, prompt: str, ...):
    kaggle_url = _get_dynamic_kaggle_url()
    if kaggle_url:
        # Route to Remote Kaggle Worker
        return poll_kaggle_result(kaggle_url, image_bytes, prompt)
    
    # Local GGUF Inference via subprocessed llama-server
    client = _load_gguf_model(model_type=model_type)
    return client.create_chat_completion(messages=messages, ...)
```

---

## 5. Frontend Real-time Hook (`frontend/src/hooks/useSSEStream.js`)
Manages the frontend state during the extraction process, listening to the backend stream and updating the UI.

```javascript
const process = useCallback(async (files) => {
  // 1. Upload files
  const { job_id } = await uploadFiles(files, token)
  
  // 2. Connect to Server-Sent Events stream
  const es = new EventSource(getStreamUrl(job_id))
  
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data)
    if (msg.event === 'result') {
      setResults(prev => [...prev, msg.data])
    } else if (msg.event === 'done') {
      setDone(true)
      es.close()
    }
  }
}, [])
```
