# Project Report: Comprehensive Core Source Codes

This document provides a detailed collection of the most critical code logic for the Multimodal Invoice Extraction System.

---

## 1. Backend API & Stream Management (`backend/main.py`)
This file handles the FastAPI server initialization and the asynchronous streaming of extraction results to the client.

```python
@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...), user_email: str = Depends(get_current_user_optional)):
    job_id = str(uuid.uuid4())
    target_dir = (UPLOAD_DIR / user_email) if user_email else (UPLOAD_DIR / "tmp" / job_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    # ... logic to save files ...
    return {"job_id": job_id}

@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    async def gen():
        for info in job["files"]:
            # Run the full extraction pipeline for each file
            res = await run_pipeline(info["path"], info["bytes"], info["name"], user_email=job["user_email"], session_id=job_id)
            yield f"data: {json.dumps({'event': 'result', 'data': res}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

---

## 2. Image Preprocessing & Enhancement (`backend/utils/image_enhancer.py`)
Critical logic for handling dual-invoice scans and enhancing images for the VLM model.

```python
def split_for_extraction(image_bytes: bytes) -> list[bytes]:
    """Divide & Conquer: Splits large/tall images into meaningful segments."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if w > h * 1.8: # Wide Side-by-Side
        left = img.crop((0, 0, w // 2, h))
        right = img.crop((w // 2, 0, w, h))
        return [enhance_for_vlm_pil(left), enhance_for_vlm_pil(right)]
    elif h > w * 1.5: # Tall Overlapping Split
        top = img.crop((0, 0, w, int(h * 0.55)))
        bottom = img.crop((0, int(h * 0.45), w, h))
        return [enhance_for_vlm_pil(top), enhance_for_vlm_pil(bottom)]
    return [enhance_for_vlm(image_bytes)]

def _invert_colored_bands(img: Image.Image) -> Image.Image:
    """Inverts dark colored headers (white text on dark bg) to improve OCR accuracy."""
    # ... HSV analysis logic ...
    # Only inverts colored regions, leaves white background alone.
    return Image.fromarray(result)
```

---

## 3. VLM Model Interaction & Prompting (`backend/vlm/vlm_model.py`)
Defines the instructions for the AI model and handles the JSON parsing/cleaning of the raw output.

```python
MASTER_PROMPT_TEMPLATE = """
You are an End-to-End Layout-Aware Visual OCR AI.
Step 1: Identify ALL documents in the image.
Step 2: Identify primary native language(s).
Step 3: Perform structured extraction (JSON).
Step 3.5: Use Markdown Tables for grid data.
"""

# Language-specific context rules for multi-script support
_LANG_SPECIFIC_RULES = {
    "hindi":     "The text uses Devanagari script. Read matras (vowel marks) carefully.",
    "tamil":     "The text uses Tamil script. Letters have circular curves.",
    "telugu":    "The text uses Telugu script. Characters have rounded shapes.",
    "urdu":      "The text uses Urdu/Nastaliq script (right-to-left).",
}

def _clean_output(text: str) -> dict | None:
    # Strip markdown fences and find the JSON block
    text = text.replace("```json", "").replace("```", "").strip()
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    return {"fields": json.loads(text)}
```

---

## 4. Layout Template Mapping (`backend/utils/layout_template.py`)
Standardizes inconsistent AI outputs into a fixed schema, ensuring reliability for downstream systems.

```python
INVOICE_TEMPLATE = [
    ("vendor_name",    "top_left",     ["vendor", "shop", "seller", "company"]),
    ("invoice_date",   "top_right",    ["date", "bill_date", "dated"]),
    ("items",          "center",       ["items", "particulars", "description"]),
    ("total_amount",   "bottom_right", ["total", "grand_total", "payable"]),
]

def map_to_standard_template(raw_extraction: dict):
    # Uses fuzzy alias matching to map raw fields to the canonical template
    result = {}
    for field_name, zone, aliases in INVOICE_TEMPLATE:
        val = _find_value(raw_extraction, aliases)
        result[field_name] = val or ""
    return result
```

---

## 5. Remote Kaggle Routing (`backend/vlm/gguf_engine.py`)
Routes requests to a high-performance Kaggle GPU worker when local VRAM is insufficient.

```python
def query_local_llava(image_bytes, prompt, ...):
    kaggle_url = _get_dynamic_kaggle_url()
    if kaggle_url:
        # Route to Kaggle API
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        job_id = start_remote_job(kaggle_url, b64, prompt)
        # Poll for completion
        while True:
            res = check_status(kaggle_url, job_id)
            if res["status"] == "completed": return res["data"]
            time.sleep(5)
    
    # Local Inference fallback
    client = _load_gguf_model()
    return client.create_chat_completion(...)
```

---

## 6. Database Persistence (`db/repository.py`)
Handles saving extraction results to MongoDB for historical tracking and analysis.

```python
async def save_result(result: dict[str, Any]) -> str | None:
    """Persist a pipeline result dict to MongoDB."""
    db = get_async_db()
    if db is None: return None

    doc = make_invoice_document(
        file_name=result.get("image_name"),
        vlm_fields=result.get("vlm_fields"),
        vlm_source=result.get("vlm_source"),
        status="success"
    )

    try:
        inserted = await db["invoice_results"].insert_one(doc)
        return str(inserted.inserted_id)
    except Exception as exc:
        return None
```

---

## 7. Frontend Real-time Processing (`frontend/src/hooks/useSSEStream.js`)
Manages the real-time UI updates as invoices are processed one by one.

```javascript
const process = useCallback(async (files) => {
    // Start upload and get Job ID
    const { job_id } = await uploadFiles(files, token);
    
    // Connect to Server-Sent Events (SSE)
    const es = new EventSource(`/api/stream/${job_id}`);
    es.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.event === 'result') {
            setResults(prev => [...prev, msg.data]);
        } else if (msg.event === 'done') {
            setDone(true);
            es.close();
        }
    };
}, []);
```
