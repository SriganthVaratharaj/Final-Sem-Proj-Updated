# 👥 Team Work Distribution - Multimodal Document Intelligence System

Intha document-la namma project team members (4 members) oda individually divided modules, backend/frontend contribution, and implementation-ah clear-ah **Tanglish-la** explain panniruku. Review and Presentation-ku prepare panna intha breakdown romba use-aagum.

---

## 👨‍💻 1. AADHISESHAN S (732722104001)
### **Role: Frontend Developer & UI Specialist**

*   **Work Done & Modules Covered:**
    *   React.js frontend structure design build panninar in **[HomePage.jsx](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/frontend/src/pages/HomePage.jsx)** and **[ResultTabs.jsx](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/frontend/src/components/ResultTabs.jsx)**.
    *   **In-App Camera Scanner:** Mobile/Laptop camera video stream frame stream panni canvas-la draw panra module write panninar (`CaptureScreen.jsx`). Intha canvas optimization and image downscaling (cap at 1200px) mobile browser-la RAM crash prevent pannum.
    *   **Locked Touch Cropping Selection:** Scanner screen and upload zone-la user touch drag selection panni receipt shape mattum select panna cropper boundaries design and `touch-action: none` code select panninar.
    *   **Vercel Deployment:** Frontend static compilation handle panni github commit trigger vercel CD setup configure panninar.

---

## 👨‍💻 2. JAYASIMBU J (732722104018)
### **Role: Local Master Backend & Preprocessing Engineer**

*   **Work Done & Modules Covered:**
    *   FastAPI local master node setup and pipelines build panninar in **[main.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/main.py)** and **[pipeline.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/pipeline.py)**.
    *   **CLAHE Contrast Enhancer:** OpenCV image filter pipeline integrate panni low-contrast thermal bill details readability improve panna enhancement engine write panninar (`image_enhancer.py`).
    *   **Dynamic Tall Split Module:** receipt vertical height excess ah iruntha standard overlapping pixels logic create panni overlapping margin slices calculate panna algorithm build panninar to optimize VLM tokens.
    *   **Template Mapping & Exports:** VLM native schema extraction JSON format-ah standard excel fields template format columns-la auto map generate panna exports handler build panninar (`export.py` / `report_generator.py`).

---

## 👨‍💻 3. REMO V (732722104047)
### **Role: Database, Authentication & Security Developer**

*   **Work Done & Modules Covered:**
    *   MongoDB Atlas Cloud Database setup and collection models indexing write panninar in **[models.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/db/models.py)** and Motor client connection wrapper configure panninar.
    *   **JWT Token Authorization:** Auth schema setup panni backend routing controller logic write panninar in **[routes.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/auth/routes.py)**. Email sign JWT validation implement panni guest mode vs user history mode isolation split build panninar.
    *   **Bcrypt Security Integration:** passlib library issue-ah raw library `bcrypt` hashing module write panni password hash create verification mismatch fix handle panninar in `auth_repository.py`.
    *   **Email Casing Normalization:** input emails normalization `.strip().lower()` and MongoDB database regex search logic case-insensitive query flow integration test and implement check panninar.

---

## 👨‍💻 4. SRIGANTH GV (732722104056)
### **Role: System Architect, Remote GPU & Integration Lead**

*   **Work Done & Modules Covered:**
    *   **Master-Worker Distributed Topology:** system design structure design panni cloud worker node and local orchestration bridge construct check verification handle panninar.
    *   **Remote VLM Inference Node:** Kaggle dual-T4 GPUs runtime server endpoint configure check panni Qwen2.5-VL-32B model inference integration process lock (`_vlm_lock` threads) setup finalize logic check panninar.
    *   **SSE Streaming & Keep-Alive Loop:** local master backend-la stream chunk request connection `/api/stream/{job_id}` setup panni, Cloudflare tunnel server timeout limit-ah handle panna 15s keepalive sse comments write loop implement check validation flow setup handle panninar.
    *   **Dynamic Endpoint Settings:** database config registry collection dynamic endpoint lookup integrate panni vlm endpoint sync setup implement check validation flow set.
    *   **Dummy Image Translation Fallback:** text-only translation API `/api/translate` calls remote worker crash error-ah bypass panna 1x1 transparent dummy base64 GIF integration flow code fix run check validation update handle code merge write check.
