# 👥 Team Work Distribution - Multimodal Document Intelligence System

Intha document-la namma project team members (4 members) oda individually divided modules, detailed explanations, backend/frontend files name, and exact code line ranges reference-ah clear-ah **Tanglish-la** explain panniruku. External Review and Viva-ku prepare panna intha details romba helpful-ah irukum.

---

## 👨‍💻 1. AADHISESHAN S (732722104001)
### **Role: Frontend Developer & UI Specialist**

*   **Detailed Role Overview:**
    Frontend layout design, components, user actions, responsive structure, and page state controls full-ah control panrathu intha profile thaan. React and Tailwind CSS use panni, visually stunning layout and mobile-first responsive screens flow setup panni irukaaru.

*   **Modules Covered & Code References:**
    *   **Main Home Page Layout & SSE Setup:**
        *   **File Path:** [HomePage.jsx](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/frontend/src/pages/HomePage.jsx)
        *   **Code Lines:** `HomePage` functional component (Lines 9-160)
        *   **Detailed Tanglish Explanation:** Application launch aanathum user paakura dashboard interface, drag-and-drop file upload component intha code-la thaan structure panni irukaaru. Server background processes state coordinates (Processing, Success, Logs, error) receive panna Server-Sent Events (SSE) active events handler intha component dynamic state variables hook use panni, UI-ah automatic re-render stream panna setup controller logic-ah code panni irukaaru.
    *   **In-App Camera Scanner & Gallery Upload:**
        *   **File Path:** [CaptureScreen.jsx](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/frontend/src/components/screens/CaptureScreen.jsx)
        *   **Code Lines:** Camera setup and controls: `startCamera`, `stopCamera`, `capturePhoto` (Lines 86-141)
        *   **Detailed Tanglish Explanation:** Physical document or bill copies camera lens direct capture panna client browser WebRTC standard base media devices stream parameters controllers vachu camera feed capture script setup panni irukaaru. Render stream clicks input photo capture logic pixel data format dynamic canvas control feed convert workflow set panni irukaaru.
    *   **Locked Touch Cropping Selection:**
        *   **File Path:** [CaptureScreen.jsx](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/frontend/src/components/screens/CaptureScreen.jsx)
        *   **Code Lines:** Cropping handlers: `handleDragStart`, `handleDragMove`, `handleCropConfirm` (Lines 150-244)
        *   **Detailed Tanglish Explanation:** Users camera capture panni crop selection boundaries mouse drag custom pointer movement nodes screen dynamic coordinate points map structure parameters calculations track logic coding panni coordinates server input feed structures structure panni pixel processing target control build panni irukaaru.
    *   **Results Card Details Rendering:**
        *   **File Path:** [ResultTabs.jsx](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/frontend/src/components/ResultTabs.jsx)
        *   **Code Lines:** `ResultCard` element details card (Lines 123-296)
        *   **Detailed Tanglish Explanation:** Visual logic parser return results layout formatting tabular displays side-by-side elements custom dynamic tabs design control setup intha section thaan build panni design premium UI look display elements configure execute setup panni irukaaru.

---

## 👨‍💻 2. JAYASIMBU J (732722104018)
### **Role: Local Master Backend & Preprocessing Engineer**

*   **Detailed Role Overview:**
    Local server routing controls setup, OpenCV computer vision algorithms logic implementations, raw image scaling contrast enhancements scripts, Indic document layouts split mapping modules, and Excel compiled data generation modules local structures engineer.

*   **Modules Covered & Code References:**
    *   **FastAPI Local Master Orchestration API:**
        *   **File Path:** [main.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/main.py)
        *   **Code Lines:** Routes declarations: `/api/upload` (Lines 61-73), `/api/stream/{job_id}` (Lines 75-128), and `/api/translate` (Lines 138-161)
        *   **Detailed Tanglish Explanation:** Backend server API controllers design endpoints upload pipeline initialization process background trigger modules parameters maps route handlers backend process controllers core coding file logic build setup.
    *   **CLAHE Contrast Enhancer:**
        *   **File Path:** [image_enhancer.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/utils/image_enhancer.py)
        *   **Code Lines:** `enhance_for_vlm` function (Lines 40-70)
        *   **Detailed Tanglish Explanation:** Image document scanning low light shadows pixel parameters contrast levels standard enhance logic-ku OpenCV adaptive threshold base histogram normalization CLAHE functions parameters logic custom grid mapping enhance processing coding compile panni set panni irukaaru.
    *   **Dynamic Tall Split Module:**
        *   **File Path:** [image_enhancer.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/utils/image_enhancer.py)
        *   **Code Lines:** `split_for_extraction` and `split_dual_invoice` (Lines 11-38, 72-108)
        *   **Detailed Tanglish Explanation:** Tall recipes long bills documents vertical pixels size analyzer layout trace boundaries coordinate points overlapping regions blocks-ah partition logic setup pixel slices crop background process backend functions loop execute array compile parameters structure coding.
    *   **Fuzzy Template Mapping:**
        *   **File Path:** [layout_template.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/utils/layout_template.py)
        *   **Code Lines:** `map_to_standard_template` and `_find_value` (Lines 30-70)
        *   **Detailed Tanglish Explanation:** Different document layout types standard parameters list database schemas fields mapping name matching string tokens score calculations custom fuzzy string search algorithm build keys map values compiler backend maps structure script compile function setup.
    *   **Excel Export Compilation:**
        *   **File Path:** [export.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/utils/export.py)
        *   **Code Lines:** `export_to_excel` function (Lines 20-80)
        *   **Detailed Tanglish Explanation:** Raw results data arrays pandas dataframe key maps columns sorting excel format custom layout writing outputs compiler files local device storage directory path output setups script controllers execution setup.

---

## 👨‍💻 3. REMO V (732722104047)
### **Role: Database, Authentication & Security Developer**

*   **Detailed Role Overview:**
    MongoDB Atlas system configuration and database architecture setup, backend authorization token route policies, user logins sessions data records collection validation schema setups, password hashing validation scripts security integrations coding.

*   **Modules Covered & Code References:**
    *   **MongoDB Atlas Schema Models:**
        *   **File Path:** [models.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/db/models.py)
        *   **Code Lines:** `make_invoice_document` dictionary constructor (Lines 12-84)
        *   **Detailed Tanglish Explanation:** Document fields structure dictionary validations rules setup collections indexes coordinates schema model logic definitions parameters config database entries dynamic setup compiler definitions.
    *   **JWT Token Authorization Routing:**
        *   **File Path:** [routes.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/auth/routes.py)
        *   **Code Lines:** Endpoints: `get_current_user_optional` (Lines 20-29), `login` (Lines 48-69), and `register` (Lines 71-87)
        *   **Detailed Tanglish Explanation:** Web login register security JWT web cookies parse extraction verification user identity checks session types active verification dynamic routing backend endpoints code validations compiler functions setup.
    *   **Direct Bcrypt Security Integration:**
        *   **File Path:** [auth_repository.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/db/auth_repository.py)
        *   **Code Lines:** `hash_password`, `verify_password`, and `create_user` (Lines 10-60)
        *   **Detailed Tanglish Explanation:** Password safe hashing and matching algorithm verification. External passlib helper crash errors protect logic standard direct bcrypt methods salt generators verification setup code security module controller repository compiling.
    *   **Email Validation Casing Normalization:**
        *   **File Path:** [routes.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/auth/routes.py)
        *   **Code Lines:** Email casing strip lower in route flows: lines 50, 73
        *   **Detailed Tanglish Explanation:** Database email search filters mismatch bypass user input errors registers blockage check lowercase normal transformations logic auth flow integrations checks.

---

## 👨‍💻 4. SRIGANTH GV (732722104056)
### **Role: System Architect, Remote GPU & Integration Lead**

*   **Detailed Role Overview:**
    System architecture design, backend-frontend pipeline links setup, remote GPU worker inference API connections control, SSE connections timeout protect systems, models input parameters configurations integration architect.

*   **Modules Covered & Code References:**
    *   **Pipeline Orchestration Flow:**
        *   **File Path:** [pipeline.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/pipeline.py)
        *   **Code Lines:** `run_pipeline` function (Lines 20-169)
        *   **Detailed Tanglish Explanation:** Complete extraction workflow: local files processing setup triggers, multi-language easyocr ocr modules execution, layout region analysis setup, remote VLM server prompt request parameters mapping compiler pipeline coordinator script flow logic code compiler setup.
    *   **Remote VLM Inference Request & Lock Serialization:**
        *   **File Path:** [gguf_engine.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/vlm/gguf_engine.py)
        *   **Code Lines:** Locks wrapping: `query_local_llava` (Lines 160-163) and `_query_local_llava_impl` (Lines 165-212)
        *   **Detailed Tanglish Explanation:** Remote model api links configuration. Multiple requests concurrent access backend crashes protect threads semaphores serial locking queues async background logic coding.
    *   **Dynamic Endpoint Settings:**
        *   **File Path:** [main.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/main.py)
        *   **Code Lines:** `/api/settings/vlm_url` (Lines 175-232)
        *   **Detailed Tanglish Explanation:** Tunnel URL changes updates dynamic settings configuration parameters dynamically change active VLM target URL dynamic storage database routes operations script coding controls.
    *   **SSE Streaming Heartbeats Loop:**
        *   **File Path:** [main.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/main.py)
        *   **Code Lines:** Stream endpoint: `/api/stream/{job_id}` (Lines 75-128)
        *   **Detailed Tanglish Explanation:** Processing runs background pipeline execution backend proxies timeouts close prevent SSE connection write loops constant keep-alive message packets yield script control coding.
    *   **Dummy Image Translation Fallback:**
        *   **File Path:** [gguf_engine.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/vlm/gguf_engine.py)
        *   **Code Lines:** 1x1 base64 GIF injection logic (Lines 173-176)
        *   **Detailed Tanglish Explanation:** Text only translation requests remote Qwen/MiniCPM engines images required decode parameter missing crash prevent. 1x1 mock transparent base64 image bytes inject logic translation safety fallback solution setup.
