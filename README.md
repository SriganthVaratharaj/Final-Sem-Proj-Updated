# 🧾 Invoice AI — Multilingual Invoice Digital Twin System

A production-ready web application for automated invoice data extraction and **spatial digital twin reconstruction** from images in **14 Indian languages**. Combines GPU-accelerated OCR, LayoutLMv3, Vision-Language Models (VLM), and adaptive image compounding to convert unstructured invoice images — including handwritten and artistic-font documents — into structured, standardised data and layout-preserving Markdown/TXT exports.

---

## 🏗️ Architecture

```text
Upload Image
     │
     ▼
┌────────────────────────────────────────────┐
│  PHASE 1 — Image Preprocessing             │
│  ┌──────────────────┐  ┌────────────────┐  │
│  │ OCR Path         │  │ VLM Path       │  │
│  │ Dual-Pass B&W    │  │ Color-preserved│  │
│  │ (Sauvola style)  │  │ CLAHE + Sharpen│  │
│  └──────────────────┘  └────────────────┘  │
└────────────────────────────────────────────┘
     │                          │
     ▼                          ▼
┌─────────────────┐    ┌──────────────────────┐
│  PaddleOCR      │    │  EasyOCR             │
│  (Sequential)   │    │  (All Indic Scripts) │
└─────────────────┘    └──────────────────────┘
     │                          │
     └──────── MERGE ───────────┘
                  │
         Smart OCR Hint QC
         ┌────────────────┐
         │ Cross-script   │
         │ contamination? │
         │ → Keep Hint    │
         └────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│  PHASE 2 — Spatial Layout & Compounding    │
│  1. OpenCV Table/Grid Detection            │
│  2. LayoutLMv3 Regional Analysis (Local)   │
│  3. Image Compounding (Full Image +        │
│     High-Res Table Zoom) sent to VLM       │
└────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│  PHASE 3 — Standalone CUDA Llama Server    │
│  Qwen3-VL-4B (GGUF Q4_K_M)                 │
│  100% GPU offload — NVIDIA GTX 1650        │
│  Smart alphabet injection (per-script)     │
└────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│  PHASE 4 — Digital Twin & Standardization  │
│  Raw VLM fields → Standardised Schema      │
│  Markdown Table → Spatial TXT / DOCX       │
└────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🌏 14-Language Multilingual Support
Full character reference alphabets for Devanagari, Bengali, Dravidian, Indo-Aryan, and Arabic-based scripts. Smart Unicode detection injects only relevant alphabets to prevent token overflow.

### 🖼️ Advanced Image Compounding (Divide & Conquer)
- **Dual-Pass Binarization:** Uses Sauvola-inspired local adaptive thresholding to preserve tiny Indic matras (vowels) which standard Otsu destroys.
- **Image Compounding:** Generates a composite image containing the full document + a high-resolution zoomed crop of the detected table region. This gives the VLM both "Big Picture" context and "Close-up" clarity for small table fonts.

### 📐 Digital Twin (Spatial Reconstruction)
- **OpenCV Table Detection:** Automatically finds and draws table bounding boxes to visually guide the VLM.
- **LayoutLMv3:** Local fallback execution for region classification (Header, Body, Footer).
- **Format Preservation:** Exports extracted data into a visually faithful `.txt` and `.docx` spatial grid, retaining the original invoice layout (Digital Twin).

### ⚡ 100% GPU Execution on 4GB VRAM
- Strict sequential execution with explicit memory clearing (`release_gpu_memory`, `release_layoutlm_memory`, `release_vlm_memory`) prevents PyTorch/Paddle CUDA Out-Of-Memory errors on a 4GB GTX 1650.
- Runs `llama-server.exe` (llama.cpp) as a standalone subprocess for the 4B Vision Language Model.

---

## 🛠️ Project Structure

```text
Final-Sem-Proj-Updated/
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── pipeline.py                # 4-phase orchestration engine
│   ├── config.py                  # Model paths, context settings
│   ├── layout/
│   │   ├── layoutlm_service.py    # LayoutLMv3 API & Local execution
│   │   └── box_adapter.py         # Box scaling utilities
│   ├── ocr/
│   │   ├── engine.py              # PaddleOCR wrapper
│   │   └── easyocr_engine.py      # EasyOCR (all Indic scripts)
│   ├── vlm/
│   │   ├── vlm_model.py           # VLM prompt building
│   │   └── gguf_engine.py         # Standalone llama-server client
│   ├── utils/
│   │   ├── image_enhancer.py      # Image compounding & dual-pass B&W
│   │   ├── layout_reconstructor.py# Digital Twin TXT/DOCX generation
│   │   ├── table_detector.py      # OpenCV table contour detection
│   │   ├── layout_template.py     # Standardised schema mapper
│   │   └── export.py              # Excel + JSON export
│   └── language_alphabets/        # 14 Indian language txt references
│       ├── tamil.txt
│       ├── hindi.txt
│       ├── bengali.txt
│       ├── gujarati.txt
│       ├── marathi.txt
│       ├── telugu.txt
│       ├── kannada.txt
│       ├── malayalam.txt
│       ├── odia.txt
│       ├── punjabi.txt
│       ├── urdu.txt
│       ├── assamese.txt
│       ├── maithili.txt
│       └── sindhi.txt
├── frontend/
│   └── src/
│       └── components/
│           └── tabs/              # OCR / Layout / VLM / Exports tabs
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.12
- Node.js 18+
- NVIDIA GPU with CUDA 12.x (recommended: 4GB+ VRAM)
- `llama-server.exe` (CUDA build) placed in `backend/vlm/llama_server_bin/`

### 1. Backend

```bash
pip install -r requirements.txt
pip install "numpy==1.26.4" --force-reinstall   # Required for PaddleOCR compatibility
cd backend
python main.py
```

API available at `http://localhost:8000`

### 2. Download Llama Server Binary

Download the CUDA-enabled `llama-server.exe` from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) and place it in:
```
backend/vlm/llama_server_bin/llama-server.exe
```

### 3. Download VLM Model

Download `Qwen3VL-4B-Instruct-Q4_K_M.gguf` and its mmproj file, place in the path configured in `backend/config.py`.

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

UI at `http://localhost:5173`

---

## 📋 Environment Configuration (`.env`)

| Variable | Description |
|:---|:---|
| `JWT_SECRET` | Secret key for auth token signing |
| `MAX_UPLOAD_SIZE_MB` | Maximum file upload size (default: `10`) |
| `VLM_LOCAL_MODEL_PATH` | Path to the GGUF model file |
| `VLM_LOCAL_MMPROJ_PATH` | Path to the mmproj (vision encoder) file |

---

## 🔬 Known Limitations & Mitigations

| Limitation | Mitigation |
|:---|:---|
| Artistic / custom fonts unreadable by OCR | Cross-script detection → VLM reads image directly |
| 4GB VRAM tight with 8192 context | Sequential OCR→VLM with explicit VRAM release |
| Handwritten bills vary per person | CLAHE equalisation + template fills all fields |
| PaddleOCR + EasyOCR CUDA type collision | Sequential execution (not parallel) |
| NumPy 2.x breaks PaddleOCR | Pinned to numpy==1.26.4 |

---

## 🛡️ License & Disclaimer

Developed as a Final Year Semester Project. All temporary guest data is purged automatically. The VLM model (Qwen3-VL) is used under its respective open-source license.
