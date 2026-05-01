# 🧾 Invoice AI — Multilingual Invoice Extraction System

A production-ready web application for automated invoice data extraction from images in **14 Indian languages**. Combines GPU-accelerated OCR, Vision-Language Models, and adaptive image preprocessing to convert unstructured invoice images — including handwritten and artistic-font documents — into structured, standardised data.

---

## 🏗️ Architecture

```
Upload Image
     │
     ▼
┌────────────────────────────────────────────┐
│  PHASE 1 — Dual-Path Image Enhancement     │
│  ┌──────────────────┐  ┌────────────────┐  │
│  │ OCR Path         │  │ VLM Path       │  │
│  │ Adaptive B&W     │  │ Color-preserved│  │
│  │ CLAHE + Denoise  │  │ CLAHE + Sharpen│  │
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
         │ → Discard hint │
         └────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│  PHASE 2 — Standalone CUDA Llama Server    │
│  Qwen3-VL-4B (GGUF Q4_K_M)               │
│  100% GPU offload — NVIDIA GTX 1650        │
│  Smart alphabet injection (per-script)     │
│  Empty output → auto-retry image-only      │
└────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│  PHASE 3 — Layout Template Mapper          │
│  Raw VLM fields → Standardised Schema      │
│  Same field positions across all invoices  │
└────────────────────────────────────────────┘
                  │
                  ▼
         Excel / JSON Export
```

---

## ✨ Key Features

### 🌏 14-Language Multilingual Support
Full character reference alphabets for:

| Script Family | Languages |
|:---|:---|
| Devanagari | Hindi, Marathi, Maithili |
| Bengali-Assamese | Bengali, Assamese |
| Dravidian | Tamil, Telugu, Kannada, Malayalam |
| Indo-Aryan | Gujarati, Odia, Punjabi (Gurmukhi) |
| Arabic-based | Urdu, Sindhi |

### 🤖 Smart Script Detection
- Detects Unicode script ranges in OCR output automatically
- Injects **only relevant** language alphabets into VLM prompt (prevents context overflow)
- Cross-script contamination detection: if OCR reads multiple conflicting scripts (e.g., artistic fonts), hint is discarded and VLM reads image directly
- Auto-retry with image-only mode when VLM returns empty output

### 🖼️ Dual-Path Image Enhancement
| Path | Purpose | Processing |
|:---|:---|:---|
| **OCR Path** | High contrast for text recognition | Upscale → Deskew → Denoise → CLAHE → Adaptive B&W → Sharpen |
| **VLM Path** | Color-preserved for layout understanding | Upscale → Deskew → CLAHE → Conservative Sharpen |

Both paths run **in parallel** — zero added latency.

### 📐 Standardised Layout Template
Every extraction — regardless of language, font style, or image quality — maps to the **same fixed schema**:

```
┌──────────────────────┬──────────────────────┐
│ vendor_name          │ invoice_number        │
│ vendor_address       │ invoice_date          │
│ vendor_gstin         │ due_date / po_number  │
├──────────────────────┼──────────────────────┤
│ buyer_name           │ buyer_address         │
│ buyer_gstin          │                       │
├──────────────────────┴──────────────────────┤
│              items (line items table)        │
├─────────────────────────────────────────────┤
│ subtotal │ cgst │ sgst │ igst │ total_amount │
├─────────────────────────────────────────────┤
│ bank_name │ account_number │ ifsc │ upi      │
└─────────────────────────────────────────────┘
```

Missing fields appear as empty — never silently dropped. Especially useful for handwritten or low-quality invoice images.

### ⚡ 100% GPU Execution on 4GB VRAM
- Standalone `llama-server.exe` subprocess (llama.cpp release build with CUDA 12.4)
- 99 layers offloaded to NVIDIA GPU — zero CPU inference
- Automatic VRAM release between OCR and VLM phases (prevents OOM)
- Context window: 8192 tokens with smart prompt sizing

### 📱 Real-time Web UI
- React + Vite frontend with live progress via Server-Sent Events (SSE)
- Tabs: OCR → Layout → VLM → Exports
- Export as Excel (`.xlsx`) or JSON

---

## 🛠️ Project Structure

```
Final-Sem-Proj-Updated/
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── pipeline.py                # 3-phase orchestration engine
│   ├── config.py                  # Model paths, context settings
│   ├── ocr/
│   │   ├── engine.py              # PaddleOCR wrapper
│   │   └── easyocr_engine.py      # EasyOCR (all Indic scripts)
│   ├── vlm/
│   │   ├── vlm_model.py           # VLM prompt building + smart injection
│   │   └── gguf_engine.py         # Standalone llama-server CUDA client
│   ├── utils/
│   │   ├── image_enhancer.py      # Dual-path preprocessing (OCR + VLM)
│   │   ├── layout_template.py     # Standardised field schema mapper
│   │   ├── image_optimizer.py     # Legacy VLM resize wrapper
│   │   └── export.py              # Excel + JSON export
│   └── language_alphabets/        # 14 Indian language character reference files
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
