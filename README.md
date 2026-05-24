# 🧾 Multimodal Document Intelligence System for Invoice and Receipt Processing Using Vision-Language Models with Layout-Aware OCR

## 📌 Abstract
Extracting structured data from highly complex, multilingual Indian invoices and thermal receipts presents a significant challenge due to artistic fonts, overlapping stamps, and diverse structural layouts. Traditional OCR-based pipelines often suffer from cross-script contamination and spatial degradation. This project introduces a **Multimodal Document Intelligence System** that directly maps image pixels to structured JSON and layout-preserving spatial grids (Digital Twins) across 14+ Indian languages, utilizing the state-of-the-art **Qwen2.5-VL-32B** model. 

To overcome local hardware constraints (e.g., standard 4GB VRAM limitations), this system implements an innovative **Master-Worker network** using FastAPI and secure Cloudflare Tunnels to offload heavy inference to Kaggle Dual-T4 GPUs.

---

## 🌊 Process Flow Diagram

```mermaid
graph TD
    A[User Opens Scanner / Uploads File] --> B[Capture Live Video Frame / Select Gallery File]
    B --> C[Interactive Canvas Cropping & Downscaling]
    C --> D[Local FastAPI Master Node]
    D --> E{Is Image Tall?}
    E -->|Yes| F[Dynamic Overlap Splitting]
    E -->|No| G[CLAHE Image Enhancement]
    F --> G
    G --> H[Prompt Orchestration & Context Build]
    H -->|Secure Tunnel with SSE Heartbeats| I[Kaggle Remote Worker Node]
    
    subgraph Kaggle Cloud GPU (Dual T4)
    I --> J[Qwen2.5-VL-32B Processing]
    J --> K[ViT Spatial Feature Extraction]
    K --> L[LLM Cross-Attention Decoding]
    L --> M[JSON & Layout Output Generation]
    end
    
    M -->|REST API Response| N[Local Post-Processing]
    N --> O[Fallback-Safe Translation Merging]
    N --> P[Render Digital Twin Text Grid]
    O --> Q[Generate Excel/JSON/Report Exports]
    Q --> R[MongoDB Persistence]
    R --> S[React UI Dashboard]
    P --> S
```

---

## 🏗️ System Architecture

```mermaid
architecture-beta
    group local(cloud)[Local Environment - 4GB VRAM]
    group remote(cloud)[Kaggle Environment - 30GB VRAM]
    
    service frontend(internet)[React UI & Canvas Cropper] in local
    service backend(server)[FastAPI Master & Export Manager] in local
    service db(database)[MongoDB History] in local
    service enhancement(database)[Image Pre-processor] in local
    
    service tunnel_local(internet)[Cloudflare Client] in local
    service tunnel_remote(internet)[Cloudflare Host] in remote
    
    service gpu(server)[Dual T4 GPUs] in remote
    service model(database)[Qwen2.5-VL-32B] in remote

    frontend:R --> L:backend
    backend:B --> T:enhancement
    backend:B --> T:db
    backend:R --> L:tunnel_local
    tunnel_local:R --> L:tunnel_remote
    tunnel_remote:R --> L:gpu
    gpu:B --> T:model
```

---

## 🔬 Core Algorithms & Methodologies

### 1. End-to-End Layout-Aware Visual OCR
Instead of relying on multi-stage OCR pipelines that are prone to bounding-box alignment errors, this system utilizes the native spatial encoding capabilities of Vision-Language Models (VLMs). The model performs OCR natively within its visual transformer block, analyzing the physical layout matrices directly. This allows it to bypass issues with stylized fonts and narrow thermal receipts without relying on brittle algorithmic heuristics.

### 2. Multimodal Fusion Architecture
The system employs a tightly coupled ViT (Vision Transformer) and LLM (Large Language Model) architecture. The visual encoder extracts rich spatial-semantic features from the document, which are cross-attended by the language decoder. This multimodal fusion allows the model to "read" the text while simultaneously understanding its structural context (e.g., distinguishing a 'Total' value from a 'Tax' value based purely on spatial positioning).

### 3. Dual-Language Spatial Translation
The "Master Prompt" algorithm leverages Chain-of-Thought (CoT) zero-shot prompting to force the AI into generating a simultaneous, dual-domain extraction:
*   **Native Domain**: Mathematically maps the exact original script (e.g., Tamil, Hindi) into a preserved spatial JSON structure.
*   **English Domain**: Performs semantic structural translation, allowing centralized ERP systems to process vernacular invoices in English without losing the spatial context.

### 4. Digital Twin Reconstruction Grid
The system bypasses geometric post-processing by commanding the VLM to natively generate a physical `.txt` grid representation of the document. This "Digital Twin" visually mimics the 2D spatial arrangement of the original invoice, preserving column alignments and visual hierarchy for human verification.

### 5. Mobile-Optimized Live Camera Video Scanner
To circumvent mobile OS crashes caused by high-resolution camera native applications taking over browser tab memory, this system implements a direct in-app media stream viewfinder via `navigator.mediaDevices.getUserMedia`. It routes the raw feed into an emerald scanner canvas overlay with animated guide frames, capturing optimized frames on demand.

### 6. Viewport-Locked Drag-and-Drop Crop Mechanics
To ensure touch-dragging does not cause page elastic bounce/scroll on mobile, the cropping interface wraps the target image tightly within a `relative inline-block` CSS wrapper configured with `touch-action: none`. Touch gestures are mapped dynamically to absolute pixel coordinates and scaled accurately against the image's source resolution regardless of responsive screen rendering. It caps cropped outputs to a maximum dimension of 1200px to avoid GPU memory overhead.

### 7. Fallback-Safe Translation Merging
To resolve partial or empty translation issues, the backend pipeline runs a merging algorithm: it initialises a dictionary with the original native extraction fields and overlays translated values from the model's `english_json` block where present. If a translation is omitted by the VLM, the native value remains intact, ensuring zero data loss in the standard layout view.

### 8. Keep-Alive SSE Heartbeat Loop
To prevent Cloudflare's strict 100-second idle connection timeout from closing the API stream during remote GPU VLM inference passes, the server implements an asynchronous keep-alive loop. It periodically yields standard SSE comment packets (`: ping\n\n`) every 15 seconds, keeping the tunnel connection warm.

### 9. Concurrency Serialization Lock
To guarantee multi-user stability on shared remote worker nodes (like Kaggle free Dual-T4 instances), the inference pipeline runs requests through a global threading synchronization lock (`_vlm_lock`). This prevents simultaneous VLM queries from exceeding available VRAM and causing GPU out-of-memory crashes.

### 10. Integrated Data Exports & Database Persistence
Every successful document extraction automatically outputs high-quality downloads of:
*   **Excel Spreadsheets (`.xlsx`)** mapping fields canonicalised by the Standardized Layout Template.
*   **JSON Data Structure** saving both raw and template fields.
*   **Notepad-style Structured Reports (`.txt`)** grouping headers, bodies, tables, and footers.
*   **Digital Twin Text Blocks** preserving vertical spatial structures.
All data payloads are saved via a MongoDB Motor client for visual historical audits on the dashboard.

---

## 🧮 Mathematical Formulations & Techniques

### 1. Contrast Limited Adaptive Histogram Equalization (CLAHE)
Thermal receipts frequently suffer from illumination gradients and faded text. Before inference, the local node applies CLAHE to maximize local contrast without amplifying noise. Unlike standard histogram equalization, CLAHE operates on small tiles (blocks) of the image and applies **Bilinear Interpolation** to stitch the results seamlessly.
**Formula:**
$$h(v) = \text{round}\left( \frac{cdf(v) - cdf_{min}}{(M \times N) - cdf_{min}} \times (L - 1) \right)$$
*Where $cdf(v)$ is the cumulative distribution function, $M \times N$ is the total number of pixels in the tile, and $L$ is the maximum pixel value.*

### 2. Naive Dynamic Resolution Algorithm
The Qwen2.5-VL engine utilizes a proprietary dynamic resolution algorithm that allows it to process images of arbitrary aspect ratios (very tall receipts or wide invoices) without resizing them into a fixed square. This preserves the pixel density of small fonts.
**Logic:** The image is dynamically partitioned into a variable number of visual tokens based on the original aspect ratio, ensuring no structural information is lost during the compression phase.

### 3. Vision Transformer (ViT) Spatial Self-Attention
The Qwen2.5-VL model replaces traditional OCR by using self-attention to correlate localized image patches (e.g., a printed price) with global structural context (e.g., the "Total" header).
**Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
*Where $Q, K, V$ represent the Query, Key, and Value matrices derived from the image patches.*

### 4. Dynamic Document Splitting Algorithm
To prevent Vision-Transformer token overflow on extremely tall grocery receipts, an overlap splitting algorithm divides the image into manageable chunks while preserving contextual boundaries.
**Formula:**
$$S_i = I[y_i : y_i + H_{chunk}, 0 : W]$$
*Where $y_{i+1} = y_i + H_{chunk} - H_{overlap}$. Results are then logically stitched during post-processing.*

### 5. Distributed Hardware Offloading
Running a 32-Billion parameter model typically requires enterprise-grade hardware. By utilizing `llama-cpp-python` with `IQ4_XS` quantization and continuous batching across Kaggle's free Dual-T4 GPUs, the system achieves enterprise-grade extraction accuracy on zero-budget infrastructure.

---

## 🧠 Model Intelligence & Training Methodology

The "Brain" of this system is the **Qwen2.5-VL-32B**, which was developed using a multi-stage training pipeline designed for high-accuracy document intelligence.

### 1. Model Architecture
- **Visual Encoder**: A Vision Transformer (ViT) with ~600M parameters that handles native 2D spatial encoding.
- **Language Decoder**: A 32-Billion parameter causal language model optimized for multilingual reasoning.
- **Modality Bridge**: Uses **Gated Cross-Attention** to fuse visual features directly into the language processing stream.

### 2. Training Datasets
The model was trained on a massive multimodal corpus, including:
- **Image-Text Pairs**: Billions of samples for basic visual-concept alignment.
- **Document Datasets**: Specialized fine-tuning on **DocVQA** (Document Visual Question Answering), **ChartQA**, and **DeepForm** (Invoice/Form datasets).
- **Indic-Specific Corpora**: Large-scale crawl of Indian vernacular scripts to ensure high-accuracy OCR for Devanagari, Tamil, etc.

### 3. Training Phases
1. **Pre-training**: Large-scale unsupervised learning for general visual understanding.
2. **Supervised Fine-Tuning (SFT)**: Learning to follow specific instructions (e.g., "Convert this invoice to JSON").
3. **Alignment (RLHF/DPO)**: Reinforcement Learning from Human Feedback ensures the model avoids hallucinations and strictly follows formatting rules.

---

## 📋 Examiner's Quick Reference (Project Logic)

| Question | Technical Answer | Source / Reference |
|:---|:---|:---|
| **What Algorithm for Faded Text?** | **CLAHE** (Contrast Limited Adaptive Histogram Equalization) | **Stephen Pizer** (UNC Chapel Hill) |
| **How does it read handwriting?** | **Spatial Self-Attention** in the ViT layer. | **Google Brain** (Dosovitskiy et al.) |
| **How are 32B models run locally?** | **Distributed Master-Worker Architecture**. | **FastAPI / Cloudflare Tunneling** |
| **What is the OCR Engine?** | **End-to-End Visual OCR** (Integrated in VLM). | **Alibaba Cloud (Qwen Team)** |
| **How is Hindi/Tamil handled?** | **Multilingual SFT** on Indic-script datasets. | **Alibaba Qwen-VL Team** |
| **Transformer Logic** | Self-Attention Mechanism ($Q, K, V$ Matrices). | **Google Research** (Vaswani et al.) |

## 🌏 Supported Languages
The visual reasoning engine natively supports and translates **14+ Indian Languages**:
`Hindi, Bengali, Tamil, Telugu, Kannada, Gujarati, Malayalam, Marathi, Odia, Punjabi, Urdu, Assamese, Maithili, Sindhi` + `English`.

---

## 🛠️ Project Structure
 
 ```text
 Final-Sem-Proj-Updated/
 ├── backend/
 │   ├── main.py                    # FastAPI entry point & API route mapping
 │   ├── pipeline.py                # Image orchestration & remote worker dispatcher
 │   ├── config.py                  # Central configuration & VLM parameters
 │   ├── auth/
 │   │   └── routes.py              # User authentication, JWT tokens & routes
 │   ├── db/
 │   │   ├── connection.py          # MongoDB client lazily-loaded instance connection
 │   │   ├── auth_repository.py     # User creation and query error handlers (503 status code)
 │   │   └── repository.py          # Invoice CRUD queries & regex text searches
 │   ├── vlm/
 │   │   ├── vlm_model.py           # VLM Prompts, dynamic translation & digital twin mapping
 │   │   └── gguf_engine.py         # Cloudflare remote worker REST client
 │   └── utils/
 │       ├── image_enhancer.py      # Local pre-processing (CLAHE & Tall split logic)
 │       ├── layout_template.py     # Standardized JSON field schemas (mapping vernacular -> English)
 │       └── export.py              # Document output compile helpers (.xlsx, .json, .txt)
 ├── frontend/
 │   ├── src/
 │   │   ├── components/            
 │   │   │   ├── Dashboard.jsx      # SVG Chart Analytics & case-insensitive keyword searches
 │   │   │   ├── AuthModal.jsx      # Token validations, signup hints & fetch safety
 │   │   │   └── ResultTabs.jsx     # Digital Twin renders, raw templates & downloads
 │   │   └── context/
 │   │       └── AuthContext.jsx    # Client-side user auth state & session storage
 ├── PROJECT_FLOW.md                # Comprehensive step-by-step pipeline execution flow (Mermaid diagrams)
 ├── .env                           # API and database environment keys
 └── README.md                      # General system overview & core algorithmic explanations
 ```
 
 ---
 
 ## 📖 Pipeline Flow Documentation
 For a highly detailed step-by-step description of the data extraction lifecycle (visual cropping, server-sent heartbeat events, remote Kaggle node locks, and translation merging), refer to [PROJECT_FLOW.md](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti%20/Updated%20Final%20Year%20Project/PROJECT_FLOW.md).

---

## ⚙️ Setup & Deployment Guide

### 1. Local Backend Setup

```bash
pip install -r requirements.txt
cd backend
python main.py
```
*API serves locally at `http://localhost:8000`*

### 2. Kaggle Worker Node Deployment
Due to the intensive VRAM requirements of the 32B VLM, the inference engine is deployed remotely:
1. Initialize a Kaggle Notebook and enable **Dual T4 GPUs**.
2. Execute the provided inference cell to spin up `llama-cpp-python` alongside a `cloudflared` tunnel.
3. Upon initialization, copy the secure `.trycloudflare.com` URL generated in the output logs.

### 3. Environment Configuration (`.env`)
Create a `.env` file in the root directory to establish the master-worker handshake:
```env
KAGGLE_VLM_URL=https://your-generated-url.trycloudflare.com
INTERNAL_MODEL_API_KEY=inv_ai_sk_d7c8dc5d523d4bffa8d1a08483f7e3ac
```

### 4. Frontend Launch

```bash
cd frontend
npm install
npm run dev
```
*UI accessible at `http://localhost:5173`*

---

## 📱 Mobile & Multi-Device Camera Testing Guide

To test the system on real mobile devices (and utilize the native mobile camera for receipt scanning), follow one of the methods below. 

### Prerequisites
1. Ensure the Kaggle worker node is running and the tunnel URL is copied.
2. Update the local `.env` file with the Kaggle URL:
   ```env
   KAGGLE_VLM_URL=https://your-kaggle-worker.trycloudflare.com
   ```

---

### Comparison of Testing Methods

| Metric | Path A: Production Build (Recommended for Demos) | Path B: Vite Dev Server (Recommended for Coding) |
| :--- | :--- | :--- |
| **How it Works** | React is built (`npm run build`). FastAPI serves static files directly. | Vite runs a hot-reloading dev server. Proxies `/api` to FastAPI. |
| **Servers Run** | Only Local Backend (`python main.py` on Port 8000). | Local Backend (Port 8000) + Vite Dev Server (Port 5173). |
| **Tunnels Run** | Single Tunnel for Port 8000. | Single Tunnel for Port 5173. |
| **Live Updates**| No. Must rebuild (`npm run build`) to see UI changes. | Yes. Hot-reloads UI instantly when React code is changed. |
| **System Overhead**| Low. Only one Node/Python server running. | Medium. Multiple active processes. |

---

### 🚀 Step-by-Step Execution

#### Method A: Serving Production Build (Simplified)
1. **Build the Frontend:**
   ```bash
   cd frontend
   npm run build
   ```
2. **Start the FastAPI Backend:**
   ```bash
   cd ../backend
   python main.py
   ```
3. **Expose the Backend via Tunnel:**
   ```bash
   # Run from the root directory:
   .\backend\cloudflared.exe tunnel --url http://localhost:8000
   ```
4. **Open on Mobile:** Load the generated HTTPS URL on your phone's browser. The mobile camera button will trigger the native camera, capture the photo, upload it to the local backend, offload to Kaggle VLM, and return results.

#### Method B: Dev Server Proxy (Live Development)
1. **Start the FastAPI Backend:**
   ```bash
   cd backend
   python main.py
   ```
2. **Start the Vite Dev Server:**
   ```bash
   cd frontend
   npm run dev
   ```
3. **Expose the Dev Server via Tunnel:**
   ```bash
   # Run from the root directory:
   .\backend\cloudflared.exe tunnel --url http://localhost:5173
   ```
4. **Open on Mobile:** Load the generated HTTPS URL on your phone's browser. Live UI modifications will automatically reflect on the mobile screen.

---

## 🛡️ License & Academic Disclaimer
Developed as an academic Final Year Semester Project focusing on bridging hardware gaps in Large Vision-Language Model deployment for regional Indian contexts.
