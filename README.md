# 🧾 Multimodal Document Intelligence System for Invoice and Receipt Processing Using Vision-Language Models with Layout-Aware OCR

## 📌 Abstract
Extracting structured data from highly complex, multilingual Indian invoices and thermal receipts presents a significant challenge due to artistic fonts, overlapping stamps, and diverse structural layouts. Traditional OCR-based pipelines often suffer from cross-script contamination and spatial degradation. This project introduces a **Multimodal Document Intelligence System** that directly maps image pixels to structured JSON and layout-preserving spatial grids (Digital Twins) across 14+ Indian languages, utilizing the state-of-the-art **Qwen2.5-VL-32B** model. 

To overcome local hardware constraints (e.g., standard 4GB VRAM limitations), this system implements an innovative **Master-Worker network** using FastAPI and secure Cloudflare Tunnels to offload heavy inference to Kaggle Dual-T4 GPUs.

---

## 🌊 Process Flow Diagram

```mermaid
graph TD
    A[User Uploads Invoice/Receipt] --> B[Local FastAPI Master Node]
    B --> C{Is Image Tall?}
    C -->|Yes| D[Dynamic Overlap Splitting]
    C -->|No| E[CLAHE Image Enhancement]
    D --> E
    E --> F[Prompt Orchestration & Context Build]
    F -->|Secure Cloudflare Tunnel| G[Kaggle Remote Worker Node]
    
    subgraph Kaggle Cloud GPU (Dual T4)
    G --> H[Qwen2.5-VL-32B Processing]
    H --> I[ViT Spatial Feature Extraction]
    I --> J[LLM Cross-Attention Decoding]
    J --> K[JSON & Layout Output Generation]
    end
    
    K -->|REST API Response| L[Local Post-Processing]
    L --> M[Parse Native & English JSON]
    L --> N[Render Digital Twin Text Grid]
    M --> O[React UI Dashboard]
    N --> O
```

---

## 🏗️ System Architecture

```mermaid
architecture-beta
    group local(cloud)[Local Environment - 4GB VRAM]
    group remote(cloud)[Kaggle Environment - 30GB VRAM]
    
    service frontend(internet)[React UI] in local
    service backend(server)[FastAPI Master] in local
    service enhancement(database)[Image Pre-processor] in local
    
    service tunnel_local(internet)[Cloudflare Client] in local
    service tunnel_remote(internet)[Cloudflare Host] in remote
    
    service gpu(server)[Dual T4 GPUs] in remote
    service model(database)[Qwen2.5-VL-32B] in remote

    frontend:R --> L:backend
    backend:B --> T:enhancement
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
│   ├── main.py                    # FastAPI entry point
│   ├── pipeline.py                # Image orchestration and Kaggle dispatcher
│   ├── config.py                  # API endpoints and system configurations
│   ├── vlm/
│   │   ├── vlm_model.py           # Master Prompt engineering & JSON parsing
│   │   └── gguf_engine.py         # Cloudflare tunnel REST client
│   └── utils/
│       ├── image_enhancer.py      # CLAHE algorithms & image splitting
│       ├── layout_template.py     # Schema standardization algorithms
│       └── export.py              # System export logic
├── frontend/                      # React-based UI mapping extraction results
├── .env                           # Cloudflare & API configuration
└── README.md
```

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

## 🛡️ License & Academic Disclaimer
Developed as an academic Final Year Semester Project focusing on bridging hardware gaps in Large Vision-Language Model deployment for regional Indian contexts.
