# 📊 Finalized PPTX Presentation Content Guide
This guide contains the exact updated text content for your presentation (**final review.pptx**), aligned with the latest report (**project report Copy5.pdf**).

---

## 📽️ Slide 1: Title Slide
*   **Slide Title:** MULTIMODAL DOCUMENT INTELLIGENCE SYSTEM FOR INVOICE AND RECEIPT PROCESSING USING VISION-LANGUAGE MODELS WITH LAYOUT-AWARE OCR
*   **Presented by:**
    1. Sriganth G V (732722104056)
    2. Jayasimbu J (732722104018)
    3. Aadhiseshan S (732722104001)
    4. Remo V (732722104047)
*   **Under Guidance of:**
    *   Dr. P. Nandhini, M.E., Ph.D.
    *   Associate Professor & HOD, Department of CSE
*   **Institution:** Sri Shanmugha College of Engineering and Technology

---

## 📽️ Slide 2: Abstract (Two Options)

### Option A (With Points):
*   **Slide Title:** Abstract
*   **Content:**
    *   **The Problem:** Automated invoice parsing fails on degraded receipts and regional Indic script code-switching.
    *   **The Model:** Direct visual extraction using OCR-free, pixel-level Qwen2.5-VL-32B Vision-Language Model.
    *   **Preprocessing:** OpenCV CLAHE enhances faded print; dynamic aspect-ratio splitting handles tall receipts.
    *   **Infrastructure:** Distributed Master-Worker setup offloads quantized tensors to remote dual-T4 GPUs via Cloudflare.
    *   **Result:** Reaches 96% Key-Value Accuracy on invoices and 92% on degraded thermal receipts.

### Option B (Without Points):
*   **Slide Title:** Abstract
*   **Content:**
    > *In the modern business environment, extracting structured data from commercial documents such as invoices and receipts is a critical challenge. Legacy OCR-based pipelines systematically fail on degraded print, complex layout structures, and regional script code-switching. To address this, we propose an OCR-free Multimodal Document Intelligence System (MDIS) driven by the Qwen2.5-VL-32B model. Using CLAHE contrast normalization, dynamic overlap height splitting, and secure Cloudflare Tunnels to remote dual-T4 GPUs, our system directly maps document pixels to structured JSON schemas and monospace ASCII layout reconstructions. It achieves a Key-Value Accuracy of 96% on invoices and 92% on degraded receipts.*

---

## 📽️ Slide 3: Problem Definition (Two Options)

### Option A (With Points):
*   **Slide Title:** Problem Definition
*   **Content:**
    *   **Early Error Propagation:** Bounding box and text block detection errors cascade in multi-stage OCR pipelines.
    *   **Document Degradation:** Print fading, folding creases, and low-contrast artifacts in thermal receipts.
    *   **Script Code-Switching:** English, numbers, and Indic regional scripts mixed together on the same document.
    *   **Rigid Templates:** Rule-based layout extraction engines fail when invoice formats or schemas change.
    *   **Loss of Layout Context:** Legacy OCR extracts flat text strings, breaking spatial item-to-pricing alignments.

### Option B (Without Points):
*   **Slide Title:** Problem Definition
*   **Content:**
    > *Traditional document understanding relies on sequential text detection and character recognition pipelines. This multi-stage approach is highly prone to error propagation where early bounding box inaccuracies cascade downstream. Furthermore, thermal receipts suffer from print fading, creasing, and low-contrast artifacts. Multilingual environments introduce script code-switching, where regional texts and English numerals are mixed, causing standard binarisation tools to fail. Additionally, template-based rule engines are not scalable and lose vital spatial relationships, such as aligning item names directly to their horizontal prices.*

---

## 📽️ Slide 4: Literature Survey (Table 1)
*   **Slide Title:** Literature Survey - Part 1
*   **Table Content:**

| Author & Year | Method Used | Features | Limitations | Proposed System Advantages |
| :--- | :--- | :--- | :--- | :--- |
| **Dosovitskiy et al. (2021) [1]** | Vision Transformer (ViT) | Processes document images as patch token sequences instead of convolutions. | Lacks language-decoding layers for text and structured data generation. | Integrated the ViT patch encoder with a 32B language decoder (Qwen2.5-VL) for direct pixel-to-JSON generation. |
| **Qwen Team (2023) [3]** | Vision-Language Model (VLM) | Joint image-text reasoning, visual grounding, and multi-script parsing. | High GPU computational demands limit deployment on local edge hardware. | Offloaded quantized (4-bit GGUF) model inference to remote dual-T4 GPUs via Cloudflare Tunnels. |
| **Pizer et al. (1987) [6]** | Contrast Limited AHE | Contrast Limited Adaptive Histogram Equalization (CLAHE) for local enhancement. | Designed for image contrast; cannot extract structured text or semantics. | Applied OpenCV local CLAHE to restore faded thermal receipts, improving model visual patch legibility. |

---

## 📽️ Slide 5: Literature Survey (Table 2)
*   **Slide Title:** Literature Survey - Part 2
*   **Table Content:**

| Author & Year | Method Used | Features | Limitations | Proposed System Advantages |
| :--- | :--- | :--- | :--- | :--- |
| **Kim et al. (2022) [5]** | OCR-Free Document Transformer | Bypasses OCR by directly mapping visual input pixels to structured target text. | Struggles with visual budget overflow on tall images or receipt strips. | Implemented a vertical aspect splitting algorithm with delta overlap margins to prevent token overflow. |
| **Du et al. (2020) [11]** | Multi-stage Pipeline OCR | Lightweight text detection and character recognition on edge devices. | Lacks spatial layout reasoning; bounding-box errors cascade downstream. | Used pixel-level gated cross-attention to naturally map layout components without coordinate lookup. |
| **Liao et al. (2020) [19]** | Differentiable Binarization | DBNet within segmentation for real-time text detection. | Highly dependent on clean thresholds; fails to translate regional Indic scripts. | Implemented Chain-of-Thought (CoT) dual-prompting to transcribe and translate Indic scripts in one pass. |

---

## 📽️ Slide 6: Existing System & Demerits
*   **Slide Title:** Existing System & Demerits
*   **System Description (Points):**
    *   **Technology Foundation:** Heavily relies on traditional Optical Character Recognition (OCR) engines (such as Tesseract or EasyOCR).
    *   **Workflow Pipeline:** Sequential execution starting with raw character binarisation followed by coordinate-based block mapping.
    *   **Limitations:** Incapable of dynamically generalizing to layout formats outside of hardcoded coordinate rules, leading to manual auditing.
*   **Demerits (Points):**
    *   Sequential layout and character recognition errors cascade downstream.
    *   Complete failure on low-contrast, degraded, folded, or faded thermal receipt print.
    *   Inability to preserve spatial relationships (loses item-price alignments).
    *   Lack of native support for mixed-language Indic script code-switching.
    *   Rigid coordinate templates that require manual re-configuration for new layouts.

---

## 📽️ Slide 7: Proposed System & Merits
*   **Slide Title:** Proposed System & Merits
*   **System Description (Points):**
    *   **OCR-Free Model:** Directly maps visual features of document pixels to structured outputs using the Qwen2.5-VL-32B VLM model.
    *   **Integrated Pipeline:** Employs OpenCV local CLAHE enhancement and dynamic vertical overlap height splitting for tall receipts.
    *   **Distributed Host:** Bypasses local VRAM limitations via secure Cloudflare Tunnels to remote dual NVIDIA T4 GPU worker nodes.
*   **Merits (Points):**
    *   OCR-free end-to-end processing eliminates text-detection error propagation.
    *   Adaptive contrast preprocessing (CLAHE) restores faded thermal ink legibility.
    *   Aspect-ratio preserving overlap splitting handles tall receipts without token overflow.
    *   Monospace ASCII Digital Twin outputs enable spatial layout auditing.
    *   Zero-cost distributed hosting runs heavy models on cheap local machines.

---

## 📽️ Slide 8: System Architecture
*   **Slide Title:** System Architecture
*   **Content:**
    *   **Distributed Master-Worker Topology:** Split-node system that runs a massive 32-B parameter model on remote hardware while keeping local client requirements lightweight.
    *   **Tiers Breakdown:**
        *   **Presentation Tier:** React.js UI providing upload, progress tracking, and side-by-side data auditing.
        *   **Local Orchestration Tier:** FastAPI local Master backend managing CLAHE enhancements, overlap splitting, and result serialization.
        *   **Secure Tunneling:** cloudflared daemon establishing a zero-trust network bridge over HTTPS.
        *   **Deep Learning Inference Tier:** quantized GGUF model executed under C++ llama-cpp-python using remote dual NVIDIA T4 GPUs.
    *   **Visual Reference:**
        ![System Architecture](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/system_architecture_white.png)

---

## 📽️ Slide 9: Preprocessing Techniques (CLAHE)
*   **Slide Title:** Preprocessing Techniques: OpenCV & CLAHE
*   **Content:**
    *   **Grayscale Normalization:** Converts RGB inputs to standard single-channel grayscale arrays to stabilize pixel value distributions.
    *   **Tile-Based Adaptive Equalization:** Divides the document image into localized $M \times N$ tiles (typically $8 \times 8$) and applies histogram equalization to each tile independently.
    *   **Contrast Limit Clipping:** Clips the local contrast value above a threshold to prevent the amplification of paper crease noise and background shadows, rendering faded ink highly legible.

---

## 📽️ Slide 10: Inference Techniques (Quantization & SSE)
*   **Slide Title:** Inference Techniques: Quantization & SSE
*   **Content:**
    *   **IQ4 XS 4-Bit Quantization:** Quantizes weights of the 32B model using importance matrices. Shrinks VRAM requirements from ~70GB to ~19GB, allowing high-speed inference on affordable GPUs.
    *   **Server-Sent Events (SSE):** Establishes a persistent, unidirectional HTTP connection from FastAPI to the React client.
    *   **Chunk-by-Chunk Streaming:** Streams raw extraction tokens and ASCII layout grids to the user interface in real-time as they are generated by the model.

---

## 📽️ Slide 11: Networking Mechanisms (Secure Tunneling)
*   **Slide Title:** Networking Mechanisms: Cloudflare Tunnels
*   **Content:**
    *   **cloudflared Daemon:** Runs background processes on the local FastAPI Master Node and the remote GPU Worker Node.
    *   **Zero-Trust Edge Routing:** Outbound connections from both nodes connect to Cloudflare edge servers, establishing a virtual secure bridge.
    *   **NAT & Firewall Bypass:** Transmits serialized tensors securely without requiring inbound port-forwarding, public IPs, or firewall exceptions.

---

## 📽️ Slide 12: Fusion Mechanisms (Gated Cross-Attention)
*   **Slide Title:** Fusion Mechanisms: Gated Cross-Attention
*   **Content:**
    *   **Visual Tokenization:** Vision Transformer (ViT) encodes document images into grids of $14 \times 14$ patches, projecting visual layouts into spatial embeddings.
    *   **Decoupled Positional Embeddings:** Preserves native resolution dimensions to prevent layout scaling distortions.
    *   **Gated Fusion Layers:** Cross-attention gates align visual spatial features directly with language text tokens, mapping keys to values natively without OCR coordinates.

---

## 📽️ Slide 13: Algorithms (Dynamic Overlap Splitting)
*   **Slide Title:** Algorithms: Dynamic Overlap Splitting
*   **Content:**
    *   **Aspect Ratio Slicing:** Slices receipt images vertically when height exceeds width by a threshold factor (e.g., 1.5).
    *   **Offset Overlap Margin ($\delta$):** Computes a vertical overlap offset boundary between adjacent slices.
    *   **Boundary Protection:** Ensures text lines intersecting slice boundaries are captured in their entirety by at least one chunk, preventing character cropping.

---

## 📽️ Slide 14: Algorithms (Deduplication & Stitching)
*   **Slide Title:** Algorithms: Deduplication & Stitching
*   **Content:**
    *   **Fuzzy Template Matching:** Matches extracted keys across multiple receipt slices against standard financial templates using fuzzy alias indexing.
    *   **Coordinate Stitching:** Re-projects bounding coordinates of extracted items into the global document coordinate space.
    *   **Line-Item Deduplication:** Identifies and merges duplicate entries falling inside the overlap margin ($\delta$), compiling a clean, consolidated JSON payload.

---

## 📽️ Slide 15: System Modules - Part 1: Front-end Upload & Image Preprocessing
*   **Slide Title:** System Modules: Front-end & Image Preprocessing
*   **Content:**
    *   **Module 1: User Interface & Document Upload:**
        *   Developed using React.js, implementing WebRTC in-app camera controls.
        *   Offscreen canvas rendering to prevent mobile memory crashes.
        *   Viewport-locked drag cropper to optimize VLM extraction boundaries.
    *   **Module 2: Grayscale & OpenCV Preprocessing:**
        *   Invokes OpenCV library to convert binary RGB images to grayscale.
        *   Analyzes width, height, and aspect ratio metrics for receipt layout checks.
    *   **Module 3: OpenCV CLAHE Enhancement:**
        *   Divides the image into localized adaptive histogram tiles.
        *   Enhances local contrast parameters to restore faded text elements.
*   **References:** [UploadZone.jsx](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/frontend/src/components/UploadZone.jsx), [image_enhancer.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/utils/image_enhancer.py)

---

## 📽️ Slide 16: System Modules - Part 2: Splitting & Remote Inference
*   **Slide Title:** System Modules: Overlap Splitting & Inference
*   **Content:**
    *   **Module 4: Dynamic Document Splitting:**
        *   Identifies tall receipt strips (height-to-width ratio > 1.5).
        *   Splits receipts vertically into sequential chunks.
        *   Applies a standard margin overlap offset ($\delta$) to secure textual continuity.
    *   **Module 5: FastAPI Master-Worker Tunnel Bridge:**
        *   Connects local Master FastAPI server to the remote Kaggle worker.
        *   Serializes prompt templates and base64-encoded image chunks.
        *   Manages Server-Sent Events (SSE) keep-alive pings (15s intervals).
    *   **Module 6: quantized Remote Inference (llama-cpp-python):**
        *   Loads the Qwen2.5-VL-32B model inside a C++ CUDA runtime.
        *   Executes high-speed tensor arithmetic via IQ4 XS 4-bit quantization.
*   **References:** [gguf_engine.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/vlm/gguf_engine.py), [vlm_model.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/vlm/vlm_model.py)

---

## 📽️ Slide 17: System Modules - Part 3: Stitching, Validation & Export
*   **Slide Title:** System Modules: Stitching, DB & Exporters
*   **Content:**
    *   **Module 7: Chain-of-Thought (CoT) Prompting:**
        *   Formulates structured visual system schemas.
        *   Instructs VLM to output native Indic script transcriptions and translated English JSON in one pass.
    *   **Module 8: Coordinate Stitching & Line-Item Deduplication:**
        *   Stitches multi-slice JSON coordinates back to global canvas scale.
        *   Fuzzy alias scoring using RapidFuzz library to merge duplicate records.
    *   **Module 9: MongoDB Vault & Export Formatter:**
        *   Secures query isolation via JWT auth tokens linked to user emails.
        *   Automatic guest-directory file removal on tab closure.
        *   Compiles the extraction data into structured Microsoft Excel (.xlsx) spreadsheets and plain text (.txt) reports.
*   **References:** [layout_template.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/utils/layout_template.py), [export.py](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/backend/utils/export.py)

---

## 📽️ Slide 18: UML Diagrams
*   **Slide Title:** UML Diagrams: Sequence, Activity & Use Case
*   **Content:**
    *   **UML Sequence Diagram (Fig. 5.3):** Chronicles chronological messaging flow from UI upload request -> Master pre-processing -> Cloudflare Tunnel bridge -> Kaggle remote quantized GPU processing -> structured text returns.
    *   **UML User Activity Diagram (Fig. 5.2):** Details parallel workflows using swimlanes comparing React touch cropping, FastAPI local aspect splits, and remote VLM spatial attention decoding.
    *   **UML Use Case Diagram (Fig. 5.5):** Highlights actor mappings between the Auditing Candidate User and System Actors (FastAPI Local Automator, Qwen-VL Remote Brain).
    *   **Visual Reference:**
        ![UML Sequence Diagram](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/uml_sequence_diagram.png)

---

## 📽️ Slide 19: System Design: Block Diagram & DFD
*   **Slide Title:** System Design: Block Diagram & Dataflow (DFD)
*   **Content:**
    *   **System Block Diagram (Fig. 5.1):** Maps structural component links: React dashboard uploader -> OpenCV grayscale conversion -> Cloudflare tunnel serialization -> Llama.cpp worker GPU execution -> local master coordinate stitching -> SSE client rendering.
    *   **Dataflow Diagram (DFD) (Fig. 5.4):**
        *   *DFD Level 0 (Context):* Illustrates basic data interfaces between the User and the core extraction process.
        *   *DFD Level 1:* Charts internal processes including CLAHE enhancement, vertical slice splitting, VLM token projection, coordinate validation, and local schema checks.
    *   **Visual Reference:**
        ![System Block Diagram](file:///e:/Desktop/Antigravity/Final%20Sem%20Project%20Anti/Updated%20Final%20Year%20Project/block_diagram_final.png)

---

## 📽️ Slide 20: Result, Output & Screenshots - Part 1: Performance Metrics
*   **Slide Title:** Extraction Performance & Output Results
*   **Content:**
    *   **Heterogeneous Dataset Accuracy:**
        *   **96% Key-Value Accuracy** achieved on standard commercial invoices.
        *   **92% Key-Value Accuracy** achieved on degraded, low-contrast thermal receipts.
    *   **Processing Latency:** Average processing latency of **12-15 seconds** per document using IQ4 XS 4-bit quantization on remote dual NVIDIA T4 GPUs.
    *   **Layout Preservation Score:** High structural alignment in monospace ASCII output, enabling side-by-side human auditing.
    *   **Performance Ablation (Key-Value Accuracy Impact):**
        *   Baseline VLM without pre-processing: 87.5%
        *   VLM with CLAHE enhancement: 91.2%
        *   VLM with CLAHE + Dynamic Splitting: 96.0%

---

## 📽️ Slide 21: Result, Output & Screenshots - Part 2: Home & Upload Screen
*   **Slide Title:** System Dashboard: Home & Upload Screens
*   **Content:**
    *   **Home Screen Dashboard (Fig. B.1):**
        *   Clean web application interface developed using React.js.
        *   Provides "Use Camera" WebRTC interface and "Upload File" drag-and-drop zone.
        *   Displays historical database vault table from MongoDB linked to the user account.
    *   **Invoice Upload & Crop Screen (Fig. B.2):**
        *   Responsive touch-cropper utility for manual boundary selection.
        *   Limits VRAM memory footprint by executing client-side image downscaling.
    *   **Report Page Reference:** Fig.No:B.1 (Page 44) & Fig.No:B.2 (Page 45) of the project report.

---

## 📽️ Slide 22: Result, Output & Screenshots - Part 3: Processing & Output Screen
*   **Slide Title:** System Dashboard: Processing & Extraction Outputs
*   **Content:**
    *   **Processing Progress Panel (Fig. B.3):**
        *   Renders real-time status card indicators (Extracting Text, Layout Analysis, VLM Inference, Exporting Output).
        *   Powered by Server-Sent Events (SSE) keep-alive streaming from FastAPI.
    *   **Extraction Results Panel (Fig. B.4):**
        *   Displays extracted JSON results inside structured key-value tables.
        *   Renders native script output side-by-side with translated English outputs.
        *   Includes monospace ASCII "Digital Twin" preview panel and download buttons for Excel (.xlsx) and plain text (.txt) exports.
    *   **Report Page Reference:** Fig.No:B.3 (Page 46) & Fig.No:B.4 (Page 47) of the project report.

---

## 📽️ Slide 23: Future Scope & Enhancements (Expanded)
*   **Slide Title:** Future Scope & Enhancements
*   **Content:**
    *   **KV-Cache Pre-Warming & Prefetching:** Caches static prompt structures and layout schemas on GPU VRAM to reduce prompt-token parsing overhead and decrease latency by 30-40%.
    *   **Localized Indic Script Fine-Tuning:** Compiles an annotated Indic financial document dataset to fine-tune the visual encoder on handwritten merchant notes and regional cursive fonts.
    *   **Audit-Trail Confidence Scoring:** Implements token-level probability scoring to flag low-confidence JSON extractions (<85%) in red, routing them to human auditors for review.
    *   **ERP ERP Connectors:** Deploys integration hooks to post validated invoice details directly to enterprise database ledger backends (e.g., SAP, Oracle Financials).

---

## 📽️ Slide 24: Conclusion (Two Options)

### Option A (With Points):
*   **Slide Title:** Conclusion
*   **Content:**
    *   **Success:** Developed a highly precise, OCR-free document understanding system.
    *   **Core Model:** Unified visual-linguistic decoder Qwen2.5-VL-32B deployed directly at the pixel level.
    *   **Cost-Efficient:** Bypassed local computing constraints via Cloudflare Tunnels and remote T4 GPUs.
    *   **Performance:** Reached 96% Key-Value Accuracy on invoices and 92% on degraded thermal receipts.
    *   **Human Auditing:** Integrated spatial monospace ASCII "Digital Twin" grids for rapid human layout verification.

### Option B (Without Points):
*   **Slide Title:** Conclusion
*   **Content:**
    > *We have successfully built and validated a Multimodal Document Intelligence System (MDIS) that shifts document processing from legacy OCR-based pipelines to a unified, OCR-free pixel-level Vision-Language Model. By integrating the Qwen2.5-VL-32B model with local CLAHE image enhancement, dynamic overlap splitting, and secure Cloudflare Tunnel offloading, the system achieves 96% accuracy on invoices and 92% on degraded receipts. The inclusion of monospace ASCII Digital Twin representations alongside JSON schemas enables rapid human auditing, establishing a highly scalable, cost-effective automation solution for enterprise accounting workflows.*

---

## 📽️ Slide 25: List of References
*(In original non-alphabetical numbering sequence to align with inline citations)*
*   **Slide Title:** References
*   **Content:**
    1.  Dosovitskiy, A. et al. (2021) 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', Proc. ICLR.
    2.  Vaswani, A. et al. (2017) 'Attention is All you Need', Advances in NeurIPS.
    3.  Qwen Team (2023) 'Qwen-VL: A Versatile Vision-Language Model...', arXiv.
    4.  Xu, Y. et al. (2020) 'LayoutLM: Pre-training of Text and Layout...', Proc. SIGKDD.
    5.  Kim, G. et al. (2022) 'OCR-free Document Understanding Transformer (Donut)', Proc. ECCV.
    6.  Pizer, S. M. et al. (1987) 'Adaptive Histogram Equalization...', Vol. 39.
    7.  Liu, Y. et al. (2024) 'TextMonkey: An OCR-Free Large Multimodal Model...', arXiv.
    8.  Smith, R. (2007) 'An Overview of the Tesseract OCR Engine', Proc. ICDAR.
    9.  Huang, Y. et al. (2022) 'LayoutLMv3: Pre-training for Document AI...', Proc. ACM MM.
    10. Chen, Z. et al. (2023) 'InternVL: Scaling up Vision Foundation Models...', arXiv.
    11. Du, Y. et al. (2020) 'PP-OCR: A Practical Ultra Lightweight OCR System', arXiv.
    12. Tang, Z. et al. (2023) 'Unifying Vision, Text, and Layout (UDOP)', Proc. CVPR.
    13. Davis, B. et al. (2022) 'End-to-End Document Recognition (Dessurt)', Proc. ECCV Workshops.
    14. Lee, K. et al. (2023) 'Pix2Struct: Screenshot Parsing...', Proc. ICML.
    15. Wei, J. et al. (2022) 'Chain-of-Thought Prompting in LLMs', Advances in NeurIPS.
    16. Xu, Y. et al. (2021) 'LayoutLMv2: Multi-modal Pre-training...', Proc. ACL.
    17. Ye, Q. et al. (2023) 'mPLUG-DocOwl: Modularized Multimodal...', arXiv.
    18. Bhatia, V. (2021) 'OCR challenges in Indian scripts: A survey', ACM Computing Surveys.
    19. Liao, M. et al. (2020) 'Real-Time Scene Text Detection with Differentiable Binarization', Proc. AAAI.
    20. Cui, C. et al. (2025) 'PaddleOCR-VL: Boosting Multilingual Document Parsing...', arXiv.
