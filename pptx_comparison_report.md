# 📊 PPTX Formatting & Content Alignment Report
This report provides a slide-by-slide comparison between your project presentation (**final review.pptx**) and the example presentation (**final review ECG.pptx**). It outlines exactly what content you need to add, modify, or format to align with the ECG template.

---

## 🛠️ Key Format Differences to Address

1. **Literature Survey Layout (CRITICAL):**
   * **Your Current PPT:** Paragraphs of text explaining models (Tesseract, LayoutLMv3, Qwen2-VL).
   * **ECG Template PPT:** A structured **Literature Survey Table** listing *Author & Year*, *Method Used*, *Features*, *Limitations*, and *Proposed System Advantage*.
   * **Action:** Replace the paragraphs with the structured tables provided below.

2. **Existing System Demerits:**
   * **Your Current PPT:** Heading is "Problems in Existing System".
   * **ECG Template PPT:** Heading is "Demerits" with a clear list.
   * **Action:** Rename the heading to **Demerits:**.

3. **Proposed System Merits:**
   * **Your Current PPT:** Just a paragraph explaining the Proposed System.
   * **ECG Template PPT:** Has a matching **Merits:** bullet list.
   * **Action:** Add the **Merits:** bullet list.

4. **Technologies Used & Techniques (Slide 10):**
   * **Your Current PPT:** Split across two slides of lists.
   * **ECG Template PPT:** Combined on a single slide with two columns: **Technologies Used** on the left, and **Techniques** on the right.
   * **Action:** Combine them into a single slide using the two-column structure.

5. **Algorithm Complexity & Formula (Slide 11):**
   * **Your Current PPT:** Standard descriptions of CLAHE and Splitting.
   * **ECG Template PPT:** Has a numbered **Steps:** list and a mathematical **Formula** for time complexity.
   * **Action:** Format the steps and add the complexity formula provided below.

6. **Screenshots Consolidation:**
   * **Your Current PPT:** 4 slides with one screenshot each.
   * **ECG Template PPT:** Combined figures on fewer slides (e.g. Figure 1,2 on Slide 17, Figure 3,4 on Slide 18).
   * **Action:** Combine your screenshots into 2 or 3 slides (e.g., Home/Upload on one, Processing/Output on another).

---

## 📝 Slide-by-Slide Content Mapping & Text Changes

### 1. Slide 1: Title Slide
* **ECG Format:**
  * Title (Large)
  * Presenters (Listed with numbers)
  * Under Guidance of (Guide Name)
  * HOD Name (optional)
  * Department details
* **Your Change:**
  Format your members list like this:
  > **Presented by:**
  > 1. Sriganth G V (732722104056)
  > 2. Jayasimbu J (732722104018)
  > 3. Aadhiseshan S (732722104001)
  > 4. Remo V (732722104047)
  >
  > **Under Guidance of:**
  > Dr. P. Nandhini, HOD/CSE

---

### 2. Slide 4 & 5: Literature Survey Tables (Replaces current paragraphs)
Create two slides containing tables with the following content:

#### Slide 4: Literature Survey - Table 1
| Author & Year | Method Used | Features | Limitations | Proposed System Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Du et al. (2020)** | PP-OCR / PaddleOCR | Multi-language character recognition using lightweight deep networks. | Loses spatial context; cannot link values (e.g., item name with price). | Integrated VLM visual reasoning to preserve spatial layout associations. |
| **Huang et al. (2022)** | LayoutLMv3 Model | Text, image, and visual position (bounding boxes) alignment. | Heavily dependent on separate OCR engine inputs; cumulative error risk. | Natively interprets document images directly, bypassing external OCR. |

#### Slide 5: Literature Survey - Table 2 (Continued)
| Author & Year | Method Used | Features | Limitations | Proposed System Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Bai et al. (2024)** | Qwen2-VL Model | Visual reasoning using varying resolutions in Vision Transformers (ViT). | High VRAM requirement (8B-32B params); cannot run locally on cheap hardware. | Implemented Master-Worker architecture to offload inference to remote GPUs. |
| **Hu et al. (2024)** | MiniCPM-V 2.6 | Multimodal visual LLM with efficient local quantization support. | CPU-intensive execution causes latency on low-power host machines. | Dynamic fallback execution on remote GPU tunnels using GGUF format. |

---

### 3. Slide 7: Existing System
* **Heading Change:** Change "Problems in Existing System" to **Demerits:**.
* **List Content:**
  * Demerits:
    * Lack of spatial layout understanding during text extraction.
    * Inability to process low-quality, faded, or handwritten receipts.
    * Poor scalability on low-power consumer-grade hardware.
    * High manual verification and correction cost in enterprise workflows.

---

### 4. Slide 8: Proposed System
* **Your Change:** Keep the description paragraph, and append the **Merits:** list:
  * Merits:
    * High-precision visual reasoning that matches human document understanding.
    * Preserves spatial relationships using "Digital Twin" JSON mapping.
    * Scalable Master-Worker setup that runs heavy models on consumer PCs.
    * Out-of-the-box support for 10+ Indic scripts and handwritten text.
    * Real-time SSE streaming for transparent execution feedback.

---

### 5. Slide 10: Technologies & Techniques (Two-Column Layout)
Replaces your current Slide 10 & 11. Create a single slide with two columns:

* **Left Column: Technologies Used:**
  * **Frontend:** React.js, TailwindCSS, Axios
  * **Backend:** FastAPI, Uvicorn ASGI, Pydantic
  * **Core AI/ML:** Qwen2.5-VL (GGUF), Llama.cpp (CUDA)
  * **Image Vision:** OpenCV (CLAHE), Pillow
  * **Database:** MongoDB Atlas, Motor Driver
  * **Tunnels:** Cloudflare Argo Tunnel
* **Right Column: Techniques:**
  * Contrast Limited Adaptive Histogram Equalization (CLAHE).
  * Dynamic height segment splitting with pixel overlaps.
  * Spatial Self-Attention visual reasoning.
  * 4-bit weights quantization and normalization.
  * Secure base64 API token serialization.
  * Server-Sent Events (SSE) stream heartbeats.

---

### 6. Slide 11: Algorithm
* **Your Change:** Format Slide 11 into a clear step-by-step list and add the **Formula**:
* **Steps:**
  1. Upload document image/PDF via the React interface.
  2. Preprocess using CLAHE to enhance faded characters and correct deskewing.
  3. Divide tall receipts vertically with overlapping parameters.
  4. Perform visual reasoning inference on the remote GPU worker.
  5. Parse raw output and stitch overlapping segments back into a unified schema.
  6. Save structured fields to MongoDB and export spreadsheets.
* **Formula:**
  $$T(R) = O(N \times P)$$
  Where:
  * $T(R)$ = Total processing execution time.
  * $N$ = Number of segmented image slices.
  * $P$ = Image resolution patches processed by the Vision Transformer.

---

### 7. Slide 12: Mechanism
* **Your Change:** Add a "Working Mechanism" heading and format the points:
  * Working Mechanism:
    * Local Master node handles acquisition, deskewing, and image enhancements.
    * Encrypted Argo Tunnels securely route base64 image data to the worker.
    * Remote Dual-T4 GPU worker runs the quantized Qwen2.5-VL engine.
    * Spatial self-attention interprets layout keys and values without templates.
    * Local Master runs fuzzy scoring to compile results into pandas DataFrames.
    * UI updates live status using SSE heartbeats until extraction completes.

---

### 8. Slide 13 & 14: Modules
Split your modules slide into two slides with bullet points:

#### Slide 13: System Modules (1 & 2)
* **Image Intelligence & Preprocessing Module:**
  * Document capture controls (HTML5 WebRTC camera & canvas).
  * CLAHE contrast restoration filters.
  * Vertical segment cropping and dynamic resolution adjustments.
* **Remote VLM Inference & Logic Module:**
  * Tunnel link handler and API endpoint updates.
  * CUDA-accelerated Llama.cpp subprocess initialization.
  * Prompts formatting and dynamic script reference injection.

#### Slide 14: System Modules (3 & 4)
* **Authentication, Security & Database Module:**
  * MongoDB schemas construction and collection indexing.
  * JWT auth tokens extraction and password bcrypt hashing.
  * Session lifecycle and automated guest uploads cleanup.
* **Standardization & Excel Export Module:**
  * Spatial "Digital Twin" bounding boxes overlay mapper.
  * Fuzzy matching parser using RapidFuzz score matching.
  * Pandas excel sheet layout formatter and direct CSV compiler.

---

### 9. Slide 17 & 18: Screenshots Combined
* **Slide 17 Title:** Results & Screenshots (Core App)
  * *Figure 1 & 2:* Home Screen & Upload Page (Render side-by-side)
* **Slide 18 Title:** Results & Screenshots (Processing & Outputs)
  * *Figure 3 & 4:* SSE Processing Panel & Extraction Details Cards (Render side-by-side)

---

## 🖼️ Project Report Image to PPT Slide Mapping
Use the following table to copy images directly from your **project report Copy4.pdf** (using their respective figure numbers and page locations) and insert them into your presentation slides in **final review.pptx**:

| Slide Number in `final review.pptx` | Slide Title | Corresponding Figure in `project report Copy4.pdf` | PDF Page Number |
| :--- | :--- | :--- | :--- |
| **Slide 9** | System Architecture | **Fig.No:5.6 Architecture diagram** | **Page 31** |
| **Slide 15** | Mechanism | **Fig.No:5.6 Architecture diagram** (or Block diagram) | **Page 31 / 26** |
| **Slide 17** | Block Diagram | **Fig.No:5.1 Block diagram** | **Page 26** |
| **Slide 18** | UML Diagram | **Fig.No:5.3 Sequence diagram** | **Page 28** |
| **Slide 19** | Result / Screenshot Figure 1 - Home Screen | **Fig.No:B.1 Home Screen** | **Page 44** |
| **Slide 20** | Result / Screenshot Figure 2 – Upload Screen | **Fig.No:B.2 Invoice Upload Screen** | **Page 45** |
| **Slide 21** | Result / Screenshot Figure 3 – Processing Screen | **Fig.No:B.3 Processing Screen** | **Page 46** |
| **Slide 22** | Result / Screenshot Figure 4 – Output Screen | **Fig.No:B.4  Output Screen** | **Page 47** |

*Note: If you want to include other diagrams in Slide 18's UML section, you can use **Fig.No:5.2 Activity diagram for user** (Page 27) and **Fig.No:5.4 Dataflow diagram** (Page 29) from the report.*
