# 🌊 Project Execution Flow

This document provides a simple, step-by-step explanation of how the **Multimodal Document Intelligence System** processes an invoice or receipt. It is designed to be easily understandable for presentation purposes and for new team members to quickly grasp the project's architecture.

---

## 🚀 Step-by-Step Flow

### Step 1: User Capture & Canvas Cropping (Frontend)
*   **Action:** The user captures a photo of an invoice using the **In-App Live Video Scanner** or selects an image/PDF file from their device.
*   **What Happens:** 
    *   If using the camera, the system streams live video to a viewfinder with visual guides, snapping a high-resolution canvas frame to prevent device OS memory crashes.
    *   The captured image is opened in a viewport-locked touch cropper canvas, allowing the user to select the document boundaries (removing background noise like table textures or unrelated text).
    *   The image is cropped and automatically downscaled to a max dimension of 1200px (to avoid model memory overhead) before sending it to the backend.

### Step 2: Local Pre-Processing (Backend)
*   **Action:** The **FastAPI Master Node** receives the image file.
*   **Image Enhancement:** The system applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to optimize contrast and enhance readability of faded text on receipt documents.
*   **Dynamic Splitting:** If the document is exceptionally tall, the system partitions it into overlapping chunks to bypass token length limitations and prevent VLM decoding failures.

### Step 3: Tunnel Handshake & Heartbeat Streaming
*   **Action:** The Master Node initiates the extraction request to the remote model.
*   **What Happens:** 
    *   The Master Node posts the image to the remote worker node through a secure **Cloudflare Tunnel**.
    *   While the heavy VLM inference runs on the GPU, the FastAPI server streams keep-alive heartbeat pings (`: ping\n\n`) over Server-Sent Events (SSE) every 15 seconds to prevent the tunnel from timing out.

### Step 4: AI Visual Extraction (Kaggle GPU Worker)
*   **Action:** The remote worker node, running on **Kaggle** utilizing powerful **Dual T4 GPUs**, receives the payload.
*   **Processing:** The **Qwen2.5-VL-32B** model processes the image:
    *   **Spatial OCR**: Extracts text from visual layout coordinates, reading native scripts natively (Tamil, Hindi, Marathi, etc.).
    *   **Translation & Transliteration**: Extracts native language values and generates a corresponding English translation block (`english_json` + `english_layout_text`).
    *   **Digital Twin**: Generates a layout-preserving plaintext grid recreating the spatial positioning of columns, header sections, and totals.

### Step 5: Translation Merging & Export Generation (Backend)
*   **Action:** The master node receives the raw extraction and generates report files.
*   **What Happens:**
    *   **Fallback-Safe Merge**: The backend merges `english_json` over the native fields key-by-key. Any untranslated fields fall back to native values, ensuring zero data loss.
    *   **Export Creation**: Automatically compiles Excel sheets (`.xlsx`), notepad reports (`.txt`), raw JSON dumps, and Digital Twin files.
    *   **MongoDB Persistence**: Saves the full analysis profile, including fields, logs, and export urls, to MongoDB.

### Step 6: Result Verification (Frontend)
*   **Action:** The React UI receives the completed extraction stream.
*   **What Happens:** The user reviews the standardized layout template (fully mapped to English), toggles the Spatial Twin tab, and downloads any generated formats.

---

## 💡 Why This Architecture? (Quick Summary)
- **Hardware Limitations:** By splitting the workload into a **Master (Local)** and **Worker (Kaggle)** node, we successfully run an enterprise-grade AI model without needing expensive local hardware.
- **End-to-End Visual OCR:** Instead of using error-prone traditional OCR that loses formatting, our Vision-Language Model reads the image *visually*, inherently understanding columns, rows, and spatial hierarchies.
- **Multilingual Support:** Translates vernacular Indian invoices seamlessly to structured English data for enterprise ERP systems.
