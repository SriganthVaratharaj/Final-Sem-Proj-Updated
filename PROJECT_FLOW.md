# 🌊 Project Execution Flow

This document provides a simple, step-by-step explanation of how the **Multimodal Document Intelligence System** processes an invoice or receipt. It is designed to be easily understandable for presentation purposes and for new team members to quickly grasp the project's architecture.

---

## 🚀 Step-by-Step Flow

### Step 1: User Upload (Frontend)
- **Action:** The user uploads a scanned invoice or thermal receipt via the **React UI Dashboard**.
- **What Happens:** The image is sent over the local network to our backend API.

### Step 2: Local Pre-Processing (Backend)
- **Action:** The **FastAPI Master Node** (running locally on our 4GB VRAM machine) receives the image.
- **Image Enhancement:** The system applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to improve the visibility of faded text, especially on thermal receipts.
- **Dynamic Splitting:** If the system detects a very tall image (like a long grocery receipt), it splits it into smaller overlapping chunks to prevent the AI from crashing.

### Step 3: Offloading to the Cloud (The Tunnel)
- **Action:** Because a 32-Billion parameter AI model cannot run on our local 4GB VRAM machine, the backend creates a secure bridge.
- **What Happens:** The backend sends the enhanced image and our "Master Prompt" through a secure **Cloudflare Tunnel** to a remote worker node.

### Step 4: AI Extraction (Kaggle GPU Worker)
- **Action:** The remote worker node, hosted on **Kaggle** utilizing powerful **Dual T4 GPUs**, receives the payload.
- **Processing:** The **Qwen2.5-VL-32B** model processes the image:
  - It natively reads the text (acting as OCR) while understanding the spatial layout (e.g., this number is below the "Total" header).
  - It extracts the data simultaneously into a structured format (JSON) while maintaining the original language (e.g., Hindi, Tamil).
  - It translates the semantic structure into English.
  - It generates a "Digital Twin" — a text grid that perfectly mimics the visual layout of the original document.

### Step 5: Post-Processing & Stitching (Backend)
- **Action:** The Kaggle node sends the generated structured data back through the tunnel to our local FastAPI server.
- **What Happens:** If the image was split in Step 2, the backend intelligently stitches the extracted JSON results back together. It also parses the outputs to ensure they are ready for display.

### Step 6: Result Display (Frontend)
- **Action:** The React Dashboard receives the final processed data.
- **What Happens:** The team or user can view the extracted native language data, the English translation, and the spatial "Digital Twin" to visually verify accuracy against the original document.

---

## 💡 Why This Architecture? (Quick Summary)
- **Hardware Limitations:** By splitting the workload into a **Master (Local)** and **Worker (Kaggle)** node, we successfully run an enterprise-grade AI model without needing expensive local hardware.
- **End-to-End Visual OCR:** Instead of using error-prone traditional OCR that loses formatting, our Vision-Language Model reads the image *visually*, inherently understanding columns, rows, and spatial hierarchies.
- **Multilingual Support:** Translates vernacular Indian invoices seamlessly to structured English data for enterprise ERP systems.
