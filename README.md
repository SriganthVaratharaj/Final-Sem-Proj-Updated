# Invoice AI — Unified OCR + Layout + VLM System

A production-ready, mobile-first web application for automated invoice data extraction. This system combines multiple AI models to move from unstructured images/PDFs to structured, validated data with high precision.

## 🚀 The AI Pipeline

The extraction process follows a high-performance 3-stage sequential pipeline:

1.  **PaddleOCR (Local Engine)**: Performs multi-language text extraction (English, Tamil, Hindi, Telugu, Kannada) and provides precise X/Y spatial coordinates for every word.
2.  **LayoutLMv3 (Structural Analysis)**: Analyzes the document's spatial layout via Hugging Face Inference API to classify structural regions (e.g., Header, Table, Footer).
3.  **VLM - LLaVA/BLIP-2 (Semantic Extraction)**: A Vision-Language Model provides the final semantic understanding to extract specific fields like Vendor Name, Tax ID, Amounts, and Line Items.

---

## ✨ Key Features

-   **📱 Mobile-First Wizard**: Optimized 3-screen flow (Capture → Processing → Results) with native rear-camera support for direct scanning.
-   **📄 Multi-Format Support**: Handles standard images (`.jpg`, `.png`, etc.) and automatically converts `.pdf` documents to images for processing.
-   **🔐 Secure Authentication**: Email-based signup/login using JWT and bcrypt password hashing.
-   **👻 Guest Mode**: Allows one-off scans without an account. Temporary data is automatically purged from the server via a cleanup beacon when the session ends.
-   **⚡ Real-time Updates**: Uses Server-Sent Events (SSE) to stream live progress from the AI pipeline to the frontend.
-   **📊 Multi-Export**: Download results in Excel (`.xlsx`), JSON, or human-readable Text formatting.

---

## 🛠️ Project Structure

```text
Final-Sem-Proj-Updated/
├── backend/            # FastAPI Orchestration Engine
├── db/                 # Database logic & Permanent Exports
│   └── outputs/        # User-specific result folders
├── frontend/           # React + Vite + TailwindCSS SPA
│   ├── uploads/        # Temporary upload workspace
│   └── src/            # UI Components & Wizard Logic
├── .env                # Environment configuration
├── requirements.txt    # Python dependencies
└── requirements-ml.txt # Machine Learning specific packages
```

---

## ⚙️ Setup & Installation

### 1. Backend Setup (Python 3.10+)

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure Environment**: Create a `.env` file in the root (see Configuration section).
3.  **Run Server**:
    ```bash
    cd backend
    python main.py
    ```
    *The API will be available at `http://localhost:8000`.*

### 2. Frontend Setup (Node.js 18+)

1.  **Install Dependencies**:
    ```bash
    cd frontend
    npm install
    ```
2.  **Run Development Server**:
    ```bash
    npm run dev
    ```
    *Access the UI at `http://localhost:5173`.*

---

## 📋 Configuration (.env)

| Variable | Description |
| :--- | :--- |
| `HF_TOKEN` | Hugging Face Access Token (Inference API) |
| `MONGO_URI` | MongoDB Connection String |
| `MONGO_DB_NAME` | Target database name (default: `invoice_ai`) |
| `JWT_SECRET` | Secret key for auth token signing |
| `MAX_UPLOAD_SIZE_MB` | Maximum file size allowed (default: `10`) |

---

## 📊 Database
The system uses **MongoDB** to store:
-   **Users**: Email and hashed credentials.
-   **Invoice Results**: Metadata, OCR text, Layout maps, and VLM extracted fields (Logged-in users only).

---

## 🛡️ License & Disclaimer
This project was developed as a Final Year Semester Project. It uses cloud-based inference for heavy models and local execution for OCR. All temporary guest data is deleted automatically to ensure privacy.
