# 👨‍💻 SRIGANTH GV (732722104056)

## 📌 Role: System Architect, Remote GPU & Integration Lead

Hello Sriganth, this folder contains all the files and modules that you have developed for the **Multimodal Document Intelligence System** project. 

### 📂 Assigned Source Files & Modules
Below are the files assigned to you along with their line counts:
1. **pipeline.py** (169 Lines) - Core processing pipeline execution logic.
2. **vlm_model.py** (353 Lines) - Vision Language Model integration and inference setup.
3. **gguf_engine.py** (270 Lines) - Execution engine for GGUF model formats.
4. **config.py** (148 Lines) - System-wide configuration variables.
5. **Dockerfile** (31 Lines) - Containerization instructions for deployment.
6. **main.py** - FastAPI heartbeats, SSE streaming, and VLM settings.

### 🧠 Core Concepts Handled
*   **Master-Worker Distributed Orchestration:** Cloudflare tunnel proxies settings integration, local backend connects to Kaggle GPU worker node APIs.
*   **SSE Streaming Keep-Alive Loops:** Terminate requests, prevent timeouts, SSE keep-alive heartbeats (`: ping\n\n`) packet headers control parameters loop.
*   **Model Request Thread Lock:** Semaphore wrapping multi-threads concurrency crash controls.
*   **VLM Prompt Layout Engineering:** System prompts designs, markdown tables, visual representation settings logic.

### 💡 Viva Defense Pointers (Enna sollanum?)
If external reviewers ask about your contribution, use these pointers:
> *"Sir, I designed the system architecture. I implemented a Master-Worker architecture that routes visual extraction tasks to remote GPU workers via secure API tunnels. I built the Server-Sent Events (SSE) stream endpoints with custom keep-alive heartbeat controls to prevent network timeouts, and designed the Visual Prompt Templates to constrain the VLM output into structured layout formats."*
