"""
backend/vlm/gguf_engine.py
Inference engine for GGUF models using standalone llama-server (GPU CUDA execution).
"""
import os
import subprocess
import time
import json
import urllib.request
import urllib.error
import logging
import base64
from pathlib import Path
from backend.config import (
    LLAVA_GGUF_PATH, LLAVA_MMPROJ_PATH,
    MINICPM_GGUF_PATH, MINICPM_MMPROJ_PATH,
    VLM_LOCAL_MAX_NEW_TOKENS, VLM_LOCAL_N_CTX, INTERNAL_MODEL_API_KEY, KAGGLE_VLM_URL
)

logger = logging.getLogger(__name__)

# Global state
_llama_process = None
_llama_client = None

class StandaloneLlamaClient:
    def __init__(self, port=8080):
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def create_chat_completion(self, messages, max_tokens=512, temperature=0.0, frequency_penalty=0.0, presence_penalty=0.0):
        payload = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }).encode('utf-8')
        
        req = urllib.request.Request(self.url, data=payload, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            logger.error(f"[gguf] Inference request failed: HTTP {e.code} - {error_body}")
            return {"choices": [{"message": {"content": f"ERROR: HTTP Error {e.code}: {error_body}"}}]}
        except Exception as e:
            logger.error(f"[gguf] Inference request failed: {e}")
            return {"choices": [{"message": {"content": f"ERROR: {str(e)}"}}]}

def _load_gguf_model(model_type="qwen"):
    """Starts the standalone GPU CUDA server and returns an API client."""
    global _llama_process, _llama_client

    # ── Model & port routing ──────────────────────────────────────────────────
    # Qwen = English/Latin → port 8080
    # MiniCPM = Indic languages → port 8081
    if model_type == "minicpm":
        model_path = MINICPM_GGUF_PATH
        mmproj_path = MINICPM_MMPROJ_PATH
        port = 8081
    else:
        model_path = LLAVA_GGUF_PATH
        mmproj_path = LLAVA_MMPROJ_PATH
        port = 8080

    # If already running the same model, reuse it
    if _llama_client is not None:
        if getattr(_llama_client, "loaded_model", None) == model_type:
            return _llama_client
        else:
            logger.info(f"[gguf] Switching VLM from {_llama_client.loaded_model} to {model_type}. Killing old server...")
            release_vlm_memory()

    if not model_path or not Path(model_path).exists():
        logger.error(f"[gguf] VLM model missing at {model_path}")
        return None

    server_exe = Path(__file__).parent / "llama_server_bin" / "llama-server.exe"
    if not server_exe.exists():
        logger.error(f"[gguf] llama-server.exe not found at {server_exe}.")
        return None

    cmd = [
        str(server_exe),
        "-m",        str(model_path),
        "--mmproj", str(mmproj_path),
        "-c",        str(VLM_LOCAL_N_CTX),
        # ── GPU acceleration flags ────────────────────────────────────────────
        "-ngl",      "99",         # Offload ALL layers to GPU (VRAM first, spill to shared)
        "--split-mode", "row",    # Row-split: uses GPU VRAM + shared GPU memory (iGPU/dGPU)
        "-t",        "8",          # 8 CPU threads for host-side work
        "--port",    str(port),
        "-cb",                     # Continuous batching for throughput
        "-np",       "1",          # Single parallel slot (we run one job at a time)
    ]

    # Model-specific tuning
    if model_type == "qwen":
        cmd.extend(["--image-min-tokens", "1024"])
    elif model_type == "minicpm":
        # MiniCPM-V needs larger image token budget for 4-bit model
        cmd.extend(["--image-max-tokens", "2048"])

    logger.info(f"[gguf] Launching {model_type.upper()} ({Path(model_path).name}) on port {port} with full GPU offload...")
    _project_root = Path(__file__).resolve().parent.parent.parent
    log_path = _project_root / "scratch" / "llama_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        _llama_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )

    # Wait for server to be ready (up to 300s)
    for _ in range(600):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            logger.info(f"[gguf] {model_type.upper()} Server ONLINE on port {port}.")
            _llama_client = StandaloneLlamaClient(port=port)
            _llama_client.loaded_model = model_type
            return _llama_client
        except Exception:
            pass

    logger.error(f"[gguf] {model_type.upper()} server failed to start within timeout.")
    return None

def query_local_llava(image_bytes: bytes, prompt: str, api_key: str = "", model_type: str = "qwen") -> str:
    """Central entry point for VLM inference (Routes to Kaggle if URL is set, else Local)."""
    # ── REMOTE KAGGLE VLM ROUTING (ASYNCHRONOUS POLLING) ──────────────────────
    if KAGGLE_VLM_URL and KAGGLE_VLM_URL.strip():
        base_url = KAGGLE_VLM_URL.rstrip('/')
        logger.info("[gguf] Starting async extraction on Kaggle: %s", base_url)
        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            payload = json.dumps({
                "image_base64": b64,
                "prompt": prompt
            }).encode('utf-8')
            
            # 1. Start the job (Increased timeout to 300s for large image uploads)
            req = urllib.request.Request(f"{base_url}/extract", data=payload, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as response:
                job_id = json.loads(response.read().decode()).get("job_id")
            
            if not job_id:
                return "ERROR: Failed to start remote job"

            # 2. Poll for the result
            logger.info("[gguf] Job started (ID: %s). Polling for result...", job_id)
            max_retries = 240 # Total 20 minutes (5s * 240)
            for i in range(max_retries):
                time.sleep(5) # Wait 5 seconds
                try:
                    status_req = urllib.request.Request(f"{base_url}/status/{job_id}")
                    with urllib.request.urlopen(status_req, timeout=20) as status_res:
                        res_data = json.loads(status_res.read().decode())
                        
                        if res_data.get("status") == "completed":
                            logger.info("[gguf] Extraction Complete!")
                            return res_data.get("data")
                        elif res_data.get("status") == "error":
                            return f"ERROR: Remote VLM failed: {res_data.get('message')}"
                        
                        logger.info("[gguf] ...still processing (%ds)...", (i+1)*5)
                except Exception as poll_e:
                    logger.warning("[gguf] Polling attempt failed (retrying): %s", poll_e)
            
            return "ERROR: Remote VLM timed out after polling limit"
            
        except Exception as e:
            logger.error("[gguf] Remote Kaggle request failed: %s", e)
            return f"ERROR: Remote VLM failed: {e}"

    # ── LOCAL GGUF INFERENCE ──────────────────────────────────────────────────
    client = _load_gguf_model(model_type=model_type)
    if client is None:
        return "ERROR: Model not available"

    try:
        messages = []
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # MiniCPM-V requires explicit <image> tag in the text prompt to map the vision embeddings
            final_prompt = f"<image>\n{prompt}" if model_type == "minicpm" else prompt

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": final_prompt}
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response = client.create_chat_completion(
            messages=messages,
            max_tokens=VLM_LOCAL_MAX_NEW_TOKENS,
            temperature=0.1,
            frequency_penalty=0.5,
            presence_penalty=0.5
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("[gguf] query failed: %s", e)
        return f"ERROR: {e}"


def query_local_paddle_vl(image_bytes: bytes, prompt: str) -> str:
    return query_local_llava(image_bytes, prompt, INTERNAL_MODEL_API_KEY)


def release_vlm_memory():
    """Kill the llama-server process to free VRAM."""
    global _llama_process, _llama_client
    _llama_client = None
    if _llama_process is not None:
        try:
            _llama_process.terminate()
            _llama_process.wait(timeout=5)
            logger.info("[gguf] Standalone GPU Server killed. VRAM released.")
        except Exception as e:
            logger.warning("[gguf] Error killing server: %s", e)
        finally:
            _llama_process = None
