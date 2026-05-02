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
from backend.config import LLAVA_GGUF_PATH, LLAVA_MMPROJ_PATH, VLM_LOCAL_MAX_NEW_TOKENS, VLM_LOCAL_N_CTX, INTERNAL_MODEL_API_KEY

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

    from backend.config import HF_MODELS_DIR
    if model_type == "minicpm":
        model_path = HF_MODELS_DIR / "MiniCPM-V-2_6-IQ2_M.gguf"
        mmproj_path = HF_MODELS_DIR / "mmproj-MiniCPM-V-2_6-f16.gguf"
        # Ensure context is large enough for MiniCPM
        ctx_size = VLM_LOCAL_N_CTX
    else:
        model_path = Path(LLAVA_GGUF_PATH)
        mmproj_path = Path(LLAVA_MMPROJ_PATH)
        ctx_size = VLM_LOCAL_N_CTX

    # If the process is already running with a DIFFERENT model, we must kill it first
    # We can track the current loaded model via a simple attribute on the client
    if _llama_client is not None:
        if getattr(_llama_client, "loaded_model", None) == model_type:
            return _llama_client
        else:
            logger.info(f"[gguf] Switching VLM from {_llama_client.loaded_model} to {model_type}. Killing old server...")
            release_vlm_memory()

    if not model_path or not model_path.exists():
        logger.error(f"[gguf] VLM model missing at {model_path}")
        return None

    server_exe = Path(__file__).parent / "llama_server_bin" / "llama-server.exe"
    if not server_exe.exists():
        logger.error(f"[gguf] llama-server.exe not found at {server_exe}.")
        return None

    port = 8080
    cmd = [
        str(server_exe),
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "-c", str(ctx_size),
        "-ngl", "99",
        "--port", str(port),
        "-t", "8",   # Use 8 CPU threads for layers that spill to system RAM
        "-cb"  # continuous batching
    ]
    
    if model_type == "qwen":
        cmd.extend(["--image-min-tokens", "1024"])

    logger.info(f"[gguf] Launching standalone CUDA server to force 100% GPU execution for model {model_type}...")
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

    # Wait for server to be ready (up to 60s)
    for _ in range(120):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            logger.info("[gguf] Standalone GPU Server is ONLINE.")
            _llama_client = StandaloneLlamaClient(port=port)
            _llama_client.loaded_model = model_type
            return _llama_client
        except Exception:
            pass

    logger.error("[gguf] Server failed to start within timeout.")
    return None


def query_local_llava(image_bytes: bytes, prompt: str, api_key: str = "", model_type="qwen") -> str:
    client = _load_gguf_model(model_type=model_type)
    if client is None:
        return "ERROR: Model not available"

    try:
        messages = []
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt}
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
