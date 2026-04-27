"""
backend/vlm/gguf_engine.py
Inference engine for GGUF models using llama-cpp-python.
"""
from __future__ import annotations
import logging, os, base64, io
from pathlib import Path
from backend.config import (
    LLAVA_GGUF_PATH, LLAVA_MMPROJ_PATH,
    VLM_LOCAL_MAX_NEW_TOKENS, VLM_LOCAL_N_CTX,
    INTERNAL_MODEL_API_KEY
)

logger = logging.getLogger(__name__)
_llava_model = None

def _load_gguf_model(model_path, mmproj_path):
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava16ChatHandler
        logger.info("[gguf] Loading model on CPU for safety...")
        chat_handler = Llava16ChatHandler(clip_model_path=str(mmproj_path))
        return Llama(model_path=str(model_path), chat_handler=chat_handler, n_ctx=VLM_LOCAL_N_CTX, n_gpu_layers=0, verbose=False)
    except: return None

def query_local_llava(image_bytes: bytes, prompt: str, api_key: str = None) -> str:
    if api_key != INTERNAL_MODEL_API_KEY: return 'UNAUTHORIZED'
    global _llava_model
    if _llava_model is None: _llava_model = _load_gguf_model(LLAVA_GGUF_PATH, LLAVA_MMPROJ_PATH)
    if _llava_model is None: return 'LOAD_FAILED'
    try:
        if not image_bytes: return prompt
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        msgs = [{'role':'system','content':'Assistant'}, {'role':'user','content':[{'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}}, {'type':'text','text':prompt}]}]
        res = _llava_model.create_chat_completion(messages=msgs, max_tokens=VLM_LOCAL_MAX_NEW_TOKENS, temperature=0.1)
        return res['choices'][0]['message']['content']
    except Exception as e: return str(e)

query_local_paddle_vl = query_local_llava
