"""Shared configuration for the voice-to-voice pipeline."""

import os

# Base paths — auto-detect container vs host
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("WORKSPACE_DIR"):
    WORKSPACE = os.environ["WORKSPACE_DIR"]
elif os.path.exists("/workspace/build/examples/llm/llm_inference"):
    WORKSPACE = "/workspace"
else:
    WORKSPACE = REPO_ROOT

# Binaries
LLM_BUILD = os.path.join(WORKSPACE, "build/examples/llm/llm_build")
LLM_INFERENCE = os.path.join(WORKSPACE, "build/examples/llm/llm_inference")
VISUAL_BUILD = os.path.join(WORKSPACE, "build/examples/multimodal/visual_build")

# --- Qwen3-VL-8B (current, Phase 1 optimized) ---
QWEN3_LLM_ENGINE = os.path.join(WORKSPACE, "engines/qwen3-vl-8b-nvfp4-llm")
QWEN3_LLM_ENGINE_FAST = os.path.join(WORKSPACE, "engines/qwen3-vl-8b-nvfp4-llm-fast")
QWEN3_LLM_ONNX = os.path.join(WORKSPACE, "onnx_models/qwen3-vl-8b-nvfp4-llm")
QWEN3_VISUAL_ENGINE = os.path.join(WORKSPACE, "engines/qwen3-vl-8b-nvfp4-visual")

# --- Qwen2.5-VL-7B + EAGLE3 (Phase 2) ---
QWEN25_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
QWEN25_HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
EAGLE3_DRAFT_REPO = "Rayzl/qwen2.5-vl-7b-eagle3-sgl"

QWEN25_BASE_DIR = os.path.join(WORKSPACE, "models/qwen2.5-vl-7b")
QWEN25_QUANTIZED_BASE = os.path.join(WORKSPACE, "models/qwen2.5-vl-7b-nvfp4")
QWEN25_QUANTIZED_DRAFT = os.path.join(WORKSPACE, "models/qwen2.5-vl-7b-eagle3-draft")
QWEN25_EAGLE3_DRAFT_DIR = os.path.join(WORKSPACE, "models/qwen2.5-vl-7b-eagle3-sgl")

QWEN25_ONNX_BASE = os.path.join(WORKSPACE, "onnx_models/qwen2.5-vl-7b-nvfp4/base")
QWEN25_ONNX_DRAFT = os.path.join(WORKSPACE, "onnx_models/qwen2.5-vl-7b-nvfp4/draft")
QWEN25_ONNX_VISUAL = os.path.join(WORKSPACE, "onnx_models/qwen2.5-vl-7b-nvfp4/visual")

QWEN25_ENGINE_LLM = os.path.join(WORKSPACE, "engines/qwen2.5-vl-7b-nvfp4/llm")
QWEN25_ENGINE_VISUAL = os.path.join(WORKSPACE, "engines/qwen2.5-vl-7b-nvfp4/visual")

# --- Whisper (Phase 3) ---
WHISPER_LARGE_ID = "openai/whisper-large-v3"
WHISPER_MEDIUM_ID = "openai/whisper-medium"
WHISPER_SMALL_ID = "openai/whisper-small"

WHISPER_LARGE_ENGINE = os.path.join(WORKSPACE, "whisper/engines/encoder.engine")
WHISPER_LARGE_PROCESSOR = os.path.join(WORKSPACE, "whisper/onnx")

WHISPER_MEDIUM_DIR = os.path.join(WORKSPACE, "whisper-medium")
WHISPER_MEDIUM_ONNX = os.path.join(WORKSPACE, "whisper-medium/onnx")
WHISPER_MEDIUM_ENGINE = os.path.join(WORKSPACE, "whisper-medium/engines/encoder.engine")

# --- TTS (Phase 4) ---
KOKORO_DIR = os.path.join(WORKSPACE, "kokoro")
KOKORO_MODEL = os.path.join(WORKSPACE, "kokoro/model_fp16.onnx")
KOKORO_VOICES = os.path.join(WORKSPACE, "kokoro/voices-v1.0.bin")
KOKORO_DEFAULT_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24000

# --- Pipeline defaults ---
DEFAULT_MAX_GENERATE_LENGTH = 128
DEFAULT_MAX_INPUT_LEN = 256
DEFAULT_MAX_KV_CACHE = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 1

# Whisper special tokens
SOT = 50258
LANG_EN = 50259
TRANSCRIBE = 50360
NO_TIMESTAMPS = 50364
EOT = 50257
WHISPER_MAX_TOKENS = 224
