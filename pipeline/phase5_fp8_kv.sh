#!/usr/bin/env bash
# Phase 5 (Optional): FP8 KV Cache for Qwen2.5-VL-7B
#
# Reduces KV cache memory by 50%, potentially faster prefill.
# CAUTION: Known accuracy issues with Qwen2.5 FP8 KV. Test before deploying.
#
# Usage:
#   # On x86 host:
#   bash pipeline/phase5_fp8_kv.sh host
#
#   # On Thor device:
#   bash pipeline/phase5_fp8_kv.sh device

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
HF_BASE="Qwen/Qwen2.5-VL-7B-Instruct"
MODELS_DIR="${WORKSPACE_DIR}/models"
ONNX_DIR="${WORKSPACE_DIR}/onnx_models/qwen2.5-vl-7b-nvfp4-fp8kv"
ENGINE_DIR="${WORKSPACE_DIR}/engines/qwen2.5-vl-7b-nvfp4-fp8kv"

step_host() {
    echo "=============================================="
    echo "Phase 5 (x86 Host): NVFP4 + FP8 KV Cache"
    echo "=============================================="
    cd "${WORKSPACE_DIR}"

    echo ""
    echo "--- Step 1: Quantize with FP8 KV cache ---"
    tensorrt-edgellm-quantize-llm \
        --model_dir "${HF_BASE}" \
        --quantization nvfp4 \
        --kv_cache_quantization fp8 \
        --output_dir "${MODELS_DIR}/qwen2.5-vl-7b-nvfp4-fp8kv"

    echo ""
    echo "--- Step 2: Export with FP8 KV cache flag ---"
    tensorrt-edgellm-export-llm \
        --model_dir "${MODELS_DIR}/qwen2.5-vl-7b-nvfp4-fp8kv" \
        --output_dir "${ONNX_DIR}" \
        --fp8_kv_cache

    echo ""
    echo "Transfer ONNX to device, then run: bash pipeline/phase5_fp8_kv.sh device"
}

step_device() {
    echo "=============================================="
    echo "Phase 5 (Thor Device): Build FP8 KV Engine"
    echo "=============================================="
    cd "${WORKSPACE_DIR}"

    mkdir -p "${ENGINE_DIR}"
    ./build/examples/llm/llm_build \
        --onnxDir "${ONNX_DIR}" \
        --engineDir "${ENGINE_DIR}" \
        --maxBatchSize 1 \
        --maxInputLen 256 \
        --maxKVCacheCapacity 1024

    echo "Engine built: ${ENGINE_DIR}"
}

case "${1:-}" in
    host)   step_host ;;
    device) step_device ;;
    *)
        echo "Usage: $0 {host|device}"
        exit 1
        ;;
esac
