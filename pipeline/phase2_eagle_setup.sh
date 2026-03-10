#!/usr/bin/env bash
# Phase 2: Qwen2.5-VL-7B + EAGLE3 Speculative Decoding Setup
#
# This is a multi-step process that runs on the x86 host (quantize + export)
# and then on the Thor device (build engines).
#
# Usage:
#   # On x86 host:
#   bash pipeline/phase2_eagle_setup.sh host
#
#   # Transfer ONNX to device, then on Thor:
#   bash pipeline/phase2_eagle_setup.sh device
#
# Following Example 3 from docs/source/developer_guide/getting-started/examples.md

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
MODEL_NAME="Qwen2.5-VL-7B-Instruct"
HF_BASE="Qwen/Qwen2.5-VL-7B-Instruct"
EAGLE3_REPO="Rayzl/qwen2.5-vl-7b-eagle3-sgl"

# Directories
MODELS_DIR="${WORKSPACE_DIR}/models"
ONNX_DIR="${WORKSPACE_DIR}/onnx_models/qwen2.5-vl-7b-nvfp4"
ENGINE_DIR="${WORKSPACE_DIR}/engines/qwen2.5-vl-7b-nvfp4"

step_host() {
    echo "=============================================="
    echo "Phase 2 (x86 Host): Qwen2.5-VL-7B + EAGLE3"
    echo "=============================================="

    cd "${WORKSPACE_DIR}"

    # Step 1: Download EAGLE3 draft model
    echo ""
    echo "--- Step 1: Download EAGLE3 draft model ---"
    if [ ! -d "qwen2.5-vl-7b-eagle3-sgl" ]; then
        git clone "https://huggingface.co/${EAGLE3_REPO}"
        cd qwen2.5-vl-7b-eagle3-sgl && git lfs pull && cd ..
    else
        echo "Draft model already downloaded."
    fi

    # Step 2: Quantize base model to NVFP4
    echo ""
    echo "--- Step 2: Quantize base model (NVFP4) ---"
    mkdir -p "${MODELS_DIR}"
    tensorrt-edgellm-quantize-llm \
        --model_dir "${HF_BASE}" \
        --quantization nvfp4 \
        --output_dir "${MODELS_DIR}/qwen2.5-vl-7b-nvfp4"

    # Step 3: Export base model with EAGLE flag
    echo ""
    echo "--- Step 3: Export base model ONNX (EAGLE base) ---"
    mkdir -p "${ONNX_DIR}"
    tensorrt-edgellm-export-llm \
        --model_dir "${MODELS_DIR}/qwen2.5-vl-7b-nvfp4" \
        --output_dir "${ONNX_DIR}/base" \
        --is_eagle_base

    # Step 4: Quantize draft model
    echo ""
    echo "--- Step 4: Quantize draft model ---"
    tensorrt-edgellm-quantize-draft \
        --base_model_dir "${HF_BASE}" \
        --draft_model_dir "qwen2.5-vl-7b-eagle3-sgl" \
        --quantization nvfp4 \
        --output_dir "${MODELS_DIR}/qwen2.5-vl-7b-eagle3-draft"

    # Step 5: Export draft model
    echo ""
    echo "--- Step 5: Export draft model ONNX ---"
    tensorrt-edgellm-export-draft \
        --draft_model_dir "${MODELS_DIR}/qwen2.5-vl-7b-eagle3-draft" \
        --base_model_dir "${HF_BASE}" \
        --output_dir "${ONNX_DIR}/draft"

    # Step 6: Export visual encoder
    echo ""
    echo "--- Step 6: Export visual encoder ONNX ---"
    tensorrt-edgellm-export-visual \
        --model_dir "${HF_BASE}" \
        --output_dir "${ONNX_DIR}/visual"

    echo ""
    echo "=============================================="
    echo "Host steps complete. Transfer ONNX to device:"
    echo "  scp -r ${ONNX_DIR} <user>@<device>:${ONNX_DIR}"
    echo "Then run: bash pipeline/phase2_eagle_setup.sh device"
    echo "=============================================="
}

step_device() {
    echo "=============================================="
    echo "Phase 2 (Thor Device): Build EAGLE3 Engines"
    echo "=============================================="

    cd "${WORKSPACE_DIR}"

    # Verify ONNX files exist
    for dir in "${ONNX_DIR}/base" "${ONNX_DIR}/draft" "${ONNX_DIR}/visual"; do
        if [ ! -d "${dir}" ]; then
            echo "ERROR: ONNX directory not found: ${dir}"
            echo "Run host steps first, then transfer ONNX to device."
            exit 1
        fi
    done

    # Step 1: Build base EAGLE engine (optimized for voice)
    echo ""
    echo "--- Step 1: Build base EAGLE engine ---"
    mkdir -p "${ENGINE_DIR}/llm"
    ./build/examples/llm/llm_build \
        --onnxDir "${ONNX_DIR}/base" \
        --engineDir "${ENGINE_DIR}/llm" \
        --maxBatchSize 1 \
        --maxInputLen 256 \
        --maxKVCacheCapacity 1024 \
        --maxVerifyTreeSize 60 \
        --maxDraftTreeSize 60 \
        --eagleBase

    # Step 2: Build draft EAGLE engine
    echo ""
    echo "--- Step 2: Build draft EAGLE engine ---"
    ./build/examples/llm/llm_build \
        --onnxDir "${ONNX_DIR}/draft" \
        --engineDir "${ENGINE_DIR}/llm" \
        --maxBatchSize 1 \
        --maxInputLen 256 \
        --maxKVCacheCapacity 1024 \
        --maxVerifyTreeSize 60 \
        --maxDraftTreeSize 60 \
        --eagleDraft

    # Step 3: Build visual engine
    echo ""
    echo "--- Step 3: Build visual encoder engine ---"
    mkdir -p "${ENGINE_DIR}/visual"
    ./build/examples/multimodal/visual_build \
        --onnxDir "${ONNX_DIR}/visual" \
        --engineDir "${ENGINE_DIR}/visual" \
        --minImageTokens 128 \
        --maxImageTokens 1024 \
        --maxImageTokensPerImage 512

    echo ""
    echo "=============================================="
    echo "Engines built successfully!"
    echo "  LLM + Draft: ${ENGINE_DIR}/llm"
    echo "  Visual:      ${ENGINE_DIR}/visual"
    echo ""
    echo "Test with:"
    echo "  python pipeline/test_phase2.py"
    echo "=============================================="
}

case "${1:-}" in
    host)   step_host ;;
    device) step_device ;;
    *)
        echo "Usage: $0 {host|device}"
        echo "  host   - Run on x86 host (quantize + export)"
        echo "  device - Run on Thor device (build engines)"
        exit 1
        ;;
esac
