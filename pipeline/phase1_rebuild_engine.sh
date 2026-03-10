#!/usr/bin/env bash
# Phase 1: Quick Wins - Rebuild LLM engine with optimized params for voice pipeline.
#
# Changes from default:
#   --maxInputLen 256       (was 1024 — voice prompts are short)
#   --maxKVCacheCapacity 1024 (was 4096 — cap total sequence length)
#   --maxBatchSize 1        (was 4 — single-user voice conversation)
#
# This reduces engine memory footprint and speeds up prefill.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
cd "${WORKSPACE_DIR}"

ONNX_DIR="${WORKSPACE_DIR}/onnx_models/qwen3-vl-8b-nvfp4-llm"
ENGINE_DIR="${WORKSPACE_DIR}/engines/qwen3-vl-8b-nvfp4-llm-fast"

echo "=============================================="
echo "Phase 1: Rebuilding LLM engine (optimized)"
echo "  ONNX:   ${ONNX_DIR}"
echo "  Engine: ${ENGINE_DIR}"
echo "  maxInputLen: 256"
echo "  maxKVCacheCapacity: 1024"
echo "  maxBatchSize: 1"
echo "=============================================="

mkdir -p "${ENGINE_DIR}"

./build/examples/llm/llm_build \
  --onnxDir "${ONNX_DIR}" \
  --engineDir "${ENGINE_DIR}" \
  --maxInputLen 256 \
  --maxKVCacheCapacity 1024 \
  --maxBatchSize 1

echo ""
echo "Engine built successfully: ${ENGINE_DIR}"
echo "Engine files:"
ls -lh "${ENGINE_DIR}/"

echo ""
echo "To test with optimized settings, run:"
echo "  python pipeline/test_phase1.py"
