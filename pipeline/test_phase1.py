"""Phase 1 validation: Test optimized Qwen3-VL engine with capped output.

Runs inference with max_generate_length=128, temperature=0, top_k=1
and compares latency against the original engine.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    LLM_INFERENCE, QWEN3_LLM_ENGINE, QWEN3_LLM_ENGINE_FAST,
    QWEN3_VISUAL_ENGINE, WORKSPACE,
    DEFAULT_MAX_GENERATE_LENGTH, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
)

INPUT_FILE = os.path.join(WORKSPACE, "pipeline/phase1_input.json")
OUTPUT_FILE = os.path.join(WORKSPACE, "pipeline/phase1_output.json")


def create_input(prompt: str, max_tokens: int = DEFAULT_MAX_GENERATE_LENGTH):
    """Create optimized input JSON for voice pipeline."""
    data = {
        "batch_size": 1,
        "temperature": DEFAULT_TEMPERATURE,
        "top_k": DEFAULT_TOP_K,
        "max_generate_length": max_tokens,
        "requests": [{
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Keep responses concise and conversational."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }]
    }
    os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)
    with open(INPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return INPUT_FILE


def run_inference(engine_dir: str, label: str):
    """Run inference and return latency + output text."""
    cmd = [
        LLM_INFERENCE,
        "--engineDir", engine_dir,
        "--multimodalEngineDir", QWEN3_VISUAL_ENGINE,
        "--inputFile", INPUT_FILE,
        "--outputFile", OUTPUT_FILE,
    ]
    print(f"\n[{label}] Running inference...")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"[{label}] FAILED: {result.stderr[-500:]}")
        return elapsed, None

    with open(OUTPUT_FILE) as f:
        output = json.load(f)

    text = output["responses"][0]["output_text"]
    print(f"[{label}] Latency: {elapsed*1000:.0f}ms")
    print(f"[{label}] Output: {text[:200]}...")
    return elapsed, text


def main():
    prompt = "What causes rainbows to appear in the sky?"
    create_input(prompt)

    results = {}

    # Test optimized engine (fast)
    if os.path.exists(QWEN3_LLM_ENGINE_FAST):
        elapsed, text = run_inference(QWEN3_LLM_ENGINE_FAST, "FAST")
        results["fast"] = elapsed
    else:
        print(f"\nOptimized engine not found: {QWEN3_LLM_ENGINE_FAST}")
        print("Run: bash pipeline/phase1_rebuild_engine.sh")

    # Test original engine for comparison
    if os.path.exists(QWEN3_LLM_ENGINE):
        elapsed, text = run_inference(QWEN3_LLM_ENGINE, "ORIGINAL")
        results["original"] = elapsed

    # Summary
    if len(results) == 2:
        speedup = results["original"] / results["fast"]
        savings = results["original"] - results["fast"]
        print(f"\n{'='*50}")
        print(f"Phase 1 Results:")
        print(f"  Original:  {results['original']*1000:.0f}ms")
        print(f"  Optimized: {results['fast']*1000:.0f}ms")
        print(f"  Speedup:   {speedup:.2f}x ({savings*1000:.0f}ms saved)")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
