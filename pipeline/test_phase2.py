"""Phase 2 validation: Test Qwen2.5-VL-7B with EAGLE3 speculative decoding.

Compares EAGLE3 inference speed against non-EAGLE baseline (if available).
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    LLM_INFERENCE, QWEN25_ENGINE_LLM, QWEN25_ENGINE_VISUAL, WORKSPACE,
    DEFAULT_MAX_GENERATE_LENGTH, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
)

INPUT_FILE = os.path.join(WORKSPACE, "pipeline/phase2_input.json")
OUTPUT_FILE = os.path.join(WORKSPACE, "pipeline/phase2_output.json")
PROFILE_FILE = os.path.join(WORKSPACE, "pipeline/phase2_profile.json")


def create_input(prompt: str, max_tokens: int = DEFAULT_MAX_GENERATE_LENGTH):
    data = {
        "batch_size": 1,
        "temperature": DEFAULT_TEMPERATURE,
        "top_k": DEFAULT_TOP_K,
        "max_generate_length": max_tokens,
        "requests": [{
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Keep responses concise."
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


def run_inference(eagle: bool, warmup: int = 1, label: str = ""):
    cmd = [
        LLM_INFERENCE,
        "--engineDir", QWEN25_ENGINE_LLM,
        "--multimodalEngineDir", QWEN25_ENGINE_VISUAL,
        "--inputFile", INPUT_FILE,
        "--outputFile", OUTPUT_FILE,
        "--warmup", str(warmup),
        "--dumpProfile",
        "--profileOutputFile", PROFILE_FILE,
    ]
    if eagle:
        cmd += [
            "--eagle",
            "--eagleDraftTopK", "10",
            "--eagleDraftStep", "6",
        ]

    print(f"\n[{label}] Running inference (eagle={eagle}, warmup={warmup})...")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"[{label}] FAILED: {result.stderr[-500:]}")
        return None

    with open(OUTPUT_FILE) as f:
        output = json.load(f)
    text = output["responses"][0]["output_text"]

    # Try to read profile for detailed timing
    profile_info = ""
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE) as f:
            profile = json.load(f)
        profile_info = f" (profile available)"

    print(f"[{label}] Total: {elapsed*1000:.0f}ms{profile_info}")
    print(f"[{label}] Output ({len(text)} chars): {text[:200]}...")
    return {"elapsed": elapsed, "text": text, "output_len": len(text)}


def main():
    # Verify engines exist
    for d in [QWEN25_ENGINE_LLM, QWEN25_ENGINE_VISUAL]:
        if not os.path.exists(d):
            print(f"Engine not found: {d}")
            print("Run: bash pipeline/phase2_eagle_setup.sh device")
            sys.exit(1)

    prompt = "What causes rainbows to appear in the sky? Explain briefly."
    create_input(prompt)

    print("=" * 60)
    print("Phase 2: Qwen2.5-VL-7B EAGLE3 Benchmark")
    print("=" * 60)

    # Run with EAGLE
    eagle_result = run_inference(eagle=True, warmup=2, label="EAGLE3")

    # Run without EAGLE for comparison
    no_eagle_result = run_inference(eagle=False, warmup=2, label="NO-EAGLE")

    # Summary
    print(f"\n{'='*60}")
    print("Phase 2 Results:")
    if eagle_result:
        print(f"  EAGLE3:   {eagle_result['elapsed']*1000:.0f}ms")
    if no_eagle_result:
        print(f"  No EAGLE: {no_eagle_result['elapsed']*1000:.0f}ms")
    if eagle_result and no_eagle_result:
        speedup = no_eagle_result["elapsed"] / eagle_result["elapsed"]
        print(f"  Speedup:  {speedup:.2f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
