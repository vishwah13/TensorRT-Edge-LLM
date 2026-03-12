"""End-to-end pipeline benchmarking.

Runs the voice pipeline N times and reports p50/p95 latencies for each stage.

Usage:
    # Quick benchmark (3 warmup + 10 runs) with current setup
    python pipeline/benchmark.py --audio test_audio.mp3

    # Full benchmark with EAGLE3
    python pipeline/benchmark.py --audio test_audio.mp3 --eagle --runs 10 --warmup 3

    # Compare all configurations
    python pipeline/benchmark.py --audio test_audio.mp3 --compare-all
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    WORKSPACE, QWEN3_LLM_ENGINE, QWEN3_LLM_ENGINE_FAST, QWEN3_VISUAL_ENGINE,
    QWEN25_ENGINE_LLM, QWEN25_ENGINE_VISUAL,
    WHISPER_LARGE_ENGINE, WHISPER_MEDIUM_ENGINE,
)


def run_benchmark(pipeline, audio_path, image_path, runs, warmup):
    """Run pipeline benchmark and return aggregated results."""
    all_timings = []

    for i in range(warmup + runs):
        phase = "warmup" if i < warmup else f"run {i - warmup + 1}/{runs}"
        print(f"  [{phase}]", end=" ", flush=True)
        t = pipeline.process(audio_path, image_path)
        if i >= warmup:
            all_timings.append(t)
        total = t.get("total_ms", 0)
        print(f"total={total:.0f}ms")

    return all_timings


def summarize(timings, label=""):
    """Print summary statistics."""
    keys = ["asr_ms", "llm_ms", "tts_ms", "time_to_first_audio_ms", "total_ms"]

    print(f"\n{'='*70}")
    if label:
        print(f"  {label}")
        print(f"{'='*70}")
    print(f"  {'Metric':<35s} {'Avg':>8s} {'P50':>8s} {'P95':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    summary = {}
    for key in keys:
        vals = sorted([t[key] for t in timings if key in t])
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        p50 = vals[len(vals) // 2]
        p95 = vals[min(int(len(vals) * 0.95), len(vals) - 1)]
        mn, mx = vals[0], vals[-1]
        print(f"  {key:<35s} {avg:>7.0f}ms {p50:>7.0f}ms {p95:>7.0f}ms {mn:>7.0f}ms {mx:>7.0f}ms")
        summary[key] = {"avg": avg, "p50": p50, "p95": p95, "min": mn, "max": mx}

    print(f"{'='*70}")
    return summary


def compare_all(audio_path, image_path, runs, warmup):
    """Compare all available configurations."""
    from voice_pipeline import (
        WhisperASR, WhisperASRSpeculative, LLMInference, KokoroTTS, VoicePipeline,
    )

    # Dummy TTS for fair comparison (skip actual audio playback)
    class NoTTS:
        _available = False
        def synthesize_and_play(self, text):
            return 0

    configs = []

    # Config 1: Current baseline (Qwen3-VL + Whisper large-v3)
    if os.path.exists(QWEN3_LLM_ENGINE) and os.path.exists(WHISPER_LARGE_ENGINE):
        configs.append({
            "name": "Baseline (Qwen3-VL + Whisper-large-v3)",
            "asr_cls": lambda: WhisperASR(
                "openai/whisper-large-v3", WHISPER_LARGE_ENGINE,
                os.path.join(WORKSPACE, "whisper/onnx")
            ),
            "llm_cls": lambda: LLMInference(QWEN3_LLM_ENGINE, QWEN3_VISUAL_ENGINE),
        })

    # Config 2: Phase 1 (optimized engine)
    if os.path.exists(QWEN3_LLM_ENGINE_FAST):
        configs.append({
            "name": "Phase 1 (Qwen3-VL fast engine)",
            "asr_cls": lambda: WhisperASR(
                "openai/whisper-large-v3", WHISPER_LARGE_ENGINE,
                os.path.join(WORKSPACE, "whisper/onnx")
            ),
            "llm_cls": lambda: LLMInference(QWEN3_LLM_ENGINE_FAST, QWEN3_VISUAL_ENGINE),
        })

    # Config 3: Phase 2 (EAGLE3)
    if os.path.exists(QWEN25_ENGINE_LLM):
        configs.append({
            "name": "Phase 2 (Qwen2.5-VL EAGLE3)",
            "asr_cls": lambda: WhisperASR(
                "openai/whisper-large-v3", WHISPER_LARGE_ENGINE,
                os.path.join(WORKSPACE, "whisper/onnx")
            ),
            "llm_cls": lambda: LLMInference(
                QWEN25_ENGINE_LLM, QWEN25_ENGINE_VISUAL, eagle=True
            ),
        })

    # Config 4: Phase 2 + 3 (EAGLE3 + Whisper medium)
    if os.path.exists(QWEN25_ENGINE_LLM) and os.path.exists(WHISPER_MEDIUM_ENGINE):
        configs.append({
            "name": "Phase 2+3 (EAGLE3 + Whisper-medium TRT)",
            "asr_cls": lambda: WhisperASR(
                "openai/whisper-medium", WHISPER_MEDIUM_ENGINE,
                os.path.join(WORKSPACE, "whisper-medium/onnx")
            ),
            "llm_cls": lambda: LLMInference(
                QWEN25_ENGINE_LLM, QWEN25_ENGINE_VISUAL, eagle=True
            ),
        })

    if not configs:
        print("No configurations available. Build at least the baseline engines.")
        return

    all_summaries = {}
    for cfg in configs:
        print(f"\n\n>>> Benchmarking: {cfg['name']}")
        asr = cfg["asr_cls"]()
        llm = cfg["llm_cls"]()
        pipeline = VoicePipeline(asr, llm, NoTTS())

        timings = run_benchmark(pipeline, audio_path, image_path, runs, warmup)
        all_summaries[cfg["name"]] = summarize(timings, cfg["name"])

        # Free GPU memory between configs
        import torch
        del asr, llm, pipeline
        torch.cuda.empty_cache()

    # Final comparison table
    print(f"\n\n{'='*70}")
    print("COMPARISON SUMMARY (total_ms)")
    print(f"{'='*70}")
    for name, s in all_summaries.items():
        if "total_ms" in s:
            t = s["total_ms"]
            print(f"  {name:50s} avg={t['avg']:>7.0f}ms  p50={t['p50']:>7.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Pipeline benchmark")
    parser.add_argument("--audio", required=True, help="Audio file")
    parser.add_argument("--image", help="Optional image")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all available configurations")

    # Single-config options
    parser.add_argument("--eagle", action="store_true")
    parser.add_argument("--fast-engine", action="store_true")
    parser.add_argument("--whisper-medium", action="store_true")

    args = parser.parse_args()

    if args.compare_all:
        compare_all(args.audio, args.image, args.runs, args.warmup)
    else:
        from voice_pipeline import build_pipeline, VoicePipeline
        # Reuse voice_pipeline's build logic
        pipe_args = argparse.Namespace(
            whisper_medium=args.whisper_medium,
            whisper_speculative=False,
            eagle=args.eagle,
            fast_engine=args.fast_engine,
            max_tokens=128,
            voice="af_heart",
            no_tts=True,
        )
        pipeline = build_pipeline(pipe_args)
        timings = run_benchmark(pipeline, args.audio, args.image, args.runs, args.warmup)
        summarize(timings, "Benchmark Results")


if __name__ == "__main__":
    main()
