# Voice-to-Voice Pipeline

Real-time voice assistant on Jetson AGX Thor: speak a question, get a spoken answer.

```
Mic → [Whisper ASR ~1.2s] → [LLM ~9s] → [Piper TTS ~300ms] → Speaker
```

**Tested on:** AGX Thor, JetPack 7.1, NGC pytorch:25.08-py3 container

---

## Quick Start

```bash
# Start container with audio devices
docker run -it --gpus all --runtime nvidia --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/snd \
  -v /home/indus_rd/Dev/Repos/TensorRT-Edge-LLM:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.08-py3 bash

# Inside container — install deps (once)
apt-get update -qq && apt-get install -y -qq ffmpeg alsa-utils
pip install openai-whisper transformers accelerate piper-tts

# Run live voice assistant
python3 pipeline/voice_pipeline.py \
  --fast-engine \
  --whisper-speculative \
  --mic hw:0,0 \
  --speaker plughw:1,3 \
  --record-seconds 5
```

Press **Enter** to record, speak, and the assistant replies through the speaker.

---

## Audio Devices (AGX Thor)

| Device | ALSA ID | Description |
|--------|---------|-------------|
| Mic | `hw:0,0` | Brio 100 USB webcam microphone |
| Speaker | `plughw:1,3` | HDMI output to monitor |

---

## Pipeline Components

### ASR — Whisper
| Mode | Flag | Latency | Notes |
|------|------|---------|-------|
| Large-v3 TRT encoder | *(default)* | ~1.7s | Highest accuracy |
| Medium + small speculative | `--whisper-speculative` | ~1.2s | 23% faster, same accuracy |
| Medium TRT encoder | `--whisper-medium` | ~500ms | Requires Phase 3 build |

### LLM — Qwen3-VL-8B
| Mode | Flag | Latency | Notes |
|------|------|---------|-------|
| Original engine | *(default)* | ~10.5s | maxInputLen=1024, maxKV=4096 |
| Optimized engine | `--fast-engine` | ~9.2s | maxInputLen=256, maxKV=1024 |
| EAGLE3 (Qwen2.5-VL-7B) | `--eagle` | ~3-5s | Requires Phase 2 (x86 host first) |

### TTS — Piper
- Voice: `en_US-lessac-medium` (61MB ONNX, 22050 Hz)
- Latency: ~230ms short sentence, ~330ms long sentence
- Real-time factor: 0.04x (40ms to generate 1s of speech)

---

## Measured Latency (as of 2026-03-10)

| Config | ASR | LLM | TTS | Total |
|--------|-----|-----|-----|-------|
| Baseline | 1,695ms | 10,572ms | — | 12,373ms |
| `--fast-engine --whisper-speculative` | 1,214ms | 9,230ms | 374ms | **12,056ms** |
| + EAGLE3 (Phase 2, pending) | ~1,214ms | ~3,000ms | ~374ms | **~5,000ms** |

---

## CLI Reference

```
python3 pipeline/voice_pipeline.py [OPTIONS]

Input:
  --audio FILE            Process a single audio file instead of live mic
  --image FILE            Optional image for visual context (VLM)

ASR:
  --whisper-speculative   Use Whisper medium + small (fastest, recommended)
  --whisper-medium        Use Whisper medium with TRT encoder (requires Phase 3)

LLM:
  --fast-engine           Use Phase 1 optimized engine (recommended)
  --eagle                 Use Qwen2.5-VL-7B with EAGLE3 (requires Phase 2)
  --max-tokens N          Max tokens to generate (default: 128)

Audio:
  --mic DEVICE            ALSA capture device (default: hw:0,0)
  --speaker DEVICE        ALSA playback device (default: plughw:1,3)
  --record-seconds N      Recording duration per turn (default: 5)

Output:
  --save-audio FILE       Save TTS audio to WAV file
  --no-tts                Skip TTS, print text only

Benchmark:
  --benchmark N           Run N timed iterations
  --warmup N              Warmup runs before benchmark (default: 1)
```

### Examples

```bash
# Single audio file, save spoken response
python3 pipeline/voice_pipeline.py --audio test_audio.mp3 --save-audio out.wav

# Live conversation, longer recording window
python3 pipeline/voice_pipeline.py --whisper-speculative --fast-engine --record-seconds 8

# Benchmark 10 runs
python3 pipeline/voice_pipeline.py --audio test_audio.mp3 --fast-engine \
  --whisper-speculative --no-tts --benchmark 10 --warmup 2

# Compare all configs
python3 pipeline/benchmark.py --audio test_audio.mp3 --compare-all
```

---

## Optimization Phases

### Phase 1 — Fast Engine (DONE)
Rebuild LLM engine with voice-optimized constraints:
```bash
bash pipeline/phase1_rebuild_engine.sh
```
Produces `engines/qwen3-vl-8b-nvfp4-llm-fast/`. Use with `--fast-engine`.

### Phase 2 — EAGLE3 Speculative Decoding (PENDING — needs x86 host)
Qwen2.5-VL-7B + EAGLE3 draft model. 2-3x decode speedup.
```bash
# On x86 host:
bash pipeline/phase2_eagle_setup.sh host

# Transfer onnx_models/qwen2.5-vl-7b-nvfp4/ to Thor, then:
bash pipeline/phase2_eagle_setup.sh device
```
Use with `--eagle` flag.

### Phase 3 — Whisper Medium (DONE, TRT engine optional)
Whisper medium + small speculative decoding in pure PyTorch — no engine build needed:
```bash
# Already works via --whisper-speculative flag

# Optional: build TRT encoder for medium (further speedup)
python3 pipeline/phase3_whisper_medium.py all
```

### Phase 4 — Piper TTS (DONE)
```bash
bash pipeline/phase4_tts_setup.sh          # install + download voice
```
Voice files: `piper/voices/en_US-lessac-medium/`

### Phase 5 — FP8 KV Cache (OPTIONAL)
```bash
bash pipeline/phase5_fp8_kv.sh host   # x86: quantize with FP8 KV
bash pipeline/phase5_fp8_kv.sh device # Thor: build engine
```
Caution: known accuracy issues with Qwen2.5 FP8 KV — test before deploying.

---

## File Layout

```
pipeline/
├── voice_pipeline.py        # Main orchestrator — run this
├── config.py                # All paths and constants
├── benchmark.py             # Latency benchmarking
├── phase1_rebuild_engine.sh # Rebuild LLM engine (optimized)
├── phase2_eagle_setup.sh    # Qwen2.5-VL-7B + EAGLE3 (host + device)
├── phase3_whisper_medium.py # Whisper medium export/build/test
├── phase4_tts_setup.sh      # Piper TTS install + voice download
├── phase5_fp8_kv.sh         # FP8 KV cache (optional)
├── test_phase1.py           # Validate Phase 1 speedup
└── test_phase2.py           # Validate EAGLE3 speedup
```
