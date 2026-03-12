# Voice-to-Voice Pipeline

Real-time voice assistant on Jetson AGX Thor: speak a question, get a spoken answer.

```
Mic → [Whisper ASR ~1.2s] → [LLM ~9s] → [Kokoro TTS ~600ms] → Speaker
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
apt-get update -qq && apt-get install -y -qq ffmpeg espeak-ng
pip install openai-whisper transformers accelerate
pip install kokoro-onnx
pip install "numpy==1.26.4"       # kokoro-onnx upgrades numpy; pin it back
pip install silero-vad --no-deps  # --no-deps keeps CUDA torch 2.8.0 intact

# Run live conversation (VAD auto-detects speech — no button pressing)
python3 pipeline/voice_pipeline.py \
  --live \
  --fast-engine \
  --whisper-speculative \
  --mic hw:0,0 \
  --speaker plughw:1,3
```

Just speak — Silero VAD auto-detects when you start and stop talking.
Press **Ctrl+C** to exit. For push-to-talk mode, omit `--live` and press **Enter** each turn.

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
| Medium + small speculative | `--whisper-speculative` | ~1.2s | 23% faster, recommended |
| Medium TRT encoder | `--whisper-medium` | ~500ms | Requires Phase 3 build |

### LLM — Qwen3-VL-8B
| Mode | Flag | Latency | Notes |
|------|------|---------|-------|
| Original engine | *(default)* | ~10.5s | maxInputLen=1024, maxKV=4096 |
| Optimized engine | `--fast-engine` | ~9.2s | maxInputLen=256, maxKV=1024 |
| EAGLE3 (Qwen2.5-VL-7B) | `--eagle` | ~3-5s | Requires Phase 2 (x86 host first) |

### TTS — Kokoro-82M
- Engine: ONNX (`kokoro/model.onnx`, ~330MB), runs on CPU via onnxruntime
- Sample rate: **24 kHz**
- Latency: ~550ms short sentence, ~1100ms long sentence
- Real-time factor: ~0.05x — 20ms synthesis per 400ms of speech

#### Available Voices

| Voice | Gender | Accent | `--voice` flag |
|-------|--------|--------|----------------|
| `af_heart` | Female | US | `--voice af_heart` *(default)* |
| `af_bella` | Female | US | `--voice af_bella` |
| `af_nova` | Female | US | `--voice af_nova` |
| `af_sky` | Female | US | `--voice af_sky` |
| `am_adam` | Male | US | `--voice am_adam` |
| `am_echo` | Male | US | `--voice am_echo` |
| `bf_emma` | Female | British | `--voice bf_emma` |
| `bm_george` | Male | British | `--voice bm_george` |

Voice files (~500KB each) auto-download on first use to `kokoro/voices/`.

---

## Measured Latency (as of 2026-03-10)

| Config | ASR | LLM | TTS | Total |
|--------|-----|-----|-----|-------|
| Baseline | 1,695ms | 10,572ms | — | 12,373ms |
| `--fast-engine --whisper-speculative` | 1,214ms | 9,230ms | ~600ms | **~11,000ms** |
| + EAGLE3 (Phase 2, pending) | ~1,214ms | ~3,000ms | ~600ms | **~5,000ms** |

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

TTS:
  --voice NAME            Kokoro voice name (default: af_heart)
                          Female US: af_heart, af_bella, af_nova, af_sky
                          Male US:   am_adam, am_echo
                          Female GB: bf_emma   Male GB: bm_george
  --no-tts                Skip TTS, print text only

Audio:
  --mic DEVICE            ALSA capture device (default: hw:0,0)
  --speaker DEVICE        ALSA playback device (default: plughw:1,3)
  --record-seconds N      Recording duration per turn, push-to-talk mode (default: 5)
  --live                  Auto-detect speech with Silero VAD (no Enter needed)
  --silence-duration N    Seconds of silence to end utterance in --live mode (default: 1.5)
  --vad-threshold N       VAD speech probability threshold 0-1 (default: 0.5)

Output:
  --save-audio FILE       Save TTS audio to WAV file

Benchmark:
  --benchmark N           Run N timed iterations
  --warmup N              Warmup runs before benchmark (default: 1)
```

### Examples

```bash
# Live conversation — VAD auto-detects speech, no button pressing
python3 pipeline/voice_pipeline.py --live --whisper-speculative --fast-engine \
  --mic hw:0,0 --speaker plughw:1,3

# Try a British accent
python3 pipeline/voice_pipeline.py --live --whisper-speculative --fast-engine \
  --mic hw:0,0 --speaker plughw:1,3 --voice bf_emma

# Push-to-talk conversation, longer recording window
python3 pipeline/voice_pipeline.py --whisper-speculative --fast-engine --record-seconds 8

# Single audio file, save spoken response
python3 pipeline/voice_pipeline.py --audio test_audio.mp3 --save-audio out.wav

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

### Phase 4 — Kokoro TTS (DONE)
Natural-sounding neural TTS replacing Piper. Uses Kokoro-82M ONNX from HuggingFace.

**Install:**
```bash
apt-get install -y espeak-ng      # G2P fallback for rare words
pip install kokoro-onnx
pip install "numpy==1.26.4"       # pin numpy back — kokoro-onnx upgrades it
```

**Model files** auto-download to `kokoro/` on first run:
- `kokoro/model.onnx` — ~330MB, downloaded once
- `kokoro/voices/<name>.bin` — ~500KB per voice, downloaded on first use
- `kokoro/voices.npz` — combined voice index, rebuilt automatically

**kokoro-onnx 0.5.0 bugs fixed in code** (`_patch_kokoro_onnx()` in `voice_pipeline.py`):
- `speed` input dtype: `int32` → `float32` (model expects float)
- `style` input shape: `(256,)` → `(1, 256)` (model expects rank-2)
These patches are applied automatically at runtime — no manual edits needed.

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
├── phase5_fp8_kv.sh         # FP8 KV cache (optional)
├── test_phase1.py           # Validate Phase 1 speedup
└── test_phase2.py           # Validate EAGLE3 speedup

kokoro/                      # Auto-created on first TTS run
├── model.onnx               # Kokoro-82M model (~330MB)
├── voices.npz               # Combined voice index (auto-built)
└── voices/
    ├── af_heart.bin         # Default voice (~500KB)
    └── <name>.bin           # Additional voices, downloaded on demand
```
