#!/usr/bin/env bash
# Phase 4: Install Piper TTS for edge text-to-speech.
#
# Piper is an ONNX-based TTS engine that runs fast on edge devices.
# ~100-200ms per sentence, many voice options.
#
# References:
#   https://github.com/rhasspy/piper
#   https://huggingface.co/rhasspy/piper-voices

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PIPER_DIR="${WORKSPACE_DIR}/piper"
VOICES_DIR="${PIPER_DIR}/voices"
VOICE_NAME="${1:-en_US-lessac-medium}"

echo "=============================================="
echo "Phase 4: Piper TTS Setup"
echo "=============================================="

mkdir -p "${PIPER_DIR}" "${VOICES_DIR}"

# Step 1: Install piper-tts
echo ""
echo "--- Step 1: Install piper-tts ---"
if command -v piper &>/dev/null; then
    echo "piper already installed: $(which piper)"
else
    pip install piper-tts
    echo "piper-tts installed."
fi

# Step 2: Download voice model
echo ""
echo "--- Step 2: Download voice: ${VOICE_NAME} ---"
VOICE_DIR="${VOICES_DIR}/${VOICE_NAME}"
if [ -d "${VOICE_DIR}" ] && ls "${VOICE_DIR}"/*.onnx &>/dev/null 2>&1; then
    echo "Voice already downloaded: ${VOICE_DIR}"
else
    mkdir -p "${VOICE_DIR}"
    cd "${VOICE_DIR}"

    # Download from HuggingFace piper-voices repo
    # URL structure: .../en/en_US/lessac/medium/en_US-lessac-medium.onnx
    VOICE_LANG_SHORT=$(echo "${VOICE_NAME}" | cut -d'_' -f1)        # en
    VOICE_LANG=$(echo "${VOICE_NAME}" | cut -d'-' -f1)              # en_US
    VOICE_QUALITY=$(echo "${VOICE_NAME}" | rev | cut -d'-' -f1 | rev) # medium
    VOICE_SPEAKER=$(echo "${VOICE_NAME}" | sed "s/^${VOICE_LANG}-//" | sed "s/-${VOICE_QUALITY}$//") # lessac

    BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    MODEL_URL="${BASE_URL}/${VOICE_LANG_SHORT}/${VOICE_LANG}/${VOICE_SPEAKER}/${VOICE_QUALITY}/${VOICE_NAME}.onnx"
    CONFIG_URL="${BASE_URL}/${VOICE_LANG_SHORT}/${VOICE_LANG}/${VOICE_SPEAKER}/${VOICE_QUALITY}/${VOICE_NAME}.onnx.json"

    echo "Downloading model..."
    wget -q "${MODEL_URL}" -O "${VOICE_NAME}.onnx" || {
        echo "Failed to download voice model. Check voice name."
        echo "Available voices: https://huggingface.co/rhasspy/piper-voices"
        exit 1
    }

    echo "Downloading config..."
    wget -q "${CONFIG_URL}" -O "${VOICE_NAME}.onnx.json" || true

    echo "Voice downloaded to: ${VOICE_DIR}"
    ls -lh "${VOICE_DIR}/"
fi

# Step 3: Quick test
echo ""
echo "--- Step 3: Quick TTS test ---"
TEST_WAV="${PIPER_DIR}/test_output.wav"
echo "Hello! This is a test of the Piper text to speech engine running on Jetson." | \
    piper --model "${VOICES_DIR}/${VOICE_NAME}/${VOICE_NAME}.onnx" \
          --output_file "${TEST_WAV}" 2>/dev/null || {
    # Fallback: try piper-tts Python API
    python3 -c "
from piper import PiperVoice
voice = PiperVoice.load('${VOICES_DIR}/${VOICE_NAME}/${VOICE_NAME}.onnx')
import wave
with wave.open('${TEST_WAV}', 'wb') as wav:
    voice.synthesize('Hello! This is a test of Piper TTS on Jetson.', wav)
print('TTS test audio written.')
" || echo "TTS test failed - may need manual setup."
}

if [ -f "${TEST_WAV}" ]; then
    echo "Test audio: ${TEST_WAV}"
    echo "Play with: aplay ${TEST_WAV}"
fi

echo ""
echo "=============================================="
echo "Piper TTS setup complete."
echo "  Voice: ${VOICE_NAME}"
echo "  Model: ${VOICES_DIR}/${VOICE_NAME}/${VOICE_NAME}.onnx"
echo "=============================================="
