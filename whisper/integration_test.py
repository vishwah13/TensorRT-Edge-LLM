"""Integration test: Whisper ASR → Qwen3-VL-8B VLM.

Transcribes audio with Whisper (TRT encoder + PyTorch decoder),
then feeds the transcription as a text prompt to Qwen3-VL via
the TensorRT Edge-LLM inference engine.
"""

import json
import os
import subprocess
import sys
import time

import torch
import tensorrt as trt
import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ENCODER_ENGINE = "/workspace/whisper/engines/encoder.engine"
PROCESSOR_DIR = "/workspace/whisper/onnx"
HF_MODEL_ID = "openai/whisper-large-v3"
LLM_ENGINE_DIR = "/workspace/engines/qwen3-vl-8b-nvfp4-llm"
VLM_ENGINE_DIR = "/workspace/engines/qwen3-vl-8b-nvfp4-visual"
LLM_INFERENCE_BIN = "/workspace/build/examples/llm/llm_inference"

SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS, EOT = 50258, 50259, 50360, 50364, 50257
MAX_TOKENS = 224


class TRTEncoder:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("input_features", (1, 128, 3000))

    def __call__(self, mel_tensor):
        out_shape = self.context.get_tensor_shape("encoder_hidden_states")
        output = torch.empty(tuple(out_shape), dtype=torch.float16, device="cuda")
        self.context.set_tensor_address("input_features", mel_tensor.data_ptr())
        self.context.set_tensor_address("encoder_hidden_states", output.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return output


def whisper_transcribe(audio_path):
    """Transcribe audio using TRT encoder + PyTorch decoder."""
    print(f"[Whisper] Loading models...")
    encoder = TRTEncoder(ENCODER_ENGINE)
    model = WhisperForConditionalGeneration.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float16, attn_implementation="eager"
    ).cuda().eval()
    processor = WhisperProcessor.from_pretrained(PROCESSOR_DIR)

    print(f"[Whisper] Processing audio: {audio_path}")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).unsqueeze(0).cuda().half()

    t_start = time.perf_counter()
    enc_out = encoder(mel)

    tokens = torch.tensor([[SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS]], device="cuda")
    for _ in range(MAX_TOKENS):
        with torch.no_grad():
            out = model(encoder_outputs=(enc_out,), decoder_input_ids=tokens)
        next_id = out.logits[0, -1, :].argmax().item()
        if next_id == EOT:
            break
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device="cuda")], dim=1)

    text = processor.tokenizer.decode(tokens[0, 4:].tolist(), skip_special_tokens=True)
    elapsed = time.perf_counter() - t_start
    print(f"[Whisper] Transcription ({elapsed*1000:.0f}ms): {text}")
    return text


def qwen_respond(transcription, image_path=None):
    """Send transcription (and optionally image) to Qwen3-VL via TRT engine."""
    content = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    content.append({
        "type": "text",
        "text": f"The user said: \"{transcription}\"\n\nRespond helpfully to what they said."
    })

    input_data = {
        "requests": [{
            "messages": [{"role": "user", "content": content}]
        }]
    }

    input_file = "/workspace/whisper/integration_input.json"
    output_file = "/workspace/whisper/integration_output.json"

    with open(input_file, "w") as f:
        json.dump(input_data, f, indent=2)

    print(f"[Qwen3-VL] Running inference...")
    cmd = [LLM_INFERENCE_BIN, "--engineDir", LLM_ENGINE_DIR,
           "--multimodalEngineDir", VLM_ENGINE_DIR,
           "--inputFile", input_file, "--outputFile", output_file]

    t_start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t_start

    if result.returncode != 0:
        print(f"[Qwen3-VL] Error: {result.stderr[-500:]}")
        return None

    with open(output_file) as f:
        output = json.load(f)

    response = output["responses"][0]["output_text"]
    print(f"[Qwen3-VL] Response ({elapsed*1000:.0f}ms): {response}")
    return response


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/test_audio.mp3"
    image_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 60)
    print("Integration Test: Whisper ASR → Qwen3-VL-8B")
    print("=" * 60)

    # Step 1: Transcribe audio
    transcription = whisper_transcribe(audio_path)

    # Step 2: Feed to Qwen3-VL
    print()
    response = qwen_respond(transcription, image_path)

    print()
    print("=" * 60)
    print("Pipeline Summary:")
    print(f"  Audio input:    {audio_path}")
    if image_path:
        print(f"  Image input:    {image_path}")
    print(f"  Transcription:  {transcription}")
    print(f"  LLM Response:   {response}")
    print("=" * 60)


if __name__ == "__main__":
    main()
