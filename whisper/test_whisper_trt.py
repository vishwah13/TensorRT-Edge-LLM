"""Whisper large-v3 inference: TensorRT encoder + PyTorch decoder.

Pipeline: audio file → mel spectrogram → TRT encoder → PyTorch decoder → text
The encoder is the bottleneck (32 transformer layers processing 3000 mel frames),
so TRT acceleration there gives the biggest win. The decoder runs natively on GPU.
"""

import time
import sys
import numpy as np
import torch
import tensorrt as trt
import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ENCODER_ENGINE = "/workspace/whisper/engines/encoder.engine"
PROCESSOR_DIR = "/workspace/whisper/onnx"
HF_MODEL_ID = "openai/whisper-large-v3"

# Whisper special tokens
SOT = 50258
LANG_EN = 50259
TRANSCRIBE = 50360
NO_TIMESTAMPS = 50364
EOT = 50257
MAX_TOKENS = 224


class TRTEncoder:
    """TensorRT engine wrapper for Whisper encoder."""

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("input_features", (1, 128, 3000))

    def __call__(self, mel_tensor):
        """mel_tensor: cuda float16 [1, 128, 3000] → [1, 1500, 1280]"""
        out_shape = self.context.get_tensor_shape("encoder_hidden_states")
        output = torch.empty(tuple(out_shape), dtype=torch.float16, device="cuda")
        self.context.set_tensor_address("input_features", mel_tensor.data_ptr())
        self.context.set_tensor_address("encoder_hidden_states", output.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        torch.cuda.synchronize()
        return output


def load_and_preprocess_audio(audio_path):
    """Load audio and compute mel spectrogram using whisper."""
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128)
    return mel.unsqueeze(0)  # [1, 128, 3000]


def greedy_decode(model, encoder_out, processor):
    """Autoregressive greedy decoding using HF model's decoder on GPU."""
    tokens = torch.tensor([[SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS]], device="cuda")
    encoder_outputs = (encoder_out,)  # keep float16 to match model dtype

    for _ in range(MAX_TOKENS):
        with torch.no_grad():
            out = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=tokens,
            )
        next_token_id = out.logits[0, -1, :].argmax().item()
        if next_token_id == EOT:
            break
        tokens = torch.cat([tokens, torch.tensor([[next_token_id]], device="cuda")], dim=1)

    generated = tokens[0, 4:].tolist()  # skip prompt tokens
    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return text, generated


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/test_audio.wav"
    print(f"Audio: {audio_path}")

    # Load models
    print("Loading TRT encoder...")
    encoder = TRTEncoder(ENCODER_ENGINE)

    print("Loading HF decoder (PyTorch, GPU)...")
    model = WhisperForConditionalGeneration.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float16, attn_implementation="eager"
    ).cuda().eval()

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(PROCESSOR_DIR)

    # Preprocess
    print("Processing audio...")
    mel = load_and_preprocess_audio(audio_path)
    mel_cuda = mel.cuda().half()

    # Warmup
    print("\nWarmup run...")
    enc_out = encoder(mel_cuda)
    _ = greedy_decode(model, enc_out, processor)

    # Timed run
    print("\n=== Timed inference ===")
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    t_enc_start = time.perf_counter()
    enc_out = encoder(mel_cuda)
    t_enc_end = time.perf_counter()

    t_dec_start = time.perf_counter()
    text, tokens = greedy_decode(model, enc_out, processor)
    t_dec_end = time.perf_counter()

    t_total = time.perf_counter() - t_start

    print(f"\nTranscription: {text}")
    print(f"\nTokens generated: {len(tokens)}")
    print(f"Encoder latency:  {(t_enc_end - t_enc_start) * 1000:.1f} ms")
    print(f"Decoder latency:  {(t_dec_end - t_dec_start) * 1000:.1f} ms")
    print(f"Total latency:    {t_total * 1000:.1f} ms")
    print(f"Real-time factor: {t_total / 30:.4f}x (30s audio)")


if __name__ == "__main__":
    main()
