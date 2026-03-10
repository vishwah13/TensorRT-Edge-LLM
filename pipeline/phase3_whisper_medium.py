"""Phase 3: Export Whisper medium encoder to ONNX and build TRT engine.

Whisper medium (769M params) is ~2x faster than large-v3 (1.5B) with
<1% WER accuracy loss. Combined with speculative decoding using
whisper-small as the assistant model, we target ~300-500ms total ASR.

Usage:
    python pipeline/phase3_whisper_medium.py export   # Export encoder ONNX
    python pipeline/phase3_whisper_medium.py build    # Build TRT engine
    python pipeline/phase3_whisper_medium.py test     # Test ASR accuracy + speed
    python pipeline/phase3_whisper_medium.py all      # Run all steps
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    WHISPER_MEDIUM_ID, WHISPER_SMALL_ID,
    WHISPER_MEDIUM_ONNX, WHISPER_MEDIUM_ENGINE, WORKSPACE,
    SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS, EOT, WHISPER_MAX_TOKENS,
)

WHISPER_MEDIUM_DIR = os.path.join(WORKSPACE, "whisper-medium")
ONNX_DIR = WHISPER_MEDIUM_ONNX
ENGINE_DIR = os.path.join(WHISPER_MEDIUM_DIR, "engines")


class EncoderWrapper(torch.nn.Module):
    """Wraps HF WhisperEncoder to output only last_hidden_state."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        return self.encoder(input_features).last_hidden_state


def step_export():
    """Export Whisper medium encoder to ONNX."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    os.makedirs(ONNX_DIR, exist_ok=True)

    print(f"Loading {WHISPER_MEDIUM_ID} with eager attention...")
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MEDIUM_ID,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).cuda().eval()

    # Save processor for tokenization
    processor = WhisperProcessor.from_pretrained(WHISPER_MEDIUM_ID)
    processor.save_pretrained(ONNX_DIR)
    print("Processor saved.")

    # Export encoder
    print("\nExporting encoder to ONNX...")
    encoder_wrapper = EncoderWrapper(model.model.encoder).cuda().eval()
    # Whisper medium uses 80 mel bins (not 128 like large-v3)
    n_mels = model.config.num_mel_bins
    dummy_mel = torch.randn(1, n_mels, 3000, dtype=torch.float16, device="cuda")

    encoder_path = os.path.join(ONNX_DIR, "encoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper,
            (dummy_mel,),
            encoder_path,
            input_names=["input_features"],
            output_names=["encoder_hidden_states"],
            dynamic_axes={
                "input_features": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(encoder_path) / 1e6
    print(f"Encoder exported: {encoder_path} ({size_mb:.0f} MB)")
    print(f"  n_mels={n_mels}, encoder_dim={model.config.d_model}")


def step_build():
    """Build TRT engine from ONNX encoder using trtexec."""
    import subprocess

    onnx_path = os.path.join(ONNX_DIR, "encoder.onnx")
    if not os.path.exists(onnx_path):
        print(f"ONNX not found: {onnx_path}")
        print("Run: python pipeline/phase3_whisper_medium.py export")
        sys.exit(1)

    os.makedirs(ENGINE_DIR, exist_ok=True)
    engine_path = os.path.join(ENGINE_DIR, "encoder.engine")

    print(f"Building TRT engine: {engine_path}")
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--minShapes=input_features:1x80x3000",
        "--optShapes=input_features:1x80x3000",
        "--maxShapes=input_features:1x80x3000",
        "--workspace=2048",
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"trtexec failed:\n{result.stderr[-1000:]}")
        sys.exit(1)

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"Engine built: {engine_path} ({size_mb:.0f} MB)")


def step_test():
    """Test Whisper medium with speculative decoding (small as assistant).

    Uses HuggingFace generate() with assistant_model for 2x speedup.
    Falls back to greedy decoding if assistant model fails.
    """
    import tensorrt as trt
    import whisper
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    audio_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORKSPACE, "test_audio.mp3")

    # --- Method 1: TRT encoder + PyTorch decoder (greedy) ---
    print("=== Method 1: TRT encoder + PyTorch decoder (greedy) ===")

    engine_path = os.path.join(ENGINE_DIR, "encoder.engine")
    if os.path.exists(engine_path):
        # Load TRT encoder
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MEDIUM_ID, torch_dtype=torch.float16, attn_implementation="eager"
        ).cuda().eval()

        n_mels = model.config.num_mel_bins
        context.set_input_shape("input_features", (1, n_mels, 3000))

        processor = WhisperProcessor.from_pretrained(ONNX_DIR)

        # Prepare audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).unsqueeze(0).cuda().half()

        # Warmup
        out_shape = context.get_tensor_shape("encoder_hidden_states")
        enc_output = torch.empty(tuple(out_shape), dtype=torch.float16, device="cuda")
        context.set_tensor_address("input_features", mel.data_ptr())
        context.set_tensor_address("encoder_hidden_states", enc_output.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        # Timed run
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # TRT encoder
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        t_enc = time.perf_counter()

        # PyTorch greedy decoder
        tokens = torch.tensor([[SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS]], device="cuda")
        for _ in range(WHISPER_MAX_TOKENS):
            with torch.no_grad():
                out = model(encoder_outputs=(enc_output,), decoder_input_ids=tokens)
            next_id = out.logits[0, -1, :].argmax().item()
            if next_id == EOT:
                break
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device="cuda")], dim=1)
        t_dec = time.perf_counter()

        text1 = processor.tokenizer.decode(tokens[0, 4:].tolist(), skip_special_tokens=True)
        print(f"  Encoder:  {(t_enc - t0)*1000:.1f}ms")
        print(f"  Decoder:  {(t_dec - t_enc)*1000:.1f}ms")
        print(f"  Total:    {(t_dec - t0)*1000:.1f}ms")
        print(f"  Text:     {text1}")

        del model, engine, context
        torch.cuda.empty_cache()
    else:
        print(f"  TRT engine not found: {engine_path}")
        print(f"  Run: python pipeline/phase3_whisper_medium.py build")

    # --- Method 2: HF generate() with speculative decoding ---
    print("\n=== Method 2: HF generate() with assistant model ===")

    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MEDIUM_ID, torch_dtype=torch.float16
    ).cuda().eval()

    assistant = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_SMALL_ID, torch_dtype=torch.float16
    ).cuda().eval()

    processor = WhisperProcessor.from_pretrained(WHISPER_MEDIUM_ID)

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.cuda().half()

    # Warmup
    with torch.no_grad():
        model.generate(input_features=input_features, max_new_tokens=10)

    # Timed: without assistant
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        ids_base = model.generate(
            input_features=input_features, max_new_tokens=WHISPER_MAX_TOKENS
        )
    torch.cuda.synchronize()
    t_base = time.perf_counter() - t0
    text_base = processor.batch_decode(ids_base, skip_special_tokens=True)[0]

    # Timed: with assistant (speculative)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        ids_spec = model.generate(
            input_features=input_features,
            assistant_model=assistant,
            max_new_tokens=WHISPER_MAX_TOKENS,
        )
    torch.cuda.synchronize()
    t_spec = time.perf_counter() - t0
    text_spec = processor.batch_decode(ids_spec, skip_special_tokens=True)[0]

    print(f"  Without assistant: {t_base*1000:.0f}ms")
    print(f"  With assistant:    {t_spec*1000:.0f}ms")
    print(f"  Speedup:           {t_base/t_spec:.2f}x")
    print(f"  Base text:         {text_base}")
    print(f"  Spec text:         {text_spec}")
    print(f"  Match:             {text_base.strip() == text_spec.strip()}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    action = sys.argv[1]
    if action == "export":
        step_export()
    elif action == "build":
        step_build()
    elif action == "test":
        step_test()
    elif action == "all":
        step_export()
        step_build()
        step_test()
    else:
        print(f"Unknown action: {action}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
