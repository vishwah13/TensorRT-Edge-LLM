"""Export Whisper large-v3 encoder and decoder to ONNX format.

Uses HuggingFace transformers with eager attention (no SDPA)
to avoid ONNX export issues with scaled_dot_product_attention.
"""

import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "openai/whisper-large-v3"
OUTPUT_DIR = "/workspace/whisper/onnx"


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        return self.encoder(input_features).last_hidden_state


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, decoder_input_ids, encoder_hidden_states):
        out = self.model(
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=decoder_input_ids,
        )
        return out.logits


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_ID} with eager attention...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).cuda().eval()

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(OUTPUT_DIR)
    print("Model loaded.")

    # --- Export Encoder ---
    print("\n=== Exporting Encoder ===")
    encoder_wrapper = EncoderWrapper(model.model.encoder).cuda().eval()
    dummy_mel = torch.randn(1, 128, 3000, dtype=torch.float16, device="cuda")

    encoder_path = os.path.join(OUTPUT_DIR, "encoder.onnx")
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

    total_enc = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("encoder")
    )
    print(f"Encoder saved to {encoder_path}")
    print(f"Encoder total size: {total_enc / 1e9:.2f} GB")

    # --- Export Decoder ---
    print("\n=== Exporting Decoder ===")
    decoder_wrapper = DecoderWrapper(model).cuda().eval()
    # Use 4 tokens so dynamic seq_len axis traces correctly (not baked to 1)
    dummy_tokens = torch.tensor([[50258, 50259, 50360, 50364]], dtype=torch.long, device="cuda")
    dummy_enc_out = torch.randn(1, 1500, 1280, dtype=torch.float16, device="cuda")

    decoder_path = os.path.join(OUTPUT_DIR, "decoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            decoder_wrapper,
            (dummy_tokens, dummy_enc_out),
            decoder_path,
            input_names=["decoder_input_ids", "encoder_hidden_states"],
            output_names=["logits"],
            dynamic_axes={
                "decoder_input_ids": {0: "batch", 1: "seq_len"},
                "encoder_hidden_states": {0: "batch"},
                "logits": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    total_dec = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("decoder")
    )
    print(f"Decoder saved to {decoder_path}")
    print(f"Decoder total size: {total_dec / 1e9:.2f} GB")

    print("\n=== All files ===")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            print(f"  {f:40s} {sz / 1e6:>10.1f} MB")

    print("\n=== ONNX export complete ===")


if __name__ == "__main__":
    main()
