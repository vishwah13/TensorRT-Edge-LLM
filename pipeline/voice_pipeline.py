"""Voice-to-voice pipeline: Whisper ASR -> LLM -> Piper TTS.

Orchestrates the full pipeline with sentence-level streaming to TTS.
Since llm_inference writes output at the end (no token streaming),
we pipeline by splitting the LLM response into sentences and feeding
them to TTS sequentially while playing audio.

Architecture:
    User speaks -> [Whisper ASR] -> [LLM inference] -> [Split sentences]
                                                          |
                                    [TTS sentence 1] -> [Play audio 1]
                                    [TTS sentence 2] -> [Play audio 2]
                                    ...

Usage:
    # Interactive conversation loop
    python pipeline/voice_pipeline.py

    # Single-shot from audio file
    python pipeline/voice_pipeline.py --audio test_audio.mp3

    # With image context
    python pipeline/voice_pipeline.py --audio test_audio.mp3 --image test_image.jpg

    # Use EAGLE3 (Phase 2 engine)
    python pipeline/voice_pipeline.py --eagle

    # Use Whisper medium (Phase 3)
    python pipeline/voice_pipeline.py --whisper-medium
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from queue import Queue

import torch
import tensorrt as trt
import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    LLM_INFERENCE, WORKSPACE,
    QWEN3_LLM_ENGINE, QWEN3_LLM_ENGINE_FAST, QWEN3_VISUAL_ENGINE,
    QWEN25_ENGINE_LLM, QWEN25_ENGINE_VISUAL,
    WHISPER_LARGE_ID, WHISPER_MEDIUM_ID, WHISPER_SMALL_ID,
    WHISPER_LARGE_ENGINE, WHISPER_LARGE_PROCESSOR,
    WHISPER_MEDIUM_ENGINE, WHISPER_MEDIUM_ONNX,
    DEFAULT_MAX_GENERATE_LENGTH, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
    SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS, EOT, WHISPER_MAX_TOKENS,
)


class TRTEncoder:
    """TensorRT engine wrapper for Whisper encoder."""

    def __init__(self, engine_path, n_mels=128):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("input_features", (1, n_mels, 3000))
        self.n_mels = n_mels

    def __call__(self, mel_tensor):
        out_shape = self.context.get_tensor_shape("encoder_hidden_states")
        output = torch.empty(tuple(out_shape), dtype=torch.float16, device="cuda")
        self.context.set_tensor_address("input_features", mel_tensor.data_ptr())
        self.context.set_tensor_address("encoder_hidden_states", output.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return output


class WhisperASR:
    """Whisper ASR with TRT encoder + PyTorch decoder.

    Supports both large-v3 (128 mel bins) and medium (80 mel bins).
    """

    def __init__(self, model_id, engine_path, processor_dir):
        self.model_id = model_id
        print(f"[ASR] Loading {model_id}...")

        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, attn_implementation="eager"
        ).cuda().eval()

        self.n_mels = self.model.config.num_mel_bins
        self.encoder = TRTEncoder(engine_path, n_mels=self.n_mels)
        self.processor = WhisperProcessor.from_pretrained(processor_dir)
        print(f"[ASR] Ready (n_mels={self.n_mels})")

    def transcribe(self, audio_path):
        """Transcribe audio file to text. Returns (text, latency_ms)."""
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        mel = mel.unsqueeze(0).cuda().half()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        enc_out = self.encoder(mel)

        tokens = torch.tensor([[SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS]], device="cuda")
        for _ in range(WHISPER_MAX_TOKENS):
            with torch.no_grad():
                out = self.model(encoder_outputs=(enc_out,), decoder_input_ids=tokens)
            next_id = out.logits[0, -1, :].argmax().item()
            if next_id == EOT:
                break
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device="cuda")], dim=1)

        text = self.processor.tokenizer.decode(tokens[0, 4:].tolist(), skip_special_tokens=True)
        latency = (time.perf_counter() - t0) * 1000
        return text.strip(), latency


class WhisperASRSpeculative:
    """Whisper ASR with HF generate() + assistant model for speculative decoding.

    Uses whisper-small as the draft model for ~2x speedup.
    No TRT encoder needed — runs entirely in PyTorch.
    """

    def __init__(self, model_id=WHISPER_MEDIUM_ID, assistant_id=WHISPER_SMALL_ID):
        print(f"[ASR] Loading {model_id} + {assistant_id} (speculative)...")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).cuda().eval()
        self.assistant = WhisperForConditionalGeneration.from_pretrained(
            assistant_id, torch_dtype=torch.float16
        ).cuda().eval()
        self.processor = WhisperProcessor.from_pretrained(model_id)
        print("[ASR] Ready (speculative decoding)")

    def transcribe(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        features = features.cuda().half()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(
                input_features=features,
                assistant_model=self.assistant,
                max_new_tokens=WHISPER_MAX_TOKENS,
                language="en",
                task="transcribe",
            )
        torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000

        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip(), latency


class LLMInference:
    """Wrapper around llm_inference C++ binary."""

    def __init__(self, engine_dir, visual_engine_dir, eagle=False):
        self.engine_dir = engine_dir
        self.visual_engine_dir = visual_engine_dir
        self.eagle = eagle
        self.input_file = os.path.join(WORKSPACE, "pipeline/_llm_input.json")
        self.output_file = os.path.join(WORKSPACE, "pipeline/_llm_output.json")
        os.makedirs(os.path.dirname(self.input_file), exist_ok=True)

    def generate(self, prompt, image_path=None, system_prompt=None,
                 max_tokens=DEFAULT_MAX_GENERATE_LENGTH):
        """Run LLM inference. Returns (response_text, latency_ms)."""
        content = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content if image_path else prompt})

        data = {
            "batch_size": 1,
            "temperature": DEFAULT_TEMPERATURE,
            "top_k": DEFAULT_TOP_K,
            "max_generate_length": max_tokens,
            "requests": [{"messages": messages}]
        }

        with open(self.input_file, "w") as f:
            json.dump(data, f, indent=2)

        cmd = [
            LLM_INFERENCE,
            "--engineDir", self.engine_dir,
            "--multimodalEngineDir", self.visual_engine_dir,
            "--inputFile", self.input_file,
            "--outputFile", self.output_file,
        ]
        if self.eagle:
            cmd += ["--eagle", "--eagleDraftTopK", "10", "--eagleDraftStep", "6"]

        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        latency = (time.perf_counter() - t0) * 1000

        if result.returncode != 0:
            raise RuntimeError(f"LLM inference failed: {result.stderr[-500:]}")

        with open(self.output_file) as f:
            output = json.load(f)

        text = output["responses"][0]["output_text"]
        return text, latency


class PiperTTS:
    """Piper TTS wrapper for text-to-speech synthesis."""

    def __init__(self, voice_name="en_US-lessac-medium", voices_dir=None):
        voices_dir = voices_dir or os.path.join(WORKSPACE, "piper/voices")
        self.model_path = os.path.join(voices_dir, voice_name, f"{voice_name}.onnx")
        self._voice = None

        if not os.path.exists(self.model_path):
            print(f"[TTS] Voice not found: {self.model_path}")
            print("[TTS] Run: bash pipeline/phase4_tts_setup.sh")
            self._available = False
        else:
            self._available = True
            print(f"[TTS] Voice: {voice_name}")

    def _get_voice(self):
        if self._voice is None and self._available:
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.model_path)
        return self._voice

    def synthesize(self, text):
        """Synthesize text to WAV file. Returns (wav_path, latency_ms)."""
        voice = self._get_voice()
        if voice is None:
            return None, 0

        wav_path = tempfile.mktemp(suffix=".wav")
        t0 = time.perf_counter()
        audio_bytes = b""
        for chunk in voice.synthesize(text):
            audio_bytes += chunk.audio_int16_bytes
        latency = (time.perf_counter() - t0) * 1000

        with wave.open(wav_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(voice.config.sample_rate)
            wav.writeframes(audio_bytes)
        return wav_path, latency

    def synthesize_and_play(self, text, speaker_device=None):
        """Synthesize and play audio. Returns latency_ms."""
        wav_path, synth_ms = self.synthesize(text)
        if wav_path is None:
            print(f"  [TTS unavailable] {text}")
            return 0

        # Play audio if aplay is available (headless environments skip playback)
        play_ms = 0
        try:
            cmd = ["aplay", "-q"]
            if speaker_device:
                cmd += ["-D", speaker_device]
            cmd.append(wav_path)
            t0 = time.perf_counter()
            subprocess.run(cmd, capture_output=True, check=True)
            play_ms = (time.perf_counter() - t0) * 1000
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass  # No audio device — TTS latency is still measured

        os.unlink(wav_path)
        return synth_ms + play_ms


def split_sentences(text):
    """Split text into sentence chunks for incremental TTS."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Merge very short fragments with the previous sentence
    merged = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if merged and len(merged[-1]) < 20:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


class VoicePipeline:
    """End-to-end voice-to-voice pipeline."""

    def __init__(self, asr, llm, tts):
        self.asr = asr
        self.llm = llm
        self.tts = tts

    def process(self, audio_path, image_path=None, speaker_device=None, save_audio=None):
        """Process audio input through full pipeline. Returns timing dict."""
        timings = {}
        t_pipeline_start = time.perf_counter()

        # Step 1: ASR
        print(f"\n[1/3] Transcribing...")
        transcription, asr_ms = self.asr.transcribe(audio_path)
        timings["asr_ms"] = asr_ms
        print(f"  [{asr_ms:.0f}ms] \"{transcription}\"")

        if not transcription:
            print("  No speech detected.")
            return timings

        # Step 2: LLM
        print(f"\n[2/3] Thinking...")
        prompt = f'The user said: "{transcription}"\n\nRespond helpfully and concisely.'
        system = "You are a helpful voice assistant. Keep responses brief and conversational."
        response, llm_ms = self.llm.generate(
            prompt, image_path=image_path, system_prompt=system
        )
        timings["llm_ms"] = llm_ms
        # Strip emoji for cleaner TTS
        response_text = response.encode("ascii", "ignore").decode("ascii").strip()
        print(f"  [{llm_ms:.0f}ms] {response_text[:200]}")

        # Step 3: TTS — synthesize each sentence and play
        print(f"\n[3/3] Speaking...")
        sentences = split_sentences(response_text)
        timings["tts_sentences"] = len(sentences)
        timings["tts_ms"] = 0

        all_audio = b""  # for --save-audio
        sample_rate = None

        t_first_audio = None
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Synthesize
            wav_path, synth_ms = self.tts.synthesize(sentence)
            timings["tts_ms"] += synth_ms

            if wav_path:
                # Collect audio for save
                if save_audio:
                    import wave as _wave
                    with _wave.open(wav_path, "rb") as wf:
                        sample_rate = wf.getframerate()
                        all_audio += wf.readframes(wf.getnframes())

                # Play
                play_ms = 0
                try:
                    cmd = ["aplay", "-q"]
                    if speaker_device:
                        cmd += ["-D", speaker_device]
                    cmd.append(wav_path)
                    t0 = time.perf_counter()
                    subprocess.run(cmd, capture_output=True, check=True)
                    play_ms = (time.perf_counter() - t0) * 1000
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass
                timings["tts_ms"] += play_ms
                os.unlink(wav_path)

            if t_first_audio is None:
                t_first_audio = (time.perf_counter() - t_pipeline_start) * 1000
                timings["time_to_first_audio_ms"] = t_first_audio
            print(f"  [{synth_ms:.0f}ms] {sentence[:70]}")

        # Save combined audio if requested
        if save_audio and all_audio and sample_rate:
            import wave as _wave
            with _wave.open(save_audio, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(all_audio)
            print(f"\n  Audio saved: {save_audio}")

        timings["total_ms"] = (time.perf_counter() - t_pipeline_start) * 1000
        return timings

    def interactive_loop(self, image_path=None, record_seconds=5,
                         mic_device=None, speaker_device=None):
        """Interactive voice conversation using microphone recording."""
        print("\n" + "=" * 60)
        print("  Voice Assistant - Talk and Listen")
        print("=" * 60)
        print(f"  Mic:     {mic_device or 'default'}")
        print(f"  Speaker: {speaker_device or 'default'}")
        print(f"  Record:  {record_seconds}s per turn")
        print(f"  Image:   {image_path or 'none'}")
        print("=" * 60)
        print("\nPress Enter to start talking. Say 'quit' or Ctrl+C to exit.\n")

        turn = 0
        while True:
            try:
                input(f"[Turn {turn + 1}] Press Enter to record...")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            # Record
            audio_path = self._record_audio(
                duration=record_seconds, mic_device=mic_device
            )
            if audio_path is None:
                continue

            turn += 1

            # Process full pipeline
            timings = self.process(
                audio_path, image_path,
                speaker_device=speaker_device,
            )
            os.unlink(audio_path)

            # Print timing
            print(f"\n  --- Turn {turn} Timing ---")
            for k, v in timings.items():
                if k.endswith("_ms"):
                    print(f"    {k:<30s}: {v:,.0f}ms")

    def _record_audio(self, duration=5, sample_rate=16000, mic_device=None):
        """Record audio from microphone using arecord."""
        audio_path = tempfile.mktemp(suffix=".wav")
        cmd = ["arecord", "-f", "S16_LE", "-r", str(sample_rate),
               "-c", "1", "-d", str(duration)]
        if mic_device:
            cmd += ["-D", mic_device]
        cmd.append(audio_path)

        print(f"  Recording {duration}s...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=duration + 2)
            if result.returncode != 0:
                print(f"  arecord error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            pass
        except FileNotFoundError:
            print("  arecord not found. Install: apt-get install alsa-utils")
            return None

        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            print(f"  Recorded: {os.path.getsize(audio_path)} bytes")
            return audio_path
        print("  Recording failed or too short.")
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Voice-to-voice pipeline")
    parser.add_argument("--audio", help="Audio file to process (skip interactive mode)")
    parser.add_argument("--image", help="Optional image for VLM context")

    # ASR config
    parser.add_argument("--whisper-medium", action="store_true",
                        help="Use Whisper medium (Phase 3) instead of large-v3")
    parser.add_argument("--whisper-speculative", action="store_true",
                        help="Use HF speculative decoding (medium + small)")

    # LLM config
    parser.add_argument("--eagle", action="store_true",
                        help="Use Qwen2.5-VL-7B with EAGLE3 (Phase 2)")
    parser.add_argument("--fast-engine", action="store_true",
                        help="Use optimized Qwen3-VL engine (Phase 1)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_GENERATE_LENGTH)

    # TTS config
    parser.add_argument("--voice", default="en_US-lessac-medium",
                        help="Piper TTS voice name")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable TTS output")

    # Audio devices
    parser.add_argument("--mic", default="hw:0,0",
                        help="ALSA mic device (default: hw:0,0 = Brio 100)")
    parser.add_argument("--speaker", default="plughw:1,3",
                        help="ALSA speaker device (default: plughw:1,3 = HDMI monitor, auto mono->stereo)")
    parser.add_argument("--record-seconds", type=int, default=5,
                        help="Seconds to record per turn (default: 5)")

    # Output
    parser.add_argument("--save-audio", metavar="FILE",
                        help="Save TTS response to WAV file")

    # Benchmark
    parser.add_argument("--benchmark", type=int, metavar="N",
                        help="Run N iterations for benchmarking")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup iterations before benchmark")

    return parser.parse_args()


def build_pipeline(args):
    """Construct pipeline components based on args."""

    # ASR
    if args.whisper_speculative:
        asr = WhisperASRSpeculative()
    elif args.whisper_medium:
        asr = WhisperASR(WHISPER_MEDIUM_ID, WHISPER_MEDIUM_ENGINE, WHISPER_MEDIUM_ONNX)
    else:
        asr = WhisperASR(WHISPER_LARGE_ID, WHISPER_LARGE_ENGINE, WHISPER_LARGE_PROCESSOR)

    # LLM
    if args.eagle:
        llm = LLMInference(QWEN25_ENGINE_LLM, QWEN25_ENGINE_VISUAL, eagle=True)
    elif args.fast_engine:
        llm = LLMInference(QWEN3_LLM_ENGINE_FAST, QWEN3_VISUAL_ENGINE)
    else:
        llm = LLMInference(QWEN3_LLM_ENGINE, QWEN3_VISUAL_ENGINE)

    # TTS
    if args.no_tts:
        tts = PiperTTS.__new__(PiperTTS)
        tts._available = False
        tts._voice = None
        tts.synthesize = lambda text: (None, 0)
        tts.synthesize_and_play = lambda text, speaker_device=None: 0
    else:
        tts = PiperTTS(voice_name=args.voice)

    return VoicePipeline(asr, llm, tts)


def main():
    args = parse_args()
    pipeline = build_pipeline(args)

    if args.benchmark and args.audio:
        # Benchmark mode
        print(f"\nBenchmark: {args.benchmark} iterations, {args.warmup} warmup")
        all_timings = []
        for i in range(args.warmup + args.benchmark):
            label = "warmup" if i < args.warmup else f"run {i - args.warmup + 1}"
            print(f"\n--- {label} ---")
            t = pipeline.process(args.audio, args.image)
            if i >= args.warmup:
                all_timings.append(t)

        # Aggregate
        print(f"\n{'='*60}")
        print(f"Benchmark Results ({args.benchmark} runs)")
        print(f"{'='*60}")
        for key in ["asr_ms", "llm_ms", "tts_ms", "time_to_first_audio_ms", "total_ms"]:
            vals = [t[key] for t in all_timings if key in t]
            if vals:
                vals.sort()
                p50 = vals[len(vals) // 2]
                p95 = vals[int(len(vals) * 0.95)]
                avg = sum(vals) / len(vals)
                print(f"  {key:30s}: avg={avg:.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms")

    elif args.audio:
        # Single-shot mode
        timings = pipeline.process(
            args.audio, args.image,
            speaker_device=args.speaker,
            save_audio=args.save_audio,
        )
        print(f"\n--- Timing Summary ---")
        for k, v in timings.items():
            if k.endswith("_ms"):
                print(f"  {k:30s}: {v:,.0f}ms")
            else:
                print(f"  {k:30s}: {v}")

    else:
        # Interactive live mode
        pipeline.interactive_loop(
            image_path=args.image,
            record_seconds=args.record_seconds,
            mic_device=args.mic,
            speaker_device=args.speaker,
        )


if __name__ == "__main__":
    main()
