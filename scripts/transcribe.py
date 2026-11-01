#!/usr/bin/env python3
"""
Inference Script for Africa-Whisper.

Transcribes audio from any of the 314 African languages using the Africa-Whisper LoRA adapter.

Requirements:
    pip install torch transformers peft soundfile librosa

Usage:
    # 1. Transcribe a single audio file:
    python transcribe.py --audio sample.wav

    # 2. Transcribe using a specific checkpoint:
    python transcribe.py --audio sample.wav --checkpoint best  # or best_cer, or root

    # 3. Use in Python code:
    from transcribe import AfricaWhisper
    asr = AfricaWhisper()
    transcript = asr.transcribe("sample.wav")
    print(transcript)
"""

import argparse
import torch
import soundfile as sf
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel


class AfricaWhisper:
    def __init__(
        self,
        base_model_id: str = "openai/whisper-large-v3-turbo",
        adapter_id: str = "AfriSpeech/africa-whisper-checkpoints",
        subfolder: str = None,
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and self.device == "cuda") else torch.float32

        print(f"Loading processor: {base_model_id} (English conditioning prefix)...")
        self.processor = AutoProcessor.from_pretrained(base_model_id)

        print(f"Loading base model: {base_model_id}...")
        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            base_model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        print(f"Attaching Africa-Whisper LoRA adapter: {adapter_id} (subfolder={subfolder})...")
        if subfolder:
            self.model = PeftModel.from_pretrained(base_model, adapter_id, subfolder=subfolder)
        else:
            self.model = PeftModel.from_pretrained(base_model, adapter_id)

        self.model.eval()
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="en", task="transcribe", no_timestamps=True
        )
        print("Africa-Whisper initialized successfully!")

    def transcribe(self, audio_path_or_array, sampling_rate: int = 16000) -> str:
        """
        Transcribes speech into universal Latin text.

        Args:
            audio_path_or_array: Path to audio file (str) or raw 1D numpy array.
            sampling_rate: Target sampling rate (defaults to 16000 Hz).
        """
        if isinstance(audio_path_or_array, str):
            audio, sr = librosa.load(audio_path_or_array, sr=sampling_rate)
        else:
            audio = audio_path_or_array
            if sampling_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.torch_dtype)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs,
                forced_decoder_ids=self.forced_decoder_ids,
                max_length=448,
            )

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Transcribe African speech with Africa-Whisper")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (wav, flac, mp3, ogg)")
    parser.add_argument("--adapter_id", type=str, default="AfriSpeech/africa-whisper-checkpoints", help="HF model adapter ID")
    parser.add_argument("--subfolder", type=str, default=None, help="Adapter subfolder: None (root), 'best', or 'best_cer'")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on ('cuda' or 'cpu')")
    args = parser.parse_args()

    asr = AfricaWhisper(
        adapter_id=args.adapter_id,
        subfolder=args.subfolder,
        device=args.device,
    )
    transcription = asr.transcribe(args.audio)
    print("\n" + "=" * 50)
    print("TRANSCRIPTION (Africa-Whisper Universal):")
    print("=" * 50)
    print(transcription)
    print("=" * 50)


if __name__ == "__main__":
    main()
