import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# 1. Load base Whisper Large-v3-Turbo
base_model_id = "openai/whisper-large-v3-turbo"
processor = AutoProcessor.from_pretrained(base_model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
).to(device)

# 2. Attach Africa-Whisper LoRA adapter
model = PeftModel.from_pretrained(model, "AfriSpeech/africa-whisper-checkpoints")
model.eval()

# 3. Transcribe audio from any of the 314 African languages
def transcribe_audio(audio_path):
    import librosa
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=torch_dtype)
    
    # Force English transcription conditioning prefix for universal Latin output
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=True)
    
    with torch.no_grad():
        predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids, max_length=448)
        
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    try:
        print("Transcribing:", audio_file)
        print("Transcript:", transcribe_audio(audio_file))
    except Exception as e:
        print("Provide a valid audio file to test inference, e.g. python quickstart.py audio.wav")
