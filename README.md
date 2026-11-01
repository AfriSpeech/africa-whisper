# Africa-Whisper 🌍
### Universal Multilingual Speech Recognition for 314 African Languages

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Checkpoints-blue)](https://huggingface.co/AfriSpeech/africa-whisper-checkpoints)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/AfriSpeech/afri2universal)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**Africa-Whisper** is an open-source speech recognition foundation model adapted from OpenAI's **Whisper Large-v3-Turbo** (809M parameters) across **314 African languages and dialects**.

Conditioned on the English transcription prompt (`<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`), Africa-Whisper leverages Whisper's vast multilingual subword priors to transcribe spoken audio directly into a regularized Latin orthography with preserved word boundaries (single spaces) and stripped punctuation, generated using [Africa-G2P](https://github.com/AfriSpeech/africa-g2p).

---

## 📊 Final Release Metrics (Checkpoint-76000 / Epoch 16.62)

We selected **Step 76,000 (Epoch 16.62)** as the official foundation release checkpoint:

| Metric | Score across 314 African Languages |
|:---|:---:|
| **Word Error Rate (WER)** | **`45.74%`** (All-time best in the entire training run) |
| **Character Error Rate (CER)** | **`20.50%`** (Near all-time record of 20.09%) |
| **Evaluation Loss (Cross-Entropy)** | **`0.7528`** (60.7% total reduction from 1.9156 baseline) |
| **Training Steps Completed** | **76,000 steps** (Effective batch size 64 = 4.86M sample exposures) |
| **Full Passes over Corpus** | **16.62 full epochs** |

The model weights, optimizer states, scheduler states, and preprocessor configs are live on Hugging Face:
👉 **[AfriSpeech/africa-whisper-checkpoints](https://huggingface.co/AfriSpeech/africa-whisper-checkpoints)**

---

## 🤝 An Open Community Project: Call for Continued Fine-Tuning

Africa-Whisper is designed as a **collaborative community project**. While this foundation checkpoint establishes a strong cross-lingual acoustic baseline across 314 languages, real-world accuracy on specific dialects, accents, and local domains will improve as the community fine-tunes on additional datasets.

We warmly invite researchers, engineers, and language communities across Africa to take this foundation checkpoint and fine-tune it further!

### Dataset Trained On

| Dataset | Languages | Total Duration | Utterances | Families Represented |
|:---|:---:|:---:|:---:|:---|
| **[Afri2Universal](https://huggingface.co/datasets/AfriSpeech/afri2universal)** | **314** | **~950.6 hours** | **292,591** | Niger-Congo, Afroasiatic, Nilo-Saharan, Khoisan |

### Suggested Datasets for Continued Community Fine-Tuning

We recommend continuing fine-tuning on these open African speech corpora:

1. **Waxal Corpus:** High-quality speech for West African languages (Wolof, Pulaar/Fulfulde, Mandinka, Jola, Serer).
2. **OmniASR African Subsets:** Multi-dialect African audio benchmarks.
3. **Mozilla Common Voice (Africa):** Crowdsourced speech (Kinyarwanda, Luganda, Swahili, Hausa, Yoruba, etc.).
4. **Google FLEURS:** 102-language benchmark with African language subsets.
5. **ALFFA (African Languages in the Field):** Field recordings for Amharic, Oromo, Tigrinya, and Wolof.
6. **Local Community Radio & Broadcast Speech:** Authentic local radio discussions, broadcasts, and news.

---

## ⚙️ Standardized Data Preparation (`prepare_dataset.py`)

To ensure that fine-tuning on new datasets maintains subword token alignment and cross-lingual acoustic transfer, **all transcripts must be pre-processed using the identical orthographic normalization standard**:

- Universal 26-character lowercase Latin (`a-z`)
- Single whitespace (` `) preserved between words (enabling real word boundary decoding)
- Stripped punctuation, tone accents, and non-Latin glyphs
- Audio format: **16,000 Hz mono FLAC or WAV**

We provide `scripts/prepare_dataset.py` powered by [Africa-G2P](https://github.com/AfriSpeech/africa-g2p):

```bash
pip install git+https://github.com/AfriSpeech/africa-g2p.git soundfile librosa datasets
```

```python
from scripts.prepare_dataset import universalize_text

# Example: Twi
clean = universalize_text("Onyankopɔn, me din de Kwame! Ɔyɛ kɛse.", lang="twi")
print(clean)
# -> 'onyankopon me din de kwame oye kese'

# Example: Amharic (Ge'ez script transliterated to universal Latin)
clean = universalize_text("ሰላም፣ ስሜ አማን ነው።", lang="amh")
print(clean)
# -> 'salaame semee amaane nawe'
```

---

## 🚀 Quickstart: Inference

### 1. Installation

```bash
git clone https://github.com/AfriSpeech/africa-whisper.git
cd africa-whisper
pip install -r requirements.txt
```

### 2. Python API

```python
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# 1. Load base Whisper Large-v3-Turbo
base_model_id = "openai/whisper-large-v3-turbo"
processor = AutoProcessor.from_pretrained(base_model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    base_model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device)

# 2. Attach Africa-Whisper LoRA adapter
model = PeftModel.from_pretrained(model, "AfriSpeech/africa-whisper-checkpoints")
model.eval()

# 3. Transcribe speech from any of the 314 African languages
def transcribe(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=True)
    
    with torch.no_grad():
        predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids, max_length=448)
        
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

print(transcribe("sample.wav"))
```

### 3. CLI Tool

```bash
python scripts/transcribe.py --audio path/to/african_speech.wav
```

---

## 🏋️ Fine-Tuning on New Datasets

Use `training/finetune.py` to adapt Africa-Whisper to your own dataset:

```bash
python training/finetune.py \
    --adapter_model_id "AfriSpeech/africa-whisper-checkpoints" \
    --dataset_name "google/fleurs" \
    --subset "yo_ng" \
    --lang_code "yor" \
    --output_dir "./output-africa-whisper-fleurs" \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 5
```

---

## 📂 Repository Structure

```
├── README.md               # Documentation and community guide
├── LICENSE                 # CC-BY-NC-4.0
├── pyproject.toml          # Package specifications
├── requirements.txt        # Dependencies
├── scripts/
│   ├── prepare_dataset.py  # Standardized Africa-G2P universal data preparation
│   └── transcribe.py       # Inference CLI & Python class
├── training/
│   ├── train_whisper.py    # Pre-training script with PEFT LoRA, multi-processing, LRU cache
│   ├── finetune.py         # Script to continue fine-tuning on new datasets
│   └── run_whisper.sh      # Launch script with Hopper / bfloat16 optimizations
└── examples/
    └── quickstart.py       # Minimal inference script
```

---

## 📖 Citation

```bibtex
@model{africa_whisper_2026,
  author    = {AfriSpeech},
  title     = {Africa-Whisper: Universal Multilingual African Speech Recognition (314 Languages)},
  year      = {2026},
  publisher = {GitHub},
  howpublished = {\\url{https://github.com/AfriSpeech/africa-whisper}}
}

@software{africa_g2p_2024,
  author       = {AfriSpeech},
  title        = {Africa-G2P: Grapheme-to-Phoneme and Universal Orthographic Converter for African Languages},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\\url{https://github.com/AfriSpeech/africa-g2p}}
}

@dataset{afri2universal_2026,
  author    = {AfriSpeech},
  title     = {Afri2Universal: Universalized Multilingual African Speech Corpus Across 314 Languages},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/AfriSpeech/afri2universal}
}
```

---

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License ([CC-BY-NC-4.0](LICENSE)).
