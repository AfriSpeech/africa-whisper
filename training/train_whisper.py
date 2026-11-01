import os
import io
import time
import math
import torch
import soundfile as sf
import pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "openai/whisper-large-v3-turbo"
OUTPUT_DIR = Path("/data/run-whisper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Initialize Processor with English conditioning for universal transcription
processor = WhisperProcessor.from_pretrained(
    MODEL_ID,
    language="en",
    task="transcribe",
)

# 2. Dataset class for Parquet partitions
from functools import lru_cache

@lru_cache(maxsize=32)
def _get_parquet_table(path: str):
    return pq.read_table(path, columns=["audio_bytes", "text"])

class Afri2UniversalDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.files = sorted(self.root.glob("language=*/part-0.parquet"))
        self.index = []
        
        for f in self.files:
            meta = pq.read_metadata(f)
            num_rows = meta.num_rows
            for r in range(num_rows):
                self.index.append((str(f), r))
                
        print(f"[{root_dir}] Indexed {len(self.index):,} samples across {len(self.files)} languages.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_path, row_idx = self.index[idx]
        t = _get_parquet_table(file_path)
        row = t.slice(row_idx, 1)
        
        audio_bytes = row["audio_bytes"][0].as_py()
        text = row["text"][0].as_py()
        
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        input_features = processor.feature_extractor(
            audio, sampling_rate=sr
        ).input_features[0]
        
        labels = processor.tokenizer(text, max_length=448, truncation=True).input_ids
        
        return {"input_features": input_features, "labels": labels}

# 3. Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(dtype=torch.bfloat16)

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        labels = labels[:, :448]
        batch["labels"] = labels
        return batch

# 4. Metrics computation
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

def main():
    print(f"Loading Whisper model: {MODEL_ID} in bfloat16...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # Configure generation
    model.config.forced_decoder_ids = None
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.suppress_tokens = []

    # Configure LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Building datasets...")
    train_dataset = Afri2UniversalDataset("/data/corpus-univ/version=0/corpus=afripa/split=train")
    dev_dataset = Afri2UniversalDataset("/data/corpus-univ/version=0/corpus=afripa/split=dev")

    # Take a representative 500-sample subset of dev for fast validation during training
    val_indices = list(range(0, len(dev_dataset), max(1, len(dev_dataset) // 500)))[:500]
    eval_dataset = torch.utils.data.Subset(dev_dataset, val_indices)
    print(f"Evaluation dataset set to {len(eval_dataset)} held-out samples for fast validation.")

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=1000,
        max_steps=200000,
        gradient_checkpointing=True,
        bf16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        save_total_limit=5,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("=== Starting Whisper Fine-Tuning on H200 (Target: 200,000 steps) ===")
    last_ckpt = None
    ckpts = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        last_ckpt = str(ckpts[-1])
        print(f"Found existing checkpoint: {last_ckpt}, resuming training!")

    trainer.train(resume_from_checkpoint=last_ckpt)

if __name__ == "__main__":
    main()
