#!/usr/bin/env python3
"""
Fine-Tuning Script for Africa-Whisper.

Continues fine-tuning Africa-Whisper LoRA adapter on new African speech datasets
(e.g., Waxal, OmniASR, Common Voice, FLEURS, or local audio).

Requirements:
    pip install torch transformers datasets peft evaluate soundfile librosa accelerate

Usage:
    python finetune_whisper.py \
        --adapter_model_id "AfriSpeech/africa-whisper-checkpoints" \
        --dataset_name "google/fleurs" \
        --subset "yo_ng" \
        --output_dir "./output-africa-whisper-fleurs"
"""

import os
import argparse
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import PeftModel, LoraConfig, get_peft_model
from prepare_dataset import universalize_text


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(dtype=torch.bfloat16)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Africa-Whisper on new datasets")
    parser.add_argument("--base_model_id", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--adapter_model_id", type=str, default="AfriSpeech/africa-whisper-checkpoints")
    parser.add_argument("--dataset_name", type=str, default="google/fleurs", help="HF dataset name or local path")
    parser.add_argument("--subset", type=str, default="yo_ng", help="Dataset language/subset")
    parser.add_argument("--lang_code", type=str, default="yor", help="Language ISO 639-3 code for Africa-G2P universalization")
    parser.add_argument("--audio_column", type=str, default="audio", help="Column name containing audio")
    parser.add_argument("--text_column", type=str, default="transcription", help="Column name containing raw transcript")
    parser.add_argument("--output_dir", type=str, default="./run-africa-whisper-continued")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading processor for {args.base_model_id} (English conditioning prefix)...")
    processor = AutoProcessor.from_pretrained(args.base_model_id, language="en", task="transcribe")

    print(f"Loading base model {args.base_model_id}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if args.adapter_model_id:
        print(f"Loading existing Africa-Whisper LoRA adapter: {args.adapter_model_id} ...")
        model = PeftModel.from_pretrained(model, args.adapter_model_id, is_trainable=True)
    else:
        print("Configuring new LoRA adapter...")
        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    print(f"Loading dataset: {args.dataset_name} ({args.subset}) ...")
    raw_dataset = load_dataset(args.dataset_name, args.subset)
    raw_dataset = raw_dataset.cast_column(args.audio_column, Audio(sampling_rate=16000))

    def prepare_sample(batch):
        audio = batch[args.audio_column]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        raw_text = batch[args.text_column]
        clean_text = universalize_text(raw_text, args.lang_code)
        batch["labels"] = processor.tokenizer(clean_text, max_length=448, truncation=True).input_ids
        return batch

    print("Normalizing transcripts to universal Latin space via Africa-G2P...")
    prepared_dataset = raw_dataset.map(
        prepare_sample,
        remove_columns=raw_dataset["train"].column_names,
        desc="Preparing dataset",
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=448,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset.get("validation", prepared_dataset.get("test")),
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
