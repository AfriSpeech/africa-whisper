#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="/mnt/volume_d2wey28/projects/afripa-asr"
cd "$PROJ_DIR"

source .venv/bin/activate

mkdir -p /mnt/volume_d2wey28/tmp
export TMPDIR="/mnt/volume_d2wey28/tmp"
export TEMP="/mnt/volume_d2wey28/tmp"
export TMP="/mnt/volume_d2wey28/tmp"
export HF_HOME="/mnt/volume_d2wey28/hf_home"
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || true)"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

echo "=== Whisper Large-v3-Turbo LoRA Fine-Tuning on H200 (Target: 200,000 steps) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Started at $(date)"

python "$PROJ_DIR/train_whisper.py"
