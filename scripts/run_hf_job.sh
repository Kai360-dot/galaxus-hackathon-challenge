#!/usr/bin/env bash
set -euo pipefail

# Requires: hf auth login (or HF_TOKEN exported) and data uploaded to a private HF dataset repo
HF_USER=$(hf auth whoami | head -n 1 | awk '{print $1}')
DATASET_REPO="${HF_USER}/galaxus-challenge-data"
OUTPUT_REPO="${HF_USER}/apertus-8b-galaxus-lora"

echo "1. Uploading local data/ to private HF dataset ${DATASET_REPO}..."
hf repo create "${DATASET_REPO}" --repo-type dataset --private -y || true
hf upload "${DATASET_REPO}" ./data . --repo-type dataset

echo "2. Launching training & 6-dimension evaluation on Hugging Face L40S GPU..."
hf jobs run \
  --flavor l40sx1 \
  --timeout 2h \
  --secrets HF_TOKEN \
  --name apertus-galaxus-train \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  bash -c "
    set -e
    pip install -q 'transformers[torch,vision] @ git+https://github.com/swiss-ai/transformers.git' peft accelerate scikit-learn pandas pillow huggingface_hub
    git clone -b feat/finetune-kaggle-pipeline https://github.com/Kai360-dot/galaxus-hackathon-challenge.git
    cd galaxus-hackathon-challenge
    hf download ${DATASET_REPO} --repo-type dataset --local-dir ./data

    # Train SigLIP-2 with class-balanced loss + TTA + 6-dim eval
    python src/train_siglip2.py \
      --train-csv data/train_clean.csv \
      --train-img data/train_images \
      --val-csv data/synthetic_test.csv \
      --val-img data/synthetic_images \
      --out-pred outputs/siglip2_val_preds.csv \
      --out-probs outputs/siglip2_val_probs.npz \
      --epochs 6 \
      --batch-size 32

    python src/tune_thresholds.py \
      --truth-csv data/synthetic_test.csv \
      --probs-npz outputs/siglip2_val_probs.npz \
      --out-json outputs/class_multipliers.json

    # Train Apertus-v1.5-8B native bf16 LoRA + 6-dim eval + dev inference
    python masking/train_lora.py \
      --train-csv data/train_clean.csv \
      --img-dir data/train_images \
      --val-csv data/synthetic_test.csv \
      --val-img data/synthetic_images \
      --val-out-csv outputs/apertus_val_preds.csv \
      --dev-csv data/dev.csv \
      --dev-img data/dev_images \
      --dev-out-csv outputs/apertus_dev_preds.csv \
      --out-dir outputs/apertus-8b-lora \
      --hub-repo ${OUTPUT_REPO} \
      --epochs 3 \
      --batch-size 4 \
      --grad-accum 4

    # Evaluate blended ensemble on all 6 dimensions
    python src/ensemble.py \
      --siglip-probs outputs/siglip2_val_probs.npz \
      --apertus-csv outputs/apertus_val_preds.csv \
      --multipliers-json outputs/class_multipliers.json \
      --out-csv outputs/ensemble_val_preds.csv

    python src/eval_dimensions.py \
      --truth data/synthetic_test.csv \
      --pred outputs/ensemble_val_preds.csv

    # Upload all evaluation tables and dev predictions back to HF repo
    hf upload ${OUTPUT_REPO} ./outputs outputs --repo-type model
  "