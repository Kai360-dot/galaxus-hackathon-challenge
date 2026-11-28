#!/usr/bin/env bash
set -euo pipefail

HF_USER="helshahaby"
DATASET_REPO="${HF_USER}/galaxus-challenge-data"
OUTPUT_REPO="${HF_USER}/apertus-8b-galaxus-lora"

echo "1. Packaging and uploading code_bundle.tar.gz to ${DATASET_REPO}..."
tar --exclude="__pycache__" --exclude="*.pyc" -czf code_bundle.tar.gz src masking scripts
hf upload "${DATASET_REPO}" code_bundle.tar.gz code_bundle.tar.gz --repo-type dataset
rm -f code_bundle.tar.gz

echo "2. Launching L40S training and 6-dimension evaluation job..."
hf jobs run \
  --flavor l40sx1 \
  --timeout 2h \
  -s HF_TOKEN \
  --name apertus-galaxus-clean \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel -- \
  bash -c "
    set -e
    apt-get update -qq && apt-get install -y -qq git
    pip uninstall -y torchaudio
    pip install -q peft accelerate scikit-learn pandas pillow huggingface_hub torchvision timm safetensors 'tokenizers>=0.22.0,<=0.23.0' regex requests tqdm numpy httpx pyyaml packaging filelock typing-extensions
    pip install -q --no-deps --force-reinstall 'git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4'

    mkdir -p /workspace/project/data /workspace/project/outputs
    cd /workspace/project

    hf download ${DATASET_REPO} code_bundle.tar.gz --repo-type dataset --local-dir .
    tar --no-same-owner --no-same-permissions -xzf code_bundle.tar.gz -C .

    hf download ${DATASET_REPO} galaxus_real_data.tar.gz --repo-type dataset --local-dir ./data
    tar --no-same-owner --no-same-permissions -xzf ./data/galaxus_real_data.tar.gz -C ./data

    python src/make_real_split.py --in-csv data/train.csv --train-out data/real_train.csv --val-out data/real_val.csv

    python src/train_siglip2.py \
      --train-csv data/real_train.csv \
      --train-img data/train_images \
      --val-csv data/real_val.csv \
      --val-img data/train_images \
      --out-pred outputs/siglip2_val_preds.csv \
      --out-probs outputs/siglip2_val_probs.npz \
      --epochs 6 \
      --batch-size 32

    python src/tune_thresholds.py \
      --truth-csv data/real_val.csv \
      --probs-npz outputs/siglip2_val_probs.npz \
      --out-json outputs/class_multipliers.json

    python src/train_siglip2.py \
      --train-csv data/train.csv \
      --train-img data/train_images \
      --val-csv data/dev.csv \
      --val-img data/dev_images \
      --out-pred outputs/siglip2_dev_preds.csv \
      --out-probs outputs/siglip2_dev_probs.npz \
      --epochs 6 \
      --batch-size 32

    python masking/train_lora.py \
      --train-csv data/real_train.csv \
      --img-dir data/train_images \
      --val-csv data/real_val.csv \
      --val-img data/train_images \
      --val-out-csv outputs/apertus_val_preds.csv \
      --dev-csv data/dev.csv \
      --dev-img data/dev_images \
      --dev-out-csv outputs/apertus_dev_preds.csv \
      --out-dir outputs/apertus-8b-lora \
      --hub-repo ${OUTPUT_REPO} \
      --epochs 3 \
      --batch-size 4 \
      --grad-accum 4

    python src/ensemble.py \
      --siglip-probs outputs/siglip2_val_probs.npz \
      --apertus-csv outputs/apertus_val_preds.csv \
      --multipliers-json outputs/class_multipliers.json \
      --out-csv outputs/ensemble_val_preds.csv

    python src/eval_dimensions.py \
      --truth data/real_val.csv \
      --pred outputs/ensemble_val_preds.csv

    python src/ensemble.py \
      --siglip-probs outputs/siglip2_dev_probs.npz \
      --apertus-csv outputs/apertus_dev_preds.csv \
      --multipliers-json outputs/class_multipliers.json \
      --out-csv outputs/ensemble_dev_preds.csv

    hf upload ${OUTPUT_REPO} ./outputs outputs --repo-type model
  "
