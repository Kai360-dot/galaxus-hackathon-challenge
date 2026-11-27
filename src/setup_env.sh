#!/usr/bin/env bash
# Python environment for Apertus v1.5 on a CUDA 12.8 Linux box (tested: Ubuntu 24.04 in WSL, RTX 3080 Ti).
# Apertus 1.5 is not in upstream transformers yet, so this pins the swiss-ai fork from the model card.
set -euo pipefail
ENV_DIR=${1:-.venv}
python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install -q uv
UV="$ENV_DIR/bin/uv"
PY="$ENV_DIR/bin/python"
# torchaudio must come from the same CUDA index as torch: the PyPI build wants CUDA 13 (libcudart.so.13).
$UV pip install --python "$PY" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
$UV pip install --python "$PY" \
  "transformers[torch,vision,audio] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4" \
  accelerate bitsandbytes "pandas>=2.1" pillow tqdm huggingface_hub kaggle==2.2.5 six py7zr  # the kaggle CLI imports six without declaring it
$UV pip install --python "$PY" torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -c "import torch, transformers; from transformers import AutoModelForMultimodalLM; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
