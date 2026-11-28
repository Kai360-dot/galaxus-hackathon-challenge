#!/usr/bin/env bash
# LoRA and heads trained together (src/lora_heads.py) on a Hugging Face Job: setup, Kaggle data, helper features
# and initial heads from the private results repo, training with a holdout score and an uploaded checkpoint every
# half epoch. A stopped job keeps the checkpoints it uploaded.
#
# Environment: COMMIT, RESULTS_REPO, EPOCHS (default 2), EXTRA_ARGS (optional).
# Secrets: HF_TOKEN, KAGGLE_CREDENTIALS_B64. Launch with JOB=lora_heads_job.sh src/jobs/launch_job.sh.
set -euo pipefail
REPO_URL=https://github.com/Kai360-dot/galaxus-hackathon-challenge.git
: "${COMMIT:?}" "${RESULTS_REPO:?}" "${HF_TOKEN:?}" "${KAGGLE_CREDENTIALS_B64:?}"
EPOCHS=${EPOCHS:-2}
RUN_NAME="lora_heads_e${EPOCHS}"
WORK=/work
mkdir -p "$WORK" && cd "$WORK"
step() { echo "[$(date -u +%H:%M:%S)] $*"; }

step "setup"
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
git clone -q "$REPO_URL" repo && git -C repo checkout -q "$COMMIT"
pip install -q uv
uv pip install --system -q \
  "transformers[torch,vision,audio] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4" \
  accelerate peft "pandas>=2.1" pillow tqdm huggingface_hub kaggle==2.2.5 six
uv pip install --system -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

step "kaggle data and helper features"
mkdir -p ~/.kaggle && echo "$KAGGLE_CREDENTIALS_B64" | base64 -d > ~/.kaggle/credentials.json
chmod 600 ~/.kaggle/credentials.json
mkdir -p data
kaggle competitions download -c hackapertus-galaxus -p data >/dev/null
python -c "import zipfile; zipfile.ZipFile('data/hackapertus-galaxus.zip').extractall('data')"
rm data/hackapertus-galaxus.zip
hf download "$RESULTS_REPO" --repo-type dataset --include "helpers/*" --local-dir "$WORK" >/dev/null
ls "$WORK/helpers"

step "train $RUN_NAME"
cd repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # images differ in size: avoid fragmentation
python src/lora_heads.py train --data-dir "$WORK/data" \
  --extras "$WORK/helpers/dino_448_mirrored" "$WORK/helpers/pixels_mirrored" --init "$WORK/helpers/init_heads.pt" \
  --output-dir "$WORK/$RUN_NAME" --epochs "$EPOCHS" --results-repo "$RESULTS_REPO" --run-name "$RUN_NAME" \
  ${EXTRA_ARGS:-} 2>&1 | grep --line-buffered -vE "it/s\]|xIELU"
step "done"
