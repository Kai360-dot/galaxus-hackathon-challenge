#!/usr/bin/env bash
# LoRA finetune on the atomic questions (src/lora_atomic.py) on a Hugging Face Job (one GPU): setup, Kaggle data,
# training, holdout score, dev predictions. The adapter and outputs go to a private HF dataset repo, also when the job
# fails or hits its timeout (whatever was saved last).
#
# Environment: COMMIT, RESULTS_REPO, IMAGE_SIZE (default 512), EPOCHS (default 1), EXTRA_ARGS (optional).
# Secrets: HF_TOKEN, KAGGLE_CREDENTIALS_B64. Launch with src/jobs/launch_job.sh.
set -euo pipefail
REPO_URL=https://github.com/Kai360-dot/galaxus-hackathon-challenge.git
: "${COMMIT:?}" "${RESULTS_REPO:?}" "${HF_TOKEN:?}" "${KAGGLE_CREDENTIALS_B64:?}"
IMAGE_SIZE=${IMAGE_SIZE:-512}
EPOCHS=${EPOCHS:-1}
RUN_NAME="lora_atomic_${IMAGE_SIZE}_e${EPOCHS}"
WORK=/work
RUN="$WORK/$RUN_NAME"
mkdir -p "$WORK" "$RUN" && cd "$WORK"
step() { echo "[$(date -u +%H:%M:%S)] $*"; }

upload() {
  step "upload $RUN to $RESULTS_REPO/$RUN_NAME (private)"
  python - <<PY || true
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="$RESULTS_REPO", repo_type="dataset", folder_path="$RUN", path_in_repo="$RUN_NAME")
PY
}
trap upload EXIT

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

step "kaggle data"
mkdir -p ~/.kaggle && echo "$KAGGLE_CREDENTIALS_B64" | base64 -d > ~/.kaggle/credentials.json
chmod 600 ~/.kaggle/credentials.json
mkdir -p data
kaggle competitions download -c hackapertus-galaxus -p data >/dev/null
python -c "import zipfile; zipfile.ZipFile('data/hackapertus-galaxus.zip').extractall('data')"
rm data/hackapertus-galaxus.zip

step "train $RUN_NAME"
cd repo
python src/lora_atomic.py train --data-dir "$WORK/data" --output-dir "$RUN" --image-size "$IMAGE_SIZE" \
  --epochs "$EPOCHS" --workers 6 --parts holdout dev ${EXTRA_ARGS:-} 2>&1 \
  | grep --line-buffered -vE "it/s\]" | tee "$RUN/train.log"
step "done"
