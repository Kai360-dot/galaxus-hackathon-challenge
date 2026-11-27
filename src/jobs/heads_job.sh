#!/usr/bin/env bash
# The heads pipeline end to end on a Hugging Face Job (one GPU): setup, Kaggle data, feature extraction in bf16,
# heads, dev submission. Features and outputs go to a private HF dataset repo, never to this repository.
#
# Environment: COMMIT (of this repository), RESULTS_REPO (private dataset repo), IMAGE_SIZE (default 512).
# Secrets: HF_TOKEN (gated Apertus access, write access to RESULTS_REPO), KAGGLE_CREDENTIALS_B64
# (base64 of ~/.kaggle/credentials.json). Launch with src/jobs/launch_heads_job.sh.
set -euo pipefail
REPO_URL=https://github.com/Kai360-dot/galaxus-hackathon-challenge.git
: "${COMMIT:?}" "${RESULTS_REPO:?}" "${HF_TOKEN:?}" "${KAGGLE_CREDENTIALS_B64:?}"
IMAGE_SIZE=${IMAGE_SIZE:-512}
WORK=/work
mkdir -p "$WORK" && cd "$WORK"
step() { echo "[$(date -u +%H:%M:%S)] $*"; }

step "setup"
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
git clone -q "$REPO_URL" repo && git -C repo checkout -q "$COMMIT"
pip install -q uv
uv pip install --system -q \
  "transformers[torch,vision,audio] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4" \
  accelerate "pandas>=2.1" pillow tqdm huggingface_hub kaggle==2.2.5 six
# The audio extra may pull a torchaudio for another CUDA; keep torch, torchvision and torchaudio on cu128.
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
ls data

step "extract features"
cd repo
python src/heads.py extract --data-dir "$WORK/data" --features-dir "$WORK/features" \
  --quantization none --image-size "$IMAGE_SIZE" 2>&1 | grep --line-buffered -v "it/s\]"

step "fit heads"
mkdir -p "$WORK/outputs"
python src/heads.py fit --data-dir "$WORK/data" --features-dir "$WORK/features" --output-dir "$WORK/outputs" \
  --hidden 0 1024 2>&1 | tee "$WORK/outputs/fit.log"

step "upload features and outputs to $RESULTS_REPO (private)"
python - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(repo_id="$RESULTS_REPO", repo_type="dataset", folder_path="$WORK/outputs",
                  path_in_repo="heads_${IMAGE_SIZE}/outputs")
api.upload_folder(repo_id="$RESULTS_REPO", repo_type="dataset", folder_path="$WORK/features",
                  path_in_repo="heads_${IMAGE_SIZE}/features")
PY
step "done"
