#!/usr/bin/env bash
# DINOv2-large finetuned for the perspective (src/dino_finetune.py) on a Hugging Face Job: setup, Kaggle data,
# training, then perspective probabilities for the holdout, dev and, when the secret FINAL_PASSPHRASE is set, the
# final images. Probabilities go to RESULTS_REPO/RUN_NAME (private), the weights to WEIGHTS_REPO when it is set.
#
# Environment: COMMIT, RESULTS_REPO, IMAGE_SIZE (default 448), RUN_NAME, WEIGHTS_REPO (optional), EXTRA_ARGS.
# Secrets: HF_TOKEN, KAGGLE_CREDENTIALS_B64, FINAL_PASSPHRASE (optional). Launch with
# JOB=dino_perspective_job.sh src/jobs/launch_job.sh <results-repo> h200 448.
set -euo pipefail
: "${COMMIT:?}" "${RESULTS_REPO:?}" "${HF_TOKEN:?}" "${KAGGLE_CREDENTIALS_B64:?}"
SIZE=${IMAGE_SIZE:-448}
RUN_NAME=${RUN_NAME:-dino_perspective_${SIZE}}
WORK=/work
RUN="$WORK/$RUN_NAME"
mkdir -p "$RUN" && cd "$WORK"
step() { echo "[$(date -u +%H:%M:%S)] $*"; }

step "setup"
cp -r /code/repo repo
pip install -q uv
uv pip install --system -q "transformers>=4.45,<5" "pandas>=2.1" pillow tqdm huggingface_hub kaggle==2.2.5 six py7zr
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

step "kaggle data"
mkdir -p ~/.kaggle && echo "$KAGGLE_CREDENTIALS_B64" | base64 -d > ~/.kaggle/credentials.json
chmod 600 ~/.kaggle/credentials.json
mkdir -p data
kaggle competitions download -c hackapertus-galaxus -p data >/dev/null
python -c "import zipfile; zipfile.ZipFile('data/hackapertus-galaxus.zip').extractall('data')"
rm data/hackapertus-galaxus.zip

FINAL_CSV=()
if [ -n "${FINAL_PASSPHRASE:-}" ]; then
  step "unpack the final images"
  (apt-get update -qq && apt-get install -y -qq p7zip-full) >/dev/null 2>&1 || true
  if ! { command -v 7z >/dev/null && 7z x -y -p"$FINAL_PASSPHRASE" -o"$WORK/data/final" "$WORK/data/final_encrypted.bin" >/dev/null; }; then
    rm -rf "$WORK/data/final"
    python - <<'PY'
import os

import py7zr

with py7zr.SevenZipFile("/work/data/final_encrypted.bin", password=os.environ["FINAL_PASSPHRASE"]) as archive:
    archive.extractall("/work/data/final")
PY
  fi
  FINAL_CSV=(--csv "$(find "$WORK/data/final" -name final.csv | head -1)" --name final)
fi

cd repo
step "train"
python src/dino_finetune.py train --data-dir "$WORK/data" --output-dir "$RUN" --size "$SIZE" --workers 20 \
  ${EXTRA_ARGS:-} 2>&1 | tee "$RUN/train.log"

step "predict"
python src/dino_finetune.py predict --data-dir "$WORK/data" --output-dir "$RUN" --size "$SIZE" --workers 20 \
  --batch 64 --parts holdout dev "${FINAL_CSV[@]}" 2>&1 | tee "$RUN/predict.log"

step "upload"
python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(repo_id="$RESULTS_REPO", repo_type="dataset", folder_path="$RUN", path_in_repo="$RUN_NAME",
                  ignore_patterns=["best.pt"])
weights = os.environ.get("WEIGHTS_REPO")
if weights:
    api.upload_file(path_or_fileobj="$RUN/best.pt", path_in_repo="$RUN_NAME/best.pt", repo_id=weights, repo_type="model",
                    commit_message="DINOv2-large finetuned for the perspective")
    print("weights uploaded to", weights)
PY
step "done"
