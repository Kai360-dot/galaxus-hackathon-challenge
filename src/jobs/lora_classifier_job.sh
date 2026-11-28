#!/usr/bin/env bash
# LoRA and heads trained together from scratch (src/lora_classifier.py) on a Hugging Face Job with one GPU.
#
# MODE=train (default): setup, Kaggle data, a trial run of two minutes on a few images that goes through every
#   step, then the real run of MINUTES minutes with a holdout score and an uploaded checkpoint every 20 minutes,
#   then holdout and dev predictions. Everything goes to RESULTS_REPO/RUN_NAME (private).
# MODE=trial: only the trial run, to test the code on a small GPU for a few cents.
# MODE=predict: the checkpoint of RUN_NAME from RESULTS_REPO predicts dev and, when the secret FINAL_PASSPHRASE is
#   set, the final images. Predictions and the submission file go to RESULTS_REPO/RUN_NAME/final_run.
#
# Environment: COMMIT, RESULTS_REPO, MODE, IMAGE_SIZE (default 512), MINUTES (default 100), RUN_NAME, EXTRA_ARGS.
# Secrets: HF_TOKEN, KAGGLE_CREDENTIALS_B64, FINAL_PASSPHRASE (optional). Launch with
# JOB=lora_classifier_job.sh src/jobs/launch_job.sh.
set -euo pipefail
REPO_URL=https://github.com/Kai360-dot/galaxus-hackathon-challenge.git
: "${COMMIT:?}" "${RESULTS_REPO:?}" "${HF_TOKEN:?}" "${KAGGLE_CREDENTIALS_B64:?}"
MODE=${MODE:-train}
IMAGE_SIZE=${IMAGE_SIZE:-512}
MINUTES=${MINUTES:-100}
RUN_NAME=${RUN_NAME:-lora_classifier_${IMAGE_SIZE}}
WORK=/work
RUN="$WORK/$RUN_NAME"
LOGS="$WORK/logs"
mkdir -p "$WORK" "$RUN" "$LOGS" && cd "$WORK"
step() { echo "[$(date -u +%H:%M:%S)] $*"; }
quiet() { grep --line-buffered -vE "it/s\]|xIELU"; }

upload_logs() {
  step "upload logs to $RESULTS_REPO/$RUN_NAME/logs_$MODE (private)"
  python - <<PY || true
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="$RESULTS_REPO", repo_type="dataset", folder_path="$LOGS", path_in_repo="$RUN_NAME/logs_$MODE")
PY
}
trap upload_logs EXIT

step "setup"
if [ -d /code/repo/src ]; then  # the launcher put the code of this commit there
  cp -r /code/repo repo
else
  command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
  git clone -q "$REPO_URL" repo && git -C repo checkout -q "$COMMIT"
fi
pip install -q uv
uv pip install --system -q \
  "transformers[torch,vision,audio] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4" \
  accelerate peft "pandas>=2.1" pillow tqdm huggingface_hub kaggle==2.2.5 six py7zr
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

cd repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # images differ in size: avoid fragmentation

if [ "$MODE" = train ] || [ "$MODE" = trial ]; then
  step "trial run on a few images"
  python src/lora_classifier.py train --data-dir "$WORK/data" --output-dir "$WORK/trial" --image-size "$IMAGE_SIZE" \
    --minutes 2 --evaluate-minutes 1 --limit 160 --holdout-limit 48 --predict-limit 48 --calibration-images 64 \
    --log-every 2 ${EXTRA_ARGS:-} 2>&1 | quiet | tee "$LOGS/trial.log"
fi

if [ "$MODE" = trial ]; then
  step "trial only"
elif [ "$MODE" = train ]; then
  step "train $RUN_NAME for $MINUTES minutes"
  python src/lora_classifier.py train --data-dir "$WORK/data" --output-dir "$RUN" --image-size "$IMAGE_SIZE" \
    --minutes "$MINUTES" --results-repo "$RESULTS_REPO" --run-name "$RUN_NAME" ${EXTRA_ARGS:-} 2>&1 \
    | quiet | tee "$LOGS/train.log"
else
  FINAL_CSV=()
  if [ -n "${FINAL_PASSPHRASE:-}" ]; then
    step "unpack the final images"
    python - <<'PY'
import os

import py7zr

with py7zr.SevenZipFile("/work/data/final_encrypted.bin", password=os.environ["FINAL_PASSPHRASE"]) as archive:
    archive.extractall("/work/data/final")
PY
    FINAL_CSV=(--csv "$(find "$WORK/data/final" -name final.csv | head -1)")
    echo "final csv: ${FINAL_CSV[1]}, $(find "$WORK/data/final" -type f | wc -l) files"
  fi

  step "checkpoint of $RUN_NAME"
  hf download "$RESULTS_REPO" --repo-type dataset --include "$RUN_NAME/checkpoint/*" --local-dir "$WORK/results" >/dev/null

  step "predict"
  python src/lora_classifier.py predict --data-dir "$WORK/data" --checkpoint "$WORK/results/$RUN_NAME/checkpoint" \
    --output-dir "$RUN/final_run" --parts dev "${FINAL_CSV[@]}" --results-repo "$RESULTS_REPO" \
    --run-name "$RUN_NAME/final_run" ${EXTRA_ARGS:-} 2>&1 | quiet | tee "$LOGS/predict.log"
fi
step "done"
