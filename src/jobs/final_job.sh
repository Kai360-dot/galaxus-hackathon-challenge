#!/usr/bin/env bash
# Apertus features of the final images on a Hugging Face Job (one GPU, bf16): download only final_encrypted.bin,
# decrypt it with the passphrase, extract layer 24 (last token + image mean) with the vague prompt and upload the
# shards to a private HF dataset repo. Inference only: nothing is trained on the final images.
#
# Environment: COMMIT (of this repository), RESULTS_REPO (private dataset repo of the account that runs the job).
# Secrets: HF_TOKEN (Apertus access, write access to RESULTS_REPO), KAGGLE_CREDENTIALS_B64, FINAL_PASSPHRASE.
#
#     JOB=final_job.sh TIMEOUT=2h src/jobs/launch_job.sh <user>/galaxus-final h200   (with FINAL_PASSPHRASE exported)
set -euo pipefail
REPO_URL=https://github.com/Kai360-dot/galaxus-hackathon-challenge.git
: "${COMMIT:?}" "${RESULTS_REPO:?}" "${HF_TOKEN:?}" "${KAGGLE_CREDENTIALS_B64:?}" "${FINAL_PASSPHRASE:?}"
WORK=/work
mkdir -p "$WORK" && cd "$WORK"
step() { echo "[$(date -u +%H:%M:%S)] $*"; }

step "setup"
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
git clone -q "$REPO_URL" repo && git -C repo checkout -q "$COMMIT"
pip install -q uv
uv pip install --system -q \
  "transformers[torch,vision,audio] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4" \
  accelerate "pandas>=2.1" pillow tqdm huggingface_hub kaggle==2.2.5 six py7zr
uv pip install --system -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

step "kaggle data (csv files and the final archive)"
mkdir -p ~/.kaggle && echo "$KAGGLE_CREDENTIALS_B64" | base64 -d > ~/.kaggle/credentials.json
chmod 600 ~/.kaggle/credentials.json
mkdir -p data
for f in train.csv dev.csv sample_submission.csv final_encrypted.bin; do
  kaggle competitions download -c hackapertus-galaxus -f "$f" -p data >/dev/null
done
for z in data/*.zip; do [ -e "$z" ] && python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('data')" "$z" && rm "$z"; done
python - <<'PY'
import os, py7zr
with py7zr.SevenZipFile("data/final_encrypted.bin", password=os.environ["FINAL_PASSPHRASE"]) as archive:
    archive.extractall("data/final")
PY
ls data data/final | head
# final.csv may sit one folder deeper inside the archive: link it to data/final/final.csv
[ -f data/final/final.csv ] || { f=$(find data/final -name final.csv | head -1); d=$(dirname "$f"); ln -s "$PWD/$d"/* data/final/; }
wc -l data/final/final.csv

step "extract features of the final images (bf16, layer 24)"
cd repo
python src/heads.py extract --data-dir "$WORK/data" --features-dir "$WORK/features" --parts final \
  --quantization none --image-size 512 --prompt vague --layers 24 --truncate 2>&1 | grep --line-buffered -v "it/s\]"

step "upload to $RESULTS_REPO (private)"
python - <<PY
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="$RESULTS_REPO", repo_type="dataset", folder_path="$WORK/features/final",
                      path_in_repo="heads_512_final/features/final")
PY
step "done"
