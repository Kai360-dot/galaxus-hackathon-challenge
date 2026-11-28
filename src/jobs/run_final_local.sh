#!/usr/bin/env bash
# Everything for the final images that runs on the local GPU (RTX 3080 Ti), once the passphrase is out:
# decrypt, DINOv2 448 (image + mirror), pixel measures (plain and mirrored), and as a fallback the Apertus features
# in 4-bit. The bf16 Apertus features come from src/jobs/final_job.sh on a Hugging Face Job.
#
#     FINAL_PASSPHRASE='…' bash src/jobs/run_final_local.sh ~/galaxus
set -euo pipefail
ROOT=${1:-$HOME/galaxus}
: "${FINAL_PASSPHRASE:?export FINAL_PASSPHRASE first}"
cd "$ROOT"
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "decrypt"
if [ ! -f data/final/final.csv ]; then
  $PY - <<'PY'
import os, py7zr
with py7zr.SevenZipFile("data/final_encrypted.bin", password=os.environ["FINAL_PASSPHRASE"]) as archive:
    archive.extractall("data/final")
PY
  [ -f data/final/final.csv ] || { f=$(find data/final -name final.csv | head -1); d=$(dirname "$f"); ln -s "$PWD/$d"/* data/final/; }
fi
wc -l data/final/final.csv

step "DINOv2 448, image and mirror"
$PY src/dino.py --data-dir data --features-dir results/dino_448_mirrored --parts final --size 448 --mirror-all 2>&1 | grep -v "it/s\]" | tail -2
step "pixel measures, plain and mirrored"
$PY src/pixel_features.py --data-dir data --features-dir results/pixels --parts final 2>&1 | tail -1
$PY src/pixel_features.py --data-dir data --features-dir results/pixels_mirrored --parts final --mirror-all 2>&1 | tail -1
step "finetuned DINOv2 outputs (both halves)"
for h in 0 1; do
  $PY src/dino_finetune.py predict --data-dir data --output-dir runs/dino_multi_h$h --size 448 --trainable-blocks 12 --batch 24 \
    --dimensions perspective subject text_classification --parts final 2>&1 | grep -E "images in|Error" || true
done
step "fallback: Apertus features in 4-bit (about 50 minutes)"
$PY src/heads.py extract --data-dir data --features-dir results/heads_512_4bit/features --parts final \
  --quantization 4bit --layers 24 --truncate --prompt vague 2>&1 | grep --line-buffered -E "images/s|Error|loaded"
step "done"
