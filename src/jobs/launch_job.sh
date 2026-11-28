#!/usr/bin/env bash
# Launch a job script of src/jobs on a Hugging Face Job. Run where you are logged in with `hf auth login` and
# `kaggle auth login`. The commit must be pushed: the job clones it from GitHub.
#
#     JOB=lora_job.sh EPOCHS=1 src/jobs/launch_job.sh <private-results-repo> [flavor] [image-size]
#     JOB=heads_job.sh src/jobs/launch_job.sh your-user/galaxus-work l40sx1 512
#     JOB=lora_classifier_job.sh MINUTES=100 src/jobs/launch_job.sh your-user/galaxus-work h200 512
#
# MODE, MINUTES and RUN_NAME are passed on to the job when set. FINAL_PASSPHRASE, when set, goes to the job as a
# secret (for the final images; never write it into a file of this repository).
set -euo pipefail
RESULTS_REPO=${1:?private results dataset repo, e.g. your-user/galaxus-work}
FLAVOR=${2:-l40sx1}
IMAGE_SIZE=${3:-512}
JOB=${JOB:-heads_job.sh}
EPOCHS=${EPOCHS:-1}
TIMEOUT=${TIMEOUT:-4h}
HF=${HF:-hf}
PYTHON=${PYTHON:-python3}  # a python with huggingface_hub
if [ -z "${COMMIT:-}" ]; then
  COMMIT=$(git rev-parse HEAD)
  git branch -r --contains "$COMMIT" | grep -q . || { echo "commit $COMMIT is not pushed"; exit 1; }
fi

"$PYTHON" - "$RESULTS_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
HfApi().create_repo(sys.argv[1], repo_type="dataset", private=True, exist_ok=True)
print(f"results repo {sys.argv[1]} (private)")
PY

# Secrets go through a file readable only by us, removed right after the launch.
secrets=$(mktemp)
chmod 600 "$secrets"
trap 'rm -f "$secrets"' EXIT
{
  echo "HF_TOKEN=$(cat ~/.cache/huggingface/token)"
  echo "KAGGLE_CREDENTIALS_B64=$(base64 < ~/.kaggle/credentials.json | tr -d '\n')"  # no -w0: macOS lacks it
  if [ -n "${FINAL_PASSPHRASE:-}" ]; then echo "FINAL_PASSPHRASE=$FINAL_PASSPHRASE"; fi
} > "$secrets"
grep -q "^KAGGLE_CREDENTIALS_B64=." "$secrets" || { echo "the Kaggle credentials could not be read"; exit 1; }

optional=()
for name in MODE MINUTES RUN_NAME; do
  if [ -n "${!name:-}" ]; then optional+=(--env "$name=${!name}"); fi
done

"$HF" jobs run --detach --flavor "$FLAVOR" --timeout "$TIMEOUT" --name "galaxus-${JOB%_job.sh}-$IMAGE_SIZE${PROMPT:+-$PROMPT}${MODE:+-$MODE}" \
  --env COMMIT="$COMMIT" --env RESULTS_REPO="$RESULTS_REPO" --env IMAGE_SIZE="$IMAGE_SIZE" --env EPOCHS="$EPOCHS" --env PROMPT="${PROMPT:-vague}" --env EXTRA_ARGS="${EXTRA_ARGS:-}" \
  ${optional[@]+"${optional[@]}"} \
  --secrets-file "$secrets" \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel \
  bash -c "python -c \"import urllib.request as u; open('/tmp/job.sh', 'wb').write(u.urlopen(\
'https://raw.githubusercontent.com/Kai360-dot/galaxus-hackathon-challenge/$COMMIT/src/jobs/$JOB').read())\" \
&& bash /tmp/job.sh"
