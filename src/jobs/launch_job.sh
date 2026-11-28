#!/usr/bin/env bash
# Launch a job script of src/jobs on a Hugging Face Job. Run where you are logged in with `hf auth login` and
# `kaggle auth login`. The commit must be pushed: the job clones it from GitHub.
#
#     JOB=lora_job.sh EPOCHS=1 src/jobs/launch_job.sh <private-results-repo> [flavor] [image-size]
#     JOB=heads_job.sh src/jobs/launch_job.sh your-user/galaxus-work l40sx1 512
#     JOB=lora_classifier_job.sh MINUTES=100 src/jobs/launch_job.sh your-user/galaxus-work h200 512
#
# MODE, MINUTES, RUN_NAME and PARTS are passed on to the job when set. FINAL_PASSPHRASE, when set, goes to the job as a
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

# The code of this commit also goes to the private results repo. A job script can take it from there (as
# lora_classifier_job.sh does), so it runs while the GitHub repository is private.
archive=$(mktemp)
git archive --format=tar.gz --prefix=repo/ "$COMMIT" src > "$archive"
"$PYTHON" - "$RESULTS_REPO" "$archive" "$COMMIT" <<'PY'
import sys
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=sys.argv[2], path_in_repo=f"code/{sys.argv[3]}.tar.gz", repo_id=sys.argv[1],
                    repo_type="dataset")
print(f"code of {sys.argv[3][:7]} uploaded")
PY

# Secrets go through a file readable only by us, removed right after the launch.
secrets=$(mktemp)
chmod 600 "$secrets"
trap 'rm -f "$secrets" "$archive"' EXIT
{
  echo "HF_TOKEN=$(cat ~/.cache/huggingface/token)"
  echo "KAGGLE_CREDENTIALS_B64=$(base64 < ~/.kaggle/credentials.json | tr -d '\n')"  # no -w0: macOS lacks it
  if [ -n "${FINAL_PASSPHRASE:-}" ]; then echo "FINAL_PASSPHRASE=$FINAL_PASSPHRASE"; fi
} > "$secrets"
grep -q "^KAGGLE_CREDENTIALS_B64=." "$secrets" || { echo "the Kaggle credentials could not be read"; exit 1; }

optional=()
for name in MODE MINUTES RUN_NAME PARTS WEIGHTS_REPO; do
  if [ -n "${!name+set}" ]; then optional+=(--env "$name=${!name}"); fi
done

"$HF" jobs run --detach --flavor "$FLAVOR" --timeout "$TIMEOUT" --name "galaxus-${JOB%_job.sh}-$IMAGE_SIZE${PROMPT:+-$PROMPT}${MODE:+-$MODE}" \
  --env COMMIT="$COMMIT" --env RESULTS_REPO="$RESULTS_REPO" --env IMAGE_SIZE="$IMAGE_SIZE" --env EPOCHS="$EPOCHS" --env PROMPT="${PROMPT:-vague}" --env EXTRA_ARGS="${EXTRA_ARGS:-}" \
  ${optional[@]+"${optional[@]}"} \
  --secrets-file "$secrets" \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel \
  bash -c "pip install -q huggingface_hub && python -c \"import tarfile; from huggingface_hub import hf_hub_download; \
tarfile.open(hf_hub_download('$RESULTS_REPO', 'code/$COMMIT.tar.gz', repo_type='dataset')).extractall('/code')\" \
&& bash /code/repo/src/jobs/$JOB"
