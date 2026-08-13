#!/usr/bin/env bash
# Rebuild the ME344 final-project workspace on a fresh tpu-student49.
# Mirrors the README install block exactly (same MaxText SHA, same two venvs).
set -euo pipefail

export PROJECT_ID=soe-hpccenter
export SUNET=qanh
export ZONE=us-west4-a
export ASSIGNMENT_URI=gs://me344-tpu-labs-west4/data/me344_final_project
export BASE_CHECKPOINT=gs://me344-tpu-labs-west4/data/qwen3-4b-instruct-2507/0/items
export UPLOAD_BUCKET=gs://me344-tpu-labs-west4/final_projects
export ME344_STUDENT_ID="$SUNET"
export ME344_OUTPUT="${UPLOAD_BUCKET}/${SUNET}/runs"
export ME344_SUBMISSION_URI="${UPLOAD_BUCKET}/${SUNET}/submission"
export ME344_BASE_CHECKPOINT="$BASE_CHECKPOINT"
export ME344_TPU_ZONE="$ZONE"
export WANDB_MODE=offline

env | grep -E '^(PROJECT_ID|SUNET|ZONE|ASSIGNMENT_URI|BASE_CHECKPOINT|UPLOAD_BUCKET|TPU_NAME|ME344_[A-Z_]+|WANDB_MODE)=' \
  | sed 's/^/export /' > "$HOME/.me344.env"

echo "=== [1/6] fetch assignment $(date -u +%H:%M:%SZ) ==="
mkdir -p "$HOME/me344_final_project"
# The TPU VM's gcloud predates `gcloud storage rsync`; the README uses cp here.
gcloud storage cp --recursive "$ASSIGNMENT_URI/*" "$HOME/me344_final_project/" \
  || gsutil -m cp -r "$ASSIGNMENT_URI/*" "$HOME/me344_final_project/"
cd "$HOME/me344_final_project"

echo "=== [2/6] uv + THP $(date -u +%H:%M:%SZ) ==="
if [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env"
sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'

export MAXTEXT_SHA=17c7172720ca813b05e5ea248dedd78a0c64612e
export MAXTEXT_URL="git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_SHA}"

echo "=== [3/6] pretrain venv $(date -u +%H:%M:%SZ) ==="
uv venv --python 3.12 --seed .venv-me344-pretrain
source .venv-me344-pretrain/bin/activate
uv pip install "maxtext[tpu] @ ${MAXTEXT_URL}" --resolution=lowest
install_tpu_pre_train_extra_deps
uv pip install "xprof==2.23.0"
deactivate

echo "=== [4/6] posttrain venv $(date -u +%H:%M:%SZ) ==="
uv venv --python 3.12 --seed .venv-me344-posttrain
source .venv-me344-posttrain/bin/activate
UV_TORCH_BACKEND=cpu uv pip install \
  "maxtext[tpu-post-train] @ ${MAXTEXT_URL}" --resolution=lowest
install_tpu_post_train_extra_deps
uv pip install jupyterlab ipykernel wandb
python -m ipykernel install --user --name me344-posttrain \
  --display-name "ME344 MaxText"
deactivate

echo "=== [5/6] verify both envs see 8 chips $(date -u +%H:%M:%SZ) ==="
for v in pretrain posttrain; do
  JAX_PLATFORMS=tpu ".venv-me344-${v}/bin/python" -c \
    "import jax; print('${v}', jax.default_backend(), jax.device_count())"
done

echo "=== [6/6] BOOTSTRAP_COMPLETE $(date -u +%H:%M:%SZ) ==="
touch "$HOME/.me344-bootstrap-complete"
