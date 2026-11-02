#!/bin/bash -l
# GPU job: deterministic val/test predictions for one checkpoint, then score them.
#   sbatch scripts/review/predict.sh <run_dir> <tag> [checkpoint] [hydra overrides...]
#SBATCH --job-name review-predict
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 8
#SBATCH --mem 48G
#SBATCH --time 3:00:00
#SBATCH --output review_out/logs/%x-%j.out
set -eo pipefail
source scripts/review/env.sh
RUN_DIR="$1"; TAG="$2"; CKPT="${3:-model.pt}"; shift 3 || shift $#
python -m review.predict "$RUN_DIR" --tag "$TAG" --checkpoint "$CKPT" "$@"
python -m review.evaluate "review_out/preds/$TAG"
