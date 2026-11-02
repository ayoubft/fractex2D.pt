#!/bin/bash -l
# GPU job (P2): one properly seeded training run, identical to the paper's SAM 2 run
# (dice_bce, AdamW 1e-4, batch 8, 111 epochs, RGB, constant LR, val-loss checkpoint)
# except for seed_everything=true and deterministic val/test. Then predict + score.
#   sbatch scripts/review/train.sh <model_cfg> <seed> <tag> [extra hydra overrides...]
#   e.g. sbatch scripts/review/train.sh sam2_seg_tiny 0 sam2_tiny_s0
#        sbatch scripts/review/train.sh hiera_tiny_scratch 0 hiera_scratch_s0
#SBATCH --job-name review-train
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 24:00:00
#SBATCH --output review_out/logs/%x-%j.out
set -eo pipefail
source scripts/review/env.sh
MODEL="$1"; SEED="$2"; TAG="$3"; shift 3
RUN_DIR="outputs_review/$TAG"
python main.py model="$MODEL" loss=dice_bce optimizer=adamw in_channels=3 batch_size=8 \
  seed="$SEED" seed_everything=true dataset.deterministic_eval=true \
  hydra.run.dir="$RUN_DIR" hydra.job.name="$TAG" "$@"
python -m review.predict "$RUN_DIR" --tag "$TAG" --checkpoint model.pt
python -m review.evaluate "review_out/preds/$TAG"
