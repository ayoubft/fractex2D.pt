#!/bin/bash -l
# GPU job (E6): params, FLOPs, latency and peak memory, all models on one A100.
#   sbatch scripts/review/cost.sh TAG=RUN_DIR [TAG=RUN_DIR ...] [--override TAG:key=val ...]
#SBATCH --job-name review-cost
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 1:00:00
#SBATCH --output review_out/logs/%x-%j.out
set -eo pipefail
source scripts/review/env.sh
python -m review.compute_cost "$@"
