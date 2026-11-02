#!/bin/bash -l
# CPU job: self-test the scoring code, then dump val/test targets (all 3 variants).
#   sbatch scripts/review/targets.sh
#SBATCH --job-name review-targets
#SBATCH --partition cpu
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 2:00:00
#SBATCH --output review_out/logs/%x-%j.out
set -eo pipefail
source scripts/review/env.sh
python -m review.selftest
python -m review.dump_targets --out review_out/targets --workers 8
