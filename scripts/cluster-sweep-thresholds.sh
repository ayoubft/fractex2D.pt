#!/bin/bash -l
#
# SLURM job: find the threshold that maximises F1 / IoU for a trained run.
#
# The network is run ONCE over the split and every threshold is scored from the
# same predictions, so this costs one inference pass regardless of grid size --
# do not loop infere.py over thresholds, that repeats identical forward passes.
#
# Usage:
#   sbatch scripts/cluster-sweep-thresholds.sh <run_dir> [split] [start stop step]
#
# Examples:
#   sbatch scripts/cluster-sweep-thresholds.sh outputs_BM_/sam2_tiny_dicebce/2026-09-18_10-30
#   sbatch scripts/cluster-sweep-thresholds.sh outputs_BM_/sam2_large/2026-09-19_02-11 val
#   sbatch scripts/cluster-sweep-thresholds.sh outputs_BM_/unet/2026-01-29_09-54 test 0.01 0.99 0.01
#
# Results land in <run_dir>/threshold_sweep_<split>_model.{csv,json}

#SBATCH --nodes 1
#SBATCH --mem 32G

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding

#SBATCH --time 2:00:00

set -euo pipefail

RUN_DIR="${1:?usage: $0 <run_dir> [split] [start stop step]}"
SPLIT="${2:-test}"
START="${3:-0.05}"
STOP="${4:-0.95}"
STEP="${5:-0.05}"

# Set up my modules
module purge
module load gcc cuda python

source path_to_venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python infere.py "$RUN_DIR" \
  --sweep-thresholds \
  --split "$SPLIT" \
  --threshold-grid "$START" "$STOP" "$STEP"
