#!/bin/bash -l
#
# SLURM launcher for the SAM2 benchmark rows.
#
# Kept separate from cluster-run.sh so the published runs stay reproducible:
# SAM2 needs a longer wall time, more memory and a warm Hugging Face cache.
#
# Usage:
#   sbatch scripts/cluster-run-sam2.sh <model> <loss> <batch_size> [dataset] [job_name]
#
# Examples:
#   sbatch scripts/cluster-run-sam2.sh sam2_seg_tiny      dice_bce 8
#   sbatch scripts/cluster-run-sam2.sh sam2_seg_tiny      huber    8
#   sbatch scripts/cluster-run-sam2.sh sam2_seg_base_plus dice_bce 4
#   sbatch scripts/cluster-run-sam2.sh sam2_seg_large     dice_bce 2
#
#   # GeoCrack instead of the samsu19/matteo21/ovaskainen23 benchmark:
#   sbatch scripts/cluster-run-sam2.sh sam2_seg_tiny      dice_bce 8 geocrack
#   sbatch scripts/cluster-run-sam2.sh sam2_seg_base_plus dice_bce 4 geocrack
#   sbatch scripts/cluster-run-sam2.sh sam2_seg_large     dice_bce 2 geocrack
#
# `loss` must be a non-logit loss (dice_bce, not dice_logit): SAM2SegNet.forward
# already returns sigmoid probabilities.
#
# The SAM2 rows run at in_channels=3 (RGB only, no DEM) because SAM2's patch
# embedding is RGB-pretrained -- which also makes GeoCrack, an RGB-only dataset,
# a drop-in swap. Everything else stays at the published defaults in
# config/config.yaml (epochs=111, threshold=0.1); the tile size comes from the
# dataset config (256 for the benchmark, GeoCrack's native 224).

#SBATCH --nodes 1
#SBATCH --mem 64G

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding

#SBATCH --time 24:00:00

set -euo pipefail

USAGE="usage: $0 <model> <loss> <batch_size> [dataset] [job_name]"
MODEL="${1:?$USAGE}"
LOSS="${2:?$USAGE}"
BATCH_SIZE="${3:?$USAGE}"
DATASET="${4:-benchmark}"
JOB_NAME="${5:-${MODEL}_${LOSS}_${DATASET}}"

# Set up my modules
module purge
module load gcc cuda python

source path_to_venv/bin/activate

# SAM2 weights are pulled from the Hugging Face Hub on first use. Compute nodes
# often have no outbound network, so point HF at a shared cache and pre-warm it
# from a login node with:
#   HF_HOME=$HF_HOME python -c "from transformers import Sam2Model; Sam2Model.from_pretrained('facebook/sam2.1-hiera-tiny')"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python main.py \
  dataset="$DATASET" \
  model="$MODEL" \
  loss="$LOSS" \
  optimizer=adamw \
  in_channels=3 \
  batch_size="$BATCH_SIZE" \
  hydra.job.name="$JOB_NAME"
