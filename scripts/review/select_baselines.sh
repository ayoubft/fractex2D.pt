#!/bin/bash -l
# GPU job (P1): validation loss of every FraXet-config baseline checkpoint under each
# DEM preprocessing variant, deterministic eval. Test is not touched.
#   sbatch scripts/review/select_baselines.sh
#SBATCH --job-name review-select
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 8
#SBATCH --mem 48G
#SBATCH --time 6:00:00
#SBATCH --output review_out/logs/%x-%j.out
set -eo pipefail
source scripts/review/env.sh
UNET=(2025-08-14_16-42 2025-08-14_16-43 2025-08-24_02-25 2025-11-03_17-38 2025-11-04_08-45 2025-12-02_10-59)
for r in "${UNET[@]}"; do
  for m in legacy raw zscore minmax; do
    python -m review.predict "outputs_BM_/unet-huber-rmspro-0.1/$r" --tag "sel_unet_${r}_${m}" \
      --splits val --no-save --out-root review_out/select "++dataset.dem_mode=$m" \
      || echo "CANDIDATE FAILED: unet $r $m"
  done
done
for m in legacy raw zscore minmax; do
  python -m review.predict "multirun_BM_/segformer/2025-08-31_10-53/model=sm_segformer" \
    --tag "sel_segformer_2025-08-31_10-53_${m}" --splits val --no-save --out-root review_out/select "++dataset.dem_mode=$m" \
    || echo "CANDIDATE FAILED: segformer $m"
done
python -m review.tb_val_loss outputs_BM_/unet-huber-rmspro-0.1/*/ "multirun_BM_/segformer/2025-08-31_10-53/model=sm_segformer/" \
  > review_out/select/tensorboard_min_val_loss.txt || true
