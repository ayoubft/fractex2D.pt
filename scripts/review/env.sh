# Sourced by the review sbatch scripts. VENV defaults to the curnagl venv.
module purge
dcsrsoft use 20240303
module load gcc cuda python
source "${VENV:-$HOME/work-detect/venv_pytorch_gpu}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
