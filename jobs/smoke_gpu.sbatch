#!/usr/bin/env bash
#SBATCH --job-name=pr-smoke-gpu
#SBATCH --partition=gpu
#SBATCH --qos=1g.5gb
#SBATCH --account=uzk
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
# HANDOFF §4.7: GPU smoke test. Smallest MIG slice; nothing is loaded onto the GPU beyond a CUDA init.
# If SIF_PATH is set, torch is checked inside the container (Plan A); otherwise in a conda env
# named by CONDA_ENV_SMOKE (fallback, recorded as such in ENV_REPORT.md).
set -uo pipefail
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
PYCHECK='import torch, sys; print("python", sys.version.split()[0]); print("torch", torch.__version__, "cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
if [[ -n "${SIF_PATH:-}" ]]; then
  echo "--- container: $SIF_PATH"
  apptainer exec --nv "$SIF_PATH" python -c "$PYCHECK"
else
  echo "--- conda env: ${CONDA_ENV_SMOKE:-umamba2} (no SIF_PATH set)"
  set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_SMOKE:-umamba2}"; set -u
  python -c "$PYCHECK"
fi
