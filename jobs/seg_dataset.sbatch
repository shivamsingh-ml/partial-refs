#!/usr/bin/env bash
#SBATCH --job-name=pr-seg-dataset
#SBATCH --partition=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Build the nnU-Net training set for retraining V2V2R's segmentation network (2026-11-16): 424 training scans
# (development + TF3-only, S0015 excluded), label fixes of configs/seg/label_fixes_v1.json, V2V2R's conversion script
# unchanged, every scan verified voxel by voxel. See partialref.baseline.seg_dataset. No GPU, no training.
#   sbatch jobs/seg_dataset.sbatch
set -euo pipefail
cd /msc/home/ssingh54/partial-refs
SHA=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain -- src jobs configs)" ] && echo "-dirty" || true)
RUN=results/seg/$(date +%Y%m%d-%H%M)-${SHA}-seg-dataset-${SLURM_JOB_ID}
echo "node=$(hostname) job=$SLURM_JOB_ID run=$RUN start=$(date -Is)"
OMP_NUM_THREADS=1 PYTHONHASHSEED=0 ~/miniconda3/envs/cbct_partialref/bin/python -m partialref.baseline.seg_dataset \
  --tf3-dir /msc/home/ssingh54/sae_tooth/data/raw/toothfairy3 --splits configs/splits/v1 \
  --fixes configs/seg/label_fixes_v1.json --umamba2-dir /msc/home/ssingh54/U-Mamba2 \
  --raw-dir data/nnunet/raw --dataset Dataset713_TF3SinglePulpDev --out "$RUN" --np 4
echo "done=$(date -Is)"
