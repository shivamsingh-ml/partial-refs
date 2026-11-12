#!/usr/bin/env bash
#SBATCH --job-name=pr-seg-check
#SBATCH --partition=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Checks on the ToothFairy3 training scans before retraining V2V2R's segmentation network (2026-11-12):
# test-set leakage, headers (nibabel + SimpleITK), label vs image fit, left/right and up/down consistency,
# labels, V2V2R's conversion, per-tooth pulp recovery. Numbers only. See partialref.baseline.seg_train_check.
#   sbatch jobs/seg_train_check.sbatch [CASE_IDS]      (no CASE_IDS = all candidates)
set -euo pipefail
cd /msc/home/ssingh54/partial-refs
CASES=${1:-}
TAG=seg-train-check$([ -n "$CASES" ] && echo "-diag" || true)
RUN=results/seg/$(date +%Y%m%d-%H%M)-$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain -- src jobs)" ] && echo "-dirty" || true)-${TAG}-${SLURM_JOB_ID}
echo "node=$(hostname) job=$SLURM_JOB_ID run=$RUN start=$(date -Is)"
OMP_NUM_THREADS=1 PYTHONHASHSEED=0 ~/miniconda3/envs/cbct_partialref/bin/python -m partialref.baseline.seg_train_check \
  --tf3-dir /msc/home/ssingh54/sae_tooth/data/raw/toothfairy3 --splits configs/splits/v1 \
  --umamba2-dir /msc/home/ssingh54/U-Mamba2 --out "$RUN" --workers 4 ${CASES:+--cases "$CASES"}
echo "done=$(date -Is)"
