#!/usr/bin/env bash
#SBATCH --job-name=pr-dino
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --account=uzk
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# DINO embeddings of tooth slices (PAPER_PLAN step 2). Inference only; small model, one A100 (40 GB) is ample.
#   sbatch jobs/dino_embed.sbatch <slices run dir> <model dir, e.g. models/dinov2-base> [LIMIT]
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
[ $# -ge 2 ] || { echo "usage: sbatch jobs/dino_embed.sbatch <slices run> <model dir> [limit]" >&2; exit 2; }
SLICES=$1; MODEL=$2; LIMIT=${3:-0}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pr_train; set -u
SUFFIX=""; if [ "$LIMIT" != 0 ]; then SUFFIX="-limit$LIMIT"; fi
DIRTY=""; if [ -n "$(git status --porcelain -- src jobs)" ]; then DIRTY="-dirty"; fi
OUT=results/perception/$(date +%Y%m%d-%H%M)-$(git rev-parse --short HEAD)${DIRTY}-dino-$(basename "$MODEL")${SUFFIX}-${SLURM_JOB_ID}
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) slices=$SLICES model=$MODEL out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python src/partialref/perception/dino_embed.py --slices "$SLICES" --model "$MODEL" --out "$OUT" --limit "$LIMIT"
echo "done=$(date -Is)"
