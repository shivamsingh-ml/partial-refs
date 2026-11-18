#!/usr/bin/env bash
#SBATCH --job-name=pr-seg-eval
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --account=uzk
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Test-set evaluation of a V2V2R segmentation network (2026-11-18): V2V2R's own run_segmentation on the ToothFairy4
# volumes, Dice against the converted ToothFairy3 manual masks. See partialref.baseline.seg_eval.
#   MODEL=released CASES=P001,F001,S0000 sbatch --export=ALL --time=00:30:00 jobs/seg_eval.sbatch
#       cheap check: released weights on 3 development scans it was trained on -> tooth Dice should be ~0.95+
#   MODEL=ours CASES=test sbatch --export=ALL jobs/seg_eval.sbatch
#       our retrained network (checkpoint_final.pth) on the 107 test scans; resubmit to continue (skips done scans)
# Predicted masks go to data/seg_eval/<name>/ (git-ignored); scores to results/seg/<run_id>/.
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
MODEL=${MODEL:?MODEL=released|ours}; CASES=${CASES:?CASES=test|id,id,...}
TF3=/msc/home/ssingh54/sae_tooth/data/raw/toothfairy3; TF4=/msc/home/ssingh54/Datasets/ToothFairy4
SUB=nnUNetTrainer_TF3_Task1_1500ep_accum2__nnUNetResEncUNetLPlans_torchres__3d_fullres_torchres_mambabot2_ps160x288x288_bs1
case "$MODEL" in
  released) SEG_MODEL=/msc/home/ssingh54/U-Mamba2/nnunet/nnUNet_results/Dataset703_ToothFairy3SinglePulp/$SUB ;;
  ours)     SEG_MODEL=$PROJECT_DIR/data/nnunet/results/Dataset713_TF3SinglePulpDev/$SUB
            [ -f "$SEG_MODEL/fold_all/checkpoint_final.pth" ] || { echo "no checkpoint_final.pth: training not finished"; exit 1; } ;;
  *) echo "MODEL must be released or ours"; exit 1 ;;
esac
SHA=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain -- src jobs external configs)" ] && echo "-dirty" || true)
CKPT_SHA=$(sha256sum "$SEG_MODEL/fold_all/checkpoint_final.pth" | cut -d' ' -f1)
# mask folder named after the checkpoint (a different model never reuses another model's masks) and the case set
CASE_KEY=$( [ "$CASES" = test ] && echo test || echo "$CASES" | tr ',' '-' )
TAG=$MODEL-${CKPT_SHA:0:12}-$CASE_KEY
PRED=$PROJECT_DIR/data/seg_eval/$TAG; ROOT=$PRED/model_root
mkdir -p "$ROOT"; ln -sfn "$SEG_MODEL" "$ROOT/nnunet_seg"
# the submitted config reads these; fix them here so a stray login-shell value cannot change the run
unset NNUNET_CHECKPOINT USE_TTA; export NNUNET_SUBDIR=nnunet_seg
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate umamba2; set -u
export PYTHONPATH=$PROJECT_DIR/src PYTHONHASHSEED=0
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL cases=$CASES sha=$SHA seg_model=$SEG_MODEL ckpt_sha256=$CKPT_SHA start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m partialref.baseline.seg_eval predict --cases "$CASES" --splits configs/splits/v1/splits.csv \
  --tf4-dir "$TF4" --model-root "$ROOT" --pred-dir "$PRED"
RUN=results/seg/$(date +%Y%m%d-%H%M)-${SHA}-seg-eval-$MODEL-$( [ "$CASES" = test ] && echo test || echo diag )-${SLURM_JOB_ID}
python -m partialref.baseline.seg_eval score --cases "$CASES" --splits configs/splits/v1/splits.csv \
  --tf3-dir "$TF3" --tf4-dir "$TF4" --pred-dir "$PRED" --out "$RUN" --workers 8 \
  --model "$SEG_MODEL/fold_all/checkpoint_final.pth sha256=$CKPT_SHA" || rc=$?
echo "scores: $RUN (score exit ${rc:-0}: 1 = some scans failed, listed in summary.json)"; echo "done=$(date -Is)"
exit "${rc:-0}"
