#!/usr/bin/env bash
#SBATCH --job-name=pr-seg-train
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --account=uzk
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
# Retrain V2V2R's segmentation network (U-Mamba2) on our 424 training scans (2026-11-17). V2V2R's recipe as in
# /msc/home/ssingh54/U-Mamba2/jobs/train_tf3.sh: trainer nnUNetTrainer_TF3_Task1_1500ep_accum2, plans
# nnUNetResEncUNetLPlans_torchres, configuration 3d_fullres_torchres_mambabot2_ps160x288x288_bs1, fold "all", started
# from self-supervised weights; here: our Dataset713 (label fixes, split v1, our intensity values) and OUR pretrained
# weights (data/pretrain/dae_v1, last round, converted by partialref.baseline.dae_to_nnunet). No test scan anywhere.
# Resources: 12 cores / 96 GB instead of the original 16 / 128 (billing 56 vs 64 per second); the diag run checks that
# data loading keeps up (round time vs the original ~580 s on an A100 40 GB).
#   MODE=diag WEIGHTS=<file> sbatch --export=ALL --time=01:00:00 jobs/seg_train.sbatch
#       ~45 min into data/nnunet/results_diag (never the real results folder): weights load, rounds run, round time.
#   MODE=full sbatch --export=ALL jobs/seg_train.sbatch        (resubmit until 1,500 rounds; resumes with --c)
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
MODE=${MODE:?MODE=diag|full}
DS=713; DSNAME=Dataset713_TF3SinglePulpDev; PLANS=nnUNetResEncUNetLPlans_torchres
CONF=3d_fullres_torchres_mambabot2_ps160x288x288_bs1; TRAINER=nnUNetTrainer_TF3_Task1_1500ep_accum2
WEIGHTS=${WEIGHTS:-$PROJECT_DIR/data/pretrain/dae_v1/dae_nnunet_weights.pth}
export nnUNet_raw=$PROJECT_DIR/data/nnunet/raw nnUNet_preprocessed=$PROJECT_DIR/data/nnunet/preprocessed
if [ "$MODE" = diag ]; then export nnUNet_results=$PROJECT_DIR/data/nnunet/results_diag_${SLURM_JOB_ID}
else export nnUNet_results=$PROJECT_DIR/data/nnunet/results; fi
export nnUNet_n_proc_DA=${SLURM_CPUS_PER_TASK:-12} PYTHONHASHSEED=0
SHA=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain -- src jobs external configs)" ] && echo "-dirty" || true)
case "$SHA" in *-dirty) [ "$MODE" = full ] && { echo "refusing a full run from a dirty tree"; exit 1; } ;; esac
# the nnU-Net fork that runs the training lives outside this repo: record it and require it unchanged
FORK=/msc/home/ssingh54/U-Mamba2
FORK_SHA=$(git -C $FORK rev-parse --short HEAD)$([ -n "$(git -C $FORK status --porcelain -- nnunetv2)" ] && echo "-dirty" || true)
echo "nnunetv2 fork: $FORK_SHA"
case "$FORK_SHA" in *-dirty) [ "$MODE" = full ] && { echo "refusing a full run: nnunetv2 fork has local changes"; exit 1; } ;; esac
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate umamba2; set -u
echo "node=$(hostname) job=$SLURM_JOB_ID mode=$MODE sha=$SHA results=$nnUNet_results start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import nnunetv2; print('nnunetv2 from', nnunetv2.__file__)"
OUTF=$nnUNet_results/$DSNAME/${TRAINER}__${PLANS}__${CONF}/fold_all
mkdir -p "$OUTF"
echo "$SHA fork=$FORK_SHA $(date -Is) job $SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) weights $WEIGHTS" >> "$OUTF/../partialref_runs.txt"
if [ -f "$OUTF/checkpoint_latest.pth" ] || [ -f "$OUTF/checkpoint_final.pth" ]; then
  echo "checkpoint found -> resuming with --c"
  CMD=(nnUNetv2_train $DS $CONF all -p $PLANS -tr $TRAINER --c)
else
  [ -f "$WEIGHTS" ] || { echo "pretrained weights missing: $WEIGHTS"; exit 1; }
  sha256sum "$WEIGHTS" | tee -a "$OUTF/../partialref_runs.txt"
  echo "fresh start from pretrained weights"
  CMD=(nnUNetv2_train $DS $CONF all -p $PLANS -tr $TRAINER -pretrained_weights "$WEIGHTS")
fi
if [ "$MODE" = diag ]; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 60 > "$OUTF/../gpu_util.txt" &
  UTIL_PID=$!
  timeout --kill-after=60 45m "${CMD[@]}" || rc=$?
  kill $UTIL_PID 2>/dev/null || true
  echo "diag stopped (rc=${rc:-0}; 124 = stopped by the 45-min limit, expected)"
  grep -h -E "Epoch time|train_loss|Epoch [0-9]+ *$" "$OUTF"/training_log_*.txt | tail -20 || true
  echo "weight loading (from this job's output):"; grep -h -E "Loading pretrained|seg_layer" "logs/pr-seg-train-${SLURM_JOB_ID}.out" || true
  echo "GPU utilisation (every 60 s; sustained values well below ~90 % would mean data loading is too slow):"
  cat "$OUTF/../gpu_util.txt"
  echo "compare round times with the original (~500-670 s on an A100-PCIE-40GB) only on the same GPU type"
  [ "${rc:-0}" = 0 ] || [ "${rc:-0}" = 124 ] || exit "$rc"
else
  "${CMD[@]}"
fi
echo "done=$(date -Is)"
