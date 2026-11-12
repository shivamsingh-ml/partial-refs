#!/usr/bin/env bash
#SBATCH --job-name=pr-v2v2r-vlm
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --account=uzk
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Screening check (2026-11-13, Shivam): what does V2V2R's released VLM see per tooth, given the dataset's
# manual masks (the same outlines our per-tooth module uses)? Submitted pipeline (external/odin2026) as is,
# released weights (screening only, not a paper result), with three off-by-default switches:
#   PRECOMPUTED_MASK (skip segmentation), KEEP_VQA_DIR (keep per-question answers), VLM_EXTERNAL_SERVER
#   (one vLLM server for all cases, started with competition_runner.py --serve-only, same settings).
# Decoding stays at the model's default (sampled, temperature 1.0 / top_p 0.8), as submitted.
#
#   MODE=test  sbatch jobs/v2v2r_vlm_check.sbatch CASE_IDS
#       converts masks both ways (mirrored / not), runs the pipeline with V2V2R's OWN segmentation exactly as
#       submitted (one server per case, segmentation on an empty GPU) and keeps its mask, picks the variant
#       with the higher mean tooth Dice against it (stops if below 0.80: grid or labels wrong), then starts
#       one shared server and runs the pipeline on the chosen manual masks.
#   MODE=run MIRROR=yes|no  sbatch --time=... jobs/v2v2r_vlm_check.sbatch CASE_IDS
#       manual-mask runs only.
# Prints only status, timings, Dice and counts -- never report text.
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
CASE_IDS=${1:?CASE_IDS}
MODE=${MODE:-test}
DATA_DIR=/msc/home/ssingh54/Datasets/ToothFairy4
TF3_DIR=/msc/home/ssingh54/sae_tooth/data/raw/toothfairy3
APP=$PROJECT_DIR/external/odin2026
SEG_MODEL=/msc/home/ssingh54/U-Mamba2/nnunet/nnUNet_results/Dataset703_ToothFairy3SinglePulp/nnUNetTrainer_TF3_Task1_1500ep_accum2__nnUNetResEncUNetLPlans_torchres__3d_fullres_torchres_mambabot2_ps160x288x288_bs1
VLM_MODEL=$PROJECT_DIR/models/Qwen3.5-9B-AWQ-dental-cbct-sft
PR_PY=$HOME/miniconda3/envs/cbct_partialref/bin/python
SHA=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain -- src jobs external)" ] && echo "-dirty" || true)
RUN=$PROJECT_DIR/results/baseline/$(date +%Y%m%d-%H%M)-${SHA}-v2v2r-vlm-${MODE}-${SLURM_JOB_ID}
mkdir -p "$RUN"
echo "node=$(hostname) job=$SLURM_JOB_ID mode=$MODE start=$(date -Is) run=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ROOT=$RUN/model_root; mkdir -p "$ROOT"; ln -s "$SEG_MODEL" "$ROOT/nnunet_seg"; ln -s "$VLM_MODEL" "$ROOT/vlm"
FS=$APP/vqa/code/fewshot_examples
[ -d "$FS" ] || cp -a "$PROJECT_DIR/.cache/odin2026-task1-code/opt_app/vqa/code/fewshot_examples" "$FS"
export MODEL_ROOT=$ROOT NNUNET_SUBDIR=nnunet_seg
export VLM_PYTHON=$HOME/miniconda3/envs/pr_serve/bin/python
export VLM_CODE_DIR=$APP/vqa/code VLM_SCHEMA=$APP/vqa/schema/schema.json VLM_MODEL_PATH=$ROOT/vlm
export REPORT_FALLBACK_ON_ERROR=0
export VLM_GPU_UTIL=0.85 VLM_MAX_MODEL_LEN=32768 VLM_MAX_IMAGES_PER_PROMPT=3 VLM_PORT=$((20000 + SLURM_JOB_ID % 10000))
export VLM_MAX_CONCURRENCY=26        # what the per-case runs chose on this GPU (smoke job 614601: KV 166,320 / 6200)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME=$PROJECT_DIR/.cache/hf VLLM_CACHE_ROOT=$PROJECT_DIR/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=$PROJECT_DIR/.cache/inductor TRITON_CACHE_DIR=$PROJECT_DIR/.cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0

# one vLLM server for many cases (same command and settings as a per-case run)
start_server() {
  (cd "$APP/vqa" && exec "$VLM_PYTHON" code/competition_runner.py --serve-only --case-id SERVER --volume /dev/null \
     --mask /dev/null --out-dir "$RUN/server" --model "$ROOT/vlm" --port "$VLM_PORT" \
     --max-model-len "$VLM_MAX_MODEL_LEN" --gpu-memory-utilization "$VLM_GPU_UTIL" \
     --max-images-per-prompt "$VLM_MAX_IMAGES_PER_PROMPT") > "$RUN/server.log" 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
}
wait_server() {
  for i in $(seq 1 180); do
    grep -q "\[serve-only\] ready" "$RUN/server.log" && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -5 "$RUN/server.log"; exit 1; }
    sleep 5
  done
  grep "\[serve-only\]" "$RUN/server.log"
  export VLM_EXTERNAL_SERVER=1
}

[ "$MODE" = run ] && start_server          # loads while the masks are converted

# masks (CPU)
VARIANTS=both; [ "$MODE" = run ] && VARIANTS=${MIRROR:?MIRROR=yes|no for MODE=run}
OMP_NUM_THREADS=2 "$PR_PY" -m partialref.baseline.v2v2r_masks --cases "$CASE_IDS" --tf3-dir "$TF3_DIR" \
  --tf4-dir "$DATA_DIR" --out-dir "$RUN/masks" --mirror "$VARIANTS" --workers 4 \
  || echo "MASK_ERRORS: cases listed above are skipped (job 616840 stopped here on one case)"

set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate umamba2; set -u

run_case() {   # $1 case, $2 tag, $3 mask file or ""
  local CID=$1 TAG=$2 MASK=$3 IN OUT rc t0
  IN=$RUN/$CID/$TAG/input/images/cbct; OUT=$RUN/$CID/$TAG/output; mkdir -p "$IN" "$OUT"
  python -c "import SimpleITK as s,sys; s.WriteImage(s.ReadImage(sys.argv[1]), sys.argv[2])" \
    "$DATA_DIR/$CID/cbct/volume.nii.gz" "$IN/$CID.mha"
  t0=$(date +%s)
  if (cd "$APP" && CASE_ID=$CID PRECOMPUTED_MASK=$MASK KEEP_VQA_DIR=$RUN/$CID/$TAG/vqa INPUT_DIR=$RUN/$CID/$TAG/input \
        OUTPUT_DIR=$OUT python inference.py) > "$RUN/$CID/$TAG/log.txt" 2>&1; then rc=0; else rc=$?; fi
  rm -f "$IN/$CID.mha"
  grep -q "competition_runner exited 0" "$RUN/$CID/$TAG/log.txt" && ok=VLM_OK || ok=VLM_FAILED
  echo "RESULT $CID $TAG rc=$rc $ok seconds=$(( $(date +%s) - t0 )) pred_files=$(ls "$RUN/$CID/$TAG/vqa/predictions" 2>/dev/null | wc -l)"
}

status=0
if [ "$MODE" = test ]; then
  for CID in ${CASE_IDS//,/ }; do run_case "$CID" own ""; done      # as submitted: one server per case
  for CID in ${CASE_IDS//,/ }; do
    PRED=$RUN/$CID/own/vqa/facts_masks/$CID.nii.gz
    for V in nomirror mirror; do
      D=$("$PR_PY" -m partialref.baseline.v2v2r_masks --dice "$PRED" "$RUN/masks/$V/$CID.nii.gz" 2>/dev/null \
          | "$PR_PY" -c 'import json,sys; print(json.load(sys.stdin)["mean_dice_teeth"])' 2>/dev/null) || D=nan
      echo "DICE $CID $V $D"
    done
  done | tee "$RUN/dice.txt"
  CHOSEN=$("$PR_PY" - "$RUN/dice.txt" <<'PY'
import sys, collections
s = collections.defaultdict(list)
for line in open(sys.argv[1]):
    _, cid, v, d = line.split()
    if d not in ("nan", "None"):
        s[v].append(float(d))
m = {v: sum(x) / len(x) for v, x in s.items() if x}
if not m:
    sys.exit("no Dice values: own-segmentation runs failed")
best = max(m, key=m.get)
if m[best] < 0.80:
    sys.exit(f"best mean tooth Dice {m[best]:.3f} < 0.80: grid or labels do not match V2V2R's masks")
print(best)
PY
)
  echo "CHOSEN $CHOSEN"
  start_server
  wait_server
else
  CHOSEN=$([ "$MIRROR" = yes ] && echo mirror || echo nomirror)
  wait_server
fi
for CID in ${CASE_IDS//,/ }; do
  if [ -f "$RUN/masks/$CHOSEN/$CID.nii.gz" ]; then run_case "$CID" manual "$RUN/masks/$CHOSEN/$CID.nii.gz"
  else echo "RESULT $CID manual SKIPPED no converted mask"; fi
done 2>&1 | tee -a "$RUN/results.txt"
grep -c "VLM_OK" "$RUN/results.txt" || status=1
grep -q "VLM_FAILED" "$RUN/results.txt" && status=1
echo "done=$(date -Is) status=$status"
exit $status
