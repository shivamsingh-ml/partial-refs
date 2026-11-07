#!/usr/bin/env bash
#SBATCH --job-name=pr-smoke-v2v2r
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --account=uzk
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# HANDOFF M0 smoke run: the submitted V2V2R pipeline (external/odin2026/inference.py) end to end on a
# few cases, no container. Segmentation in conda env `umamba2` (torch 2.5.1/cu124, as in the image's
# main env), VLM in `pr_serve` (vLLM 0.19.0, as in the image's /opt/vlmenv). Released weights
# (environment test only: NO metrics are computed or reported from this run).
# GPU: full A100 40GB. The 3g.20gb MIG slice failed (job 614600): vLLM 0.19 parses CUDA_VISIBLE_DEVICES
# as integers and crashes on SLURM's MIG UUID ('MIG-...'), so MIG slices cannot host vLLM here.
# Template fallback is OFF here so a VLM failure fails the case instead of writing a template report.
# Prints only status, timings and report lengths -- never report text.
#   sbatch jobs/smoke_v2v2r.sbatch [CASE_IDS]      default CASE_IDS=P001,F001,A003
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
CASE_IDS=${1:-P001,F001,A003}
DATA_DIR=/msc/home/ssingh54/Datasets/ToothFairy4
APP=$PROJECT_DIR/external/odin2026
SEG_MODEL=/msc/home/ssingh54/U-Mamba2/nnunet/nnUNet_results/Dataset703_ToothFairy3SinglePulp/nnUNetTrainer_TF3_Task1_1500ep_accum2__nnUNetResEncUNetLPlans_torchres__3d_fullres_torchres_mambabot2_ps160x288x288_bs1
VLM_MODEL=$PROJECT_DIR/models/Qwen3.5-9B-AWQ-dental-cbct-sft
RUN=$PROJECT_DIR/results/smoke/$(date +%Y%m%d-%H%M)-$(git rev-parse --short HEAD)-v2v2r-${SLURM_JOB_ID}
mkdir -p "$RUN"

# model root laid out as the container expects (/opt/ml/model/{nnunet_seg,vlm})
ROOT=$RUN/model_root; mkdir -p "$ROOT"; ln -s "$SEG_MODEL" "$ROOT/nnunet_seg"; ln -s "$VLM_MODEL" "$ROOT/vlm"
# few-shot examples are kept out of git (DUA); the runner needs them at run time
FS=$APP/vqa/code/fewshot_examples
[ -d "$FS" ] || cp -a "$PROJECT_DIR/.cache/odin2026-task1-code/opt_app/vqa/code/fewshot_examples" "$FS"

export MODEL_ROOT=$ROOT NNUNET_SUBDIR=nnunet_seg
export VLM_PYTHON=$HOME/miniconda3/envs/pr_serve/bin/python
export VLM_CODE_DIR=$APP/vqa/code VLM_SCHEMA=$APP/vqa/schema/schema.json VLM_MODEL_PATH=$ROOT/vlm
export REPORT_FALLBACK_ON_ERROR=0
export VLM_DTYPE=float16 VLM_GPU_UTIL=0.85 VLM_MAX_MODEL_LEN=32768 VLM_PORT=$((20000 + SLURM_JOB_ID % 10000))
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME=$PROJECT_DIR/.cache/hf VLLM_CACHE_ROOT=$PROJECT_DIR/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=$PROJECT_DIR/.cache/inductor TRITON_CACHE_DIR=$PROJECT_DIR/.cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) cases=$CASE_IDS run=$RUN"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate umamba2; set -u
python -c "import SimpleITK, cc3d, acvl_utils, batchgenerators, nnunetv2, torch; print('seg env ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$VLM_PYTHON" -c "import vllm, openai; print('vlm env ok, vllm', vllm.__version__)"

status=0
for CID in ${CASE_IDS//,/ }; do
  IN=$RUN/$CID/input/images/cbct; OUT=$RUN/$CID/output; mkdir -p "$IN" "$OUT"
  python -c "import SimpleITK as s,sys; s.WriteImage(s.ReadImage(sys.argv[1]), sys.argv[2])" \
    "$DATA_DIR/$CID/cbct/volume.nii.gz" "$IN/$CID.mha"
  echo "--- $CID"; t0=$(date +%s)
  if (cd "$APP" && INPUT_DIR=$RUN/$CID/input OUTPUT_DIR=$OUT python inference.py) > "$RUN/$CID/log.txt" 2>&1; then rc=0; else rc=$?; status=1; fi
  t1=$(date +%s)
  grep -E "=\+= (input axcodes|TIMING|  CASE TOTAL|competition_runner exited)|\[FAIL\]|WARNING|Traceback" "$RUN/$CID/log.txt" | grep -v "^\[vllm\]" | tail -12
  grep -q "competition_runner exited 0" "$RUN/$CID/log.txt" && echo "VLM_OK $CID" || { echo "VLM_FAILED $CID"; status=1; }
  python - "$OUT/diagnostic-imaging-report.json" "$CID" "$rc" "$((t1-t0))" <<'PY'
import json, sys
p, cid, rc, secs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    rep = json.load(open(p))["report"]
    print(f"RESULT {cid}: rc={rc} seconds={secs} report_chars={len(rep)} report_words={len(rep.split())} valid_json=True")
except Exception as e:
    print(f"RESULT {cid}: rc={rc} seconds={secs} report missing/invalid ({type(e).__name__})")
PY
  rm -f "$IN/$CID.mha"
done
echo "done=$(date -Is) status=$status"
exit $status
