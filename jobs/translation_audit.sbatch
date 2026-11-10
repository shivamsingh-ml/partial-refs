#!/usr/bin/env bash
#SBATCH --job-name=pr-transaudit
#SBATCH --partition=gpu
#SBATCH --qos=a100-sxm4-80gb
#SBATCH --account=uzk
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Translation audit IT -> EN with a LOCAL LLM (no external API). Starts a vLLM server (same start-up and
# DUA checks as jobs/parse_reports.sbatch, which is left untouched), runs partialref.extract.translation_audit
# on the report pairs of one parser half (or all), then stops the server. Outputs are codes only.
#   sbatch jobs/translation_audit.sbatch HALF [PLANT] [OUT_DIR] [MODEL_DIR] [SERVED_NAME]
#   e.g. sbatch jobs/translation_audit.sbatch tune 60                        (tune half + recall check on 60 pairs)
#        sbatch jobs/translation_audit.sbatch heldout 0 results/audit/<tune run>   (add the other half to that run)
#        LIMIT_PAIRS=5 sbatch --export=ALL jobs/translation_audit.sbatch tune 2    (diagnostic)
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
HALF=${1:-tune}; PLANT=${2:-0}
MODEL_DIR=${4:-$PROJECT_DIR/models/Qwen3-32B}; NAME=${5:-qwen3-32b}
MANIFEST=results/manifest/20261107-1830-c2aa718-manifest
RUN=${3:-results/audit/$(date +%Y%m%d-%H%M)-$(git rev-parse --short HEAD)-translation-audit-${NAME}-${HALF}-${SLURM_JOB_ID}}
KNOWN=$(ls -d results/audit/*-translation-screen/tune_read_verdicts.csv 2>/dev/null | tail -1 || true)
PORT=$((21000 + SLURM_JOB_ID % 9000))
mkdir -p "$RUN"; LOG="$RUN/vllm_server-${SLURM_JOB_ID}.log"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME=$PROJECT_DIR/.cache/hf VLLM_CACHE_ROOT=$PROJECT_DIR/.cache/vllm PYTHONHASHSEED=0
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) model=$MODEL_DIR half=$HALF plant=$PLANT limit_pairs=${LIMIT_PAIRS:-0} known=${KNOWN:-<none>} run=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VLLM=$HOME/miniconda3/envs/pr_serve/bin/vllm
# capture first, then grep: with pipefail, `help | grep -q` can fail via SIGPIPE even when the flag exists
HELP=$("$VLLM" serve --help=all 2>/dev/null || true)
NOLOG=""; grep -q -- "--disable-log-requests" <<<"$HELP" && NOLOG="--disable-log-requests"
echo "request-logging flag: ${NOLOG:-<none; default off in this version or unsupported>}"
"$VLLM" serve "$MODEL_DIR" --served-model-name "$NAME" --port "$PORT" --host 127.0.0.1 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --dtype auto $NOLOG > "$LOG" 2>&1 &
SERVER=$!
cleanup() {   # always: stop the server, then the DUA check (count only, never print the lines)
  kill $SERVER 2>/dev/null || true
  local n; n=$(grep -cE "Received request|prompt:|Prompt:" "$LOG" 2>/dev/null || true)
  echo "server-log lines that look like logged requests: ${n:-0}"
  if [[ "${n:-0}" != "0" ]]; then echo "WARNING: request logging appears ON; deleting the server log"; rm -f "$LOG"; fi
}
trap cleanup EXIT
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "server ready after $((i*10)) s"; break; }
  kill -0 $SERVER 2>/dev/null || { echo "server died"; grep -E "ERROR|Error|Traceback" "$LOG" | tail -5; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "server not ready after 20 min"; exit 1; }

set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate cbct_partialref; set -u
python -m partialref.extract.translation_audit --manifest "$MANIFEST" --server "http://127.0.0.1:$PORT" \
  --model "$NAME" --model-path "$MODEL_DIR" --half "$HALF" --plant "$PLANT" --limit-pairs "${LIMIT_PAIRS:-0}" \
  --out-dir "$RUN" ${KNOWN:+--known "$KNOWN"}
echo "server errors (if any):"; grep -cE " ERROR |Traceback" "$LOG" || true
echo "done=$(date -Is)"
