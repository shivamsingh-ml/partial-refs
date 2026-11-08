#!/usr/bin/env bash
#SBATCH --job-name=pr-parse
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --account=uzk
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# GT/T1: per-report findings with a LOCAL LLM (no external API). Starts a vLLM server (pr_serve env),
# waits until it is ready, runs partialref.extract.per_report for each language, scores the result
# against Lumetti et al.'s labels (internal check), then stops the server. One job for all reports.
# Report text never appears in our outputs; vLLM request logging is disabled where supported and the
# server log is only ever grepped for errors.
#   sbatch jobs/parse_reports.sbatch LIMIT_CASES LANGS [MODEL_DIR] [SERVED_NAME]
#   e.g. sbatch jobs/parse_reports.sbatch 5 en            (diagnostic)
#        sbatch jobs/parse_reports.sbatch 50 en,it        (bake-off subset)
#        sbatch jobs/parse_reports.sbatch 0 en,it         (all reports)
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
LIMIT=${1:-5}; LANGS=${2:-en}
MODEL_DIR=${3:-$PROJECT_DIR/models/Qwen3-14B}; NAME=${4:-qwen3-14b}
MANIFEST=results/manifest/20261107-1830-c2aa718-manifest
LUMETTI=$(ls -d results/lumetti/*-lumetti-labels | tail -1)/instances.parquet
EXCLUDE=P116,P117,P260,P261
PORT=$((21000 + SLURM_JOB_ID % 9000))
RUN=results/extract/$(date +%Y%m%d-%H%M)-$(git rev-parse --short HEAD)-parse-${NAME}-n${LIMIT}-${SLURM_JOB_ID}
mkdir -p "$RUN"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME=$PROJECT_DIR/.cache/hf VLLM_CACHE_ROOT=$PROJECT_DIR/.cache/vllm PYTHONHASHSEED=0
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) model=$MODEL_DIR limit=$LIMIT langs=$LANGS run=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VLLM=$HOME/miniconda3/envs/pr_serve/bin/vllm
NOLOG=""; "$VLLM" serve --help=all 2>/dev/null | grep -q -- "--disable-log-requests" && NOLOG="--disable-log-requests"
echo "request-logging flag: ${NOLOG:-<none; default off in this version or unsupported>}"
"$VLLM" serve "$MODEL_DIR" --served-model-name "$NAME" --port "$PORT" --host 127.0.0.1 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --dtype bfloat16 $NOLOG > "$RUN/vllm_server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { echo "server ready after $((i*10)) s"; break; }
  kill -0 $SERVER 2>/dev/null || { echo "server died"; grep -E "ERROR|Error|Traceback" "$RUN/vllm_server.log" | tail -5; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "server not ready after 20 min"; exit 1; }

set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate cbct_partialref; set -u
for L in ${LANGS//,/ }; do
  python -m partialref.extract.per_report --manifest "$MANIFEST" --lang "$L" --server "http://127.0.0.1:$PORT" \
    --model "$NAME" --model-path "$MODEL_DIR" --limit-cases "$LIMIT" --out-dir "$RUN"
  python -m partialref.extract.score_parser --ours "$RUN/instances_$L.parquet" --lumetti "$LUMETTI" \
    --exclude "$EXCLUDE" --tag "parser-check-${NAME}-${L}-n${LIMIT}"
done
if [[ ",$LANGS," == *",en,"* && ",$LANGS," == *",it,"* ]]; then
  echo "--- translation check: EN extractions scored against IT extractions (same parser)"
  python -m partialref.extract.score_parser --ours "$RUN/instances_en.parquet" --lumetti "$RUN/instances_it.parquet" \
    --exclude "$EXCLUDE" --tag "en-vs-it-${NAME}-n${LIMIT}"
fi
echo "server errors (if any):"; grep -cE " ERROR |Traceback" "$RUN/vllm_server.log" || true
echo "done=$(date -Is)"
