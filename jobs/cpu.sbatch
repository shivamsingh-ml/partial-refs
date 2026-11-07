#!/usr/bin/env bash
#SBATCH --job-name=pr-cpu
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Generic CPU job (HANDOFF §3): runs `python -m partialref.<module> [args...]` in cbct_partialref.
#   sbatch [--job-name=pr-<step>] [--time=..] jobs/cpu.sbatch <module> [args...]
#   e.g. sbatch --job-name=pr-manifest jobs/cpu.sbatch data.manifest --data-dir "$DATA_DIR" --limit 5
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch jobs/cpu.sbatch <module> [args...]" >&2; exit 2; }
source /msc/home/ssingh54/partial-refs/jobs/env.sh
cd "$PROJECT_DIR"
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is) module=partialref.$1"
python -m "partialref.$1" "${@:2}"
echo "done=$(date -Is)"
