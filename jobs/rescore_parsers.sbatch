#!/usr/bin/env bash
#SBATCH --job-name=pr-rescore
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Re-score finished parse runs with the fixed scorer (review 2026-11-09 #2/#3), CPU only, no model calls:
# vs Lumetti (tune + heldout, per language present), run-to-run per report (if repeat/ exists),
# EN vs IT per report (if both exist). Needs a Lumetti run that has cases.parquet.
#   sbatch jobs/rescore_parsers.sbatch results/extract/<parse run> [results/extract/<parse run> ...]
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch jobs/rescore_parsers.sbatch <parse run dir>..." >&2; exit 2; }
source /msc/home/ssingh54/partial-refs/jobs/env.sh
cd "$PROJECT_DIR"
LUMETTI_DIR=$(ls -d results/lumetti/*-lumetti-labels | tail -1)
[ -f "$LUMETTI_DIR/cases.parquet" ] || { echo "latest Lumetti run $LUMETTI_DIR has no cases.parquet" >&2; exit 1; }
EXCLUDE=P116,P117,P260,P261
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is) git=$(git rev-parse --short HEAD) lumetti=$LUMETTI_DIR"
for RUN in "$@"; do
  NAME=$(basename "$RUN")
  for L in en it; do
    [ -f "$RUN/instances_$L.parquet" ] || continue
    for H in tune heldout; do
      python -m partialref.extract.score_parser --ours "$RUN/instances_$L.parquet" --lumetti "$LUMETTI_DIR/instances.parquet" \
        --exclude "$EXCLUDE" --half "$H" --tag "rescore-${NAME}-${L}-${H}"
    done
  done
  if [ -f "$RUN/repeat/instances_en.parquet" ]; then
    python -m partialref.extract.score_parser --ours "$RUN/repeat/instances_en.parquet" --ref-parse "$RUN/instances_en.parquet" \
      --exclude "$EXCLUDE" --tag "rescore-${NAME}-repeat-en"
  fi
  if [ -f "$RUN/instances_en.parquet" ] && [ -f "$RUN/instances_it.parquet" ]; then
    python -m partialref.extract.score_parser --ours "$RUN/instances_en.parquet" --ref-parse "$RUN/instances_it.parquet" \
      --exclude "$EXCLUDE" --tag "rescore-${NAME}-en-vs-it"
  fi
done
echo "done=$(date -Is)"
