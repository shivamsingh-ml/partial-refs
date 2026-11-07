#!/usr/bin/env bash
#SBATCH --job-name=pr-cmp-copies
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Byte-compare two copies of ToothFairy4 (same relative file list required). Prints counts and
# the names of differing files only; never prints file contents. Read-only on both trees.
set -euo pipefail
A=${A:-/msc/home/ssingh54/Datasets/ToothFairy4}
B=${B:-/msc/home/ssingh54/cbct-vlm-transfer/data/raw}
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) A=$A B=$B"
diff <(cd "$A" && find . -type f | sort) <(cd "$B" && find . -type f | sort) && echo "file lists identical"
OUT=$(mktemp -d "${TMPDIR:-/tmp}/cmpcopies.XXXX"); trap 'rm -rf "$OUT"' EXIT
cd "$A"
find . -type f -print0 | xargs -0 -P 8 -I{} sh -c 'cmp -s "$1" "$2/$1" && echo SAME || echo "DIFF $1"' _ {} "$B" > "$OUT/cmp.txt" 2>/dev/null || true
echo "SAME: $(grep -c '^SAME' "$OUT/cmp.txt")  DIFF: $(grep -c '^DIFF' "$OUT/cmp.txt" || true)  (files: $(find . -type f | wc -l))"
grep '^DIFF' "$OUT/cmp.txt" | head -20 || true
echo "done=$(date -Is)"
