#!/usr/bin/env bash
#SBATCH --job-name=pr-teeth-repro
#SBATCH --partition=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Reproducibility check of the per-tooth features after pinning the Gaussian mixture to one thread (2026-11-17):
# recompute the features on the corrected labels WITHOUT limiting threads (8 cores, as on 12 Nov) and require them
# to equal the reference run (1 thread) bit for bit.
#   sbatch jobs/tooth_features_repro.sbatch
set -euo pipefail
cd /msc/home/ssingh54/partial-refs
PY=$HOME/miniconda3/envs/cbct_partialref/bin/python
REF=results/perception/20261117-1210-ba799a2-tooth-features-dense-fixed
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
echo "node=$(hostname) job=$SLURM_JOB_ID sha=$(git rev-parse --short HEAD) start=$(date -Is)"
NEW=results/perception/$("$PY" -m partialref.perception.tooth_features --splits configs/splits/v1/splits.csv \
  --tf3-dir /msc/home/ssingh54/sae_tooth/data/raw/toothfairy3 --workers 8 --tag tooth-features-dense-fixed-repro \
  --fixes configs/seg/label_fixes_v1.json | "$PY" -c 'import json,sys; print(json.loads(sys.stdin.read())["run_id"])')
echo "new run: $NEW"
"$PY" - "$REF" "$NEW" <<'PY'
import sys, pandas as pd
a = pd.read_parquet(f"{sys.argv[1]}/tooth_features.parquet").sort_values(["case_id", "fdi_mask"]).reset_index(drop=True)
b = pd.read_parquet(f"{sys.argv[2]}/tooth_features.parquet").sort_values(["case_id", "fdi_mask"]).reset_index(drop=True)
same = a.shape == b.shape and list(a.columns) == list(b.columns) and a.equals(b)
print(f"reference {a.shape}, new {b.shape}, bit-identical: {same}")
if not same and a.shape == b.shape:
    d = [c for c in a.columns if not a[c].equals(b[c])]
    print("columns that differ:", d[:15])
sys.exit(0 if same else 1)
PY
echo "done=$(date -Is)"
