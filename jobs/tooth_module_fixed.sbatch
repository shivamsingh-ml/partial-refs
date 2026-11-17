#!/usr/bin/env bash
#SBATCH --job-name=pr-teeth-fixed
#SBATCH --partition=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Per-tooth module rerun on the corrected labels (2026-11-17): the 2026-11-12 chain (features run 616780 ->
# classifier 616784 -> human reference) with ONE change: --fixes configs/seg/label_fixes_v1.json (module scope:
# tooth, pulp and jaw labels of 11 scans). Same splits, parse run, manifest, readers and settings. Then old vs new
# side by side: unfixed scans must give identical features; AUCs per type. (The screening comparison is not rerun
# here: its script writes into the VLM run folder and would overwrite the 2026-11-12 result.)
#   sbatch jobs/tooth_module_fixed.sbatch
set -euo pipefail
cd /msc/home/ssingh54/partial-refs
PY=$HOME/miniconda3/envs/cbct_partialref/bin/python
SHA=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain -- src jobs configs)" ] && echo "-dirty" || true)
echo "node=$(hostname) job=$SLURM_JOB_ID sha=$SHA start=$(date -Is)"
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
OLD_FEAT=results/perception/20261112-1026-8f55a26-tooth-features-dense
OLD_CLF=results/perception/20261112-1027-8f55a26-tooth-classifier-dense

FEAT=$("$PY" -m partialref.perception.tooth_features --splits configs/splits/v1/splits.csv \
  --tf3-dir /msc/home/ssingh54/sae_tooth/data/raw/toothfairy3 --workers 8 --tag tooth-features-dense-fixed \
  --fixes configs/seg/label_fixes_v1.json | "$PY" -c 'import json,sys; print(json.loads(sys.stdin.read())["run_id"])')
FEAT=results/perception/$FEAT; echo "features: $FEAT"
CLF_ID=$("$PY" -m partialref.perception.tooth_classifier --features "$FEAT" \
  --parse-run results/extract/20261110-2029-f4df9e5-parse-qwen3-32b-v32-n0-615806 \
  --manifest results/manifest/20261107-1830-c2aa718-manifest --readers results/propensity/20261111-2146-73f16cd-readers-dev \
  --splits configs/splits/v1/splits.csv --tag tooth-classifier-dense-fixed | tail -200 | grep -o '"run_id": "[^"]*"' | head -1 | cut -d'"' -f4)
CLF=results/perception/$CLF_ID; echo "classifier: $CLF"
[ -d "$CLF" ] || { echo "classifier run folder not found"; exit 1; }
"$PY" -m partialref.perception.human_reference --classifier-run "$CLF"
"$PY" - "$OLD_FEAT" "$FEAT" "$OLD_CLF" "$CLF" <<'PY'
import sys, pandas as pd
of, nf, oc, nc = sys.argv[1:]
a, b = pd.read_parquet(f"{of}/tooth_features.parquet"), pd.read_parquet(f"{nf}/tooth_features.parquet")
print(f"teeth: old {len(a)}, new {len(b)}; cases old {a.case_id.nunique()}, new {b.case_id.nunique()}")
fixed = ["P041", "P061", "P105", "P192", "P248", "P368", "P450", "S0002", "S0004", "S0034", "S0038"]
cols = [c for c in a.columns if c in b.columns and c not in ("case_id", "fdi_mask", "fdi_report", "error", "trace")
        and pd.api.types.is_numeric_dtype(a[c])]
k = ["case_id", "fdi_mask"]
m = a[~a.case_id.isin(fixed)].merge(b[~b.case_id.isin(fixed)], on=k, suffixes=("_o", "_n"))
diff = [c for c in cols if not ((m[c + "_o"] == m[c + "_n"]) | (m[c + "_o"].isna() & m[c + "_n"].isna())).all()]
print(f"unfixed scans: {len(m)} teeth compared, columns with any difference: {diff[:10]}")
print("teeth per fixed scan (old -> new):", {c: (int((a.case_id == c).sum()), int((b.case_id == c).sum())) for c in fixed})
ao, an = pd.read_csv(f"{oc}/auc_by_type.csv"), pd.read_csv(f"{nc}/auc_by_type.csv")
print("AUC old:\n", ao.round(4).to_string(index=False))
print("AUC new:\n", an.round(4).to_string(index=False))
PY
echo "done=$(date -Is)"
