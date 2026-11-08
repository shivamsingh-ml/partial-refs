#!/usr/bin/env bash
#SBATCH --job-name=pr-vischeck
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# P/F/S visibility features v2 (centroids, maxilla halves) and the image/report checks built on them.
# Image-only features + parser instances; no report text. Steps:
#   1. 5-case diagnostic into results/diagnostic/ (aborts if the new columns are missing)
#   2. full P/F/S run (tag visibility-features-v2)
#   3. jaw-inside shares (review #4 caveat), side geometry, report-scan fingerprint, laterality
#   sbatch jobs/visibility_checks.sbatch results/extract/<parse run>
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: sbatch jobs/visibility_checks.sbatch <parse run dir>" >&2; exit 2; }
PARSE=$1
source /msc/home/ssingh54/partial-refs/jobs/env.sh
cd "$PROJECT_DIR"
MANIFEST=results/manifest/20261107-1830-c2aa718-manifest
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is) git=$(git rev-parse --short HEAD) parse=$PARSE"
[ -f "$PARSE/extractions_en.parquet" ] || { echo "no EN extractions in $PARSE" >&2; exit 1; }

echo "--- 1. diagnostic (5 cases)"
python -m partialref.audit.visibility_features --manifest "$MANIFEST" --limit 5 --workers 4 \
  --results-dir results/diagnostic --tag visibility-features-v2-diag
DIAG=$(ls -d results/diagnostic/audit/*-visibility-features-v2-diag | tail -1)
python - "$DIAG" <<'PY'
import sys, pandas as pd
d = sys.argv[1]
cf, tf = pd.read_parquet(f"{d}/case_features.parquet"), pd.read_parquet(f"{d}/tooth_features.parquet")
need_c = ["mandible_cx", "maxilla_left_touches_lateral", "mandible_left_touches_post"]
missing = [c for c in need_c if c not in cf] + ([] if "cx" in tf else ["tooth cx"])
print("diagnostic cases:", len(cf), "teeth present:", int(tf["present"].sum()), "missing columns:", missing)
sys.exit(1 if missing or len(cf) != 5 else 0)
PY

echo "--- 2. full P/F/S run"
python -m partialref.audit.visibility_features --manifest "$MANIFEST" --workers 8 --tag visibility-features-v2
VIS=$(ls -d results/audit/*-visibility-features-v2 | tail -1)
echo "vis=$VIS"

echo "--- 3a. share of cases with each jaw fully clear of lateral/posterior faces (counts only)"
python - "$VIS" <<'PY'
import sys, pandas as pd
cf = pd.read_parquet(f"{sys.argv[1]}/case_features.parquet")
for jaw in ("mandible", "maxilla"):
    cols = [f"{jaw}_{h}_touches_{f}" for h in ("left", "right") for f in ("lateral", "post")]
    has = cf[f"{jaw}_vox"].fillna(0) > 0
    inside = has & cf.reindex(columns=cols).eq(False).all(axis=1)
    print(jaw, "label present:", int(has.sum()), "fully clear:", int(inside.sum()),
          "by centre:", inside[has].groupby(cf.loc[has, "case_id"].str[0]).mean().round(3).to_dict())
PY
echo "--- 3b. side geometry";  python -m partialref.audit.side_geometry --vis "$VIS"
echo "--- 3c. report-scan fingerprint"; python -m partialref.audit.report_scan_match --parse-run "$PARSE" --vis "$VIS"
echo "--- 3d. laterality"; python -m partialref.propensity.laterality --parse-run "$PARSE" --manifest "$MANIFEST" --vis "$VIS"
echo "done=$(date -Is)"
