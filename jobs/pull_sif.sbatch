#!/usr/bin/env bash
#SBATCH --job-name=pr-sif-pull
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# HANDOFF §5.4 Plan B: pull the V2V2R challenge image as a SIF, then record what §5.4 asks for
# (apptainer inspect, pip freeze of both Python envs, /opt/app layout). Pulled by digest (amd64
# manifest of tag odin2026-task1, read from Docker Hub on 2026-11-07) so the SIF is reproducible.
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs
IMAGE="docker://ssingh54/toothfairy4-v2v2r@sha256:b4abf81185e7e315d5b1f43a12a235d34b279b6b8c1150099b76c998799eb571"
SIF_DIR=$PROJECT_DIR/.cache/sif
SIF=$SIF_DIR/toothfairy4-v2v2r_odin2026-task1.sif
INFO=$PROJECT_DIR/results/env/sif_${SLURM_JOB_ID}
export APPTAINER_CACHEDIR=$PROJECT_DIR/.cache/apptainer
export APPTAINER_TMPDIR=$PROJECT_DIR/.cache/apptainer/tmp_${SLURM_JOB_ID}
mkdir -p "$SIF_DIR" "$INFO" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT   # build scratch only; inside .cache (HANDOFF §10)

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"; apptainer --version
df -h "$PROJECT_DIR" | tail -1
if [[ -e "$SIF" ]]; then echo "SIF exists, not overwriting: $SIF"; else
  apptainer pull "$SIF" "$IMAGE"
fi
ls -la "$SIF"; sha256sum "$SIF" | tee "$INFO/sif.sha256"

apptainer inspect "$SIF"                                   > "$INFO/inspect.txt" 2>&1 || true
apptainer exec "$SIF" python -m pip freeze                 > "$INFO/pip_freeze_main.txt" 2>&1 || true
apptainer exec "$SIF" /opt/vlmenv/bin/pip freeze           > "$INFO/pip_freeze_vlmenv.txt" 2>&1 || true
# layout only (names), to locate code/model dirs; no file contents are read
apptainer exec "$SIF" bash -c 'cd /opt/app && find . -maxdepth 3 -not -path "./wheels/*" | sort' > "$INFO/opt_app_tree.txt" 2>&1 || true
apptainer exec "$SIF" bash -c 'ls -la /opt /opt/ml 2>&1; du -sh /opt/* 2>/dev/null' > "$INFO/opt_sizes.txt" 2>&1 || true
grep -iE "^(vllm|torch|transformers|xformers|flash)" "$INFO"/pip_freeze_*.txt || true
df -h "$PROJECT_DIR" | tail -1
echo "done=$(date -Is) info=$INFO"
