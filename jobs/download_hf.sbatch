#!/usr/bin/env bash
#SBATCH --job-name=pr-dl-hf
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Download one Hugging Face model repo into $PROJECT_DIR/models/<local> (gitignored), pinned to the
# commit resolved now, which is recorded in results/env/models_<jobid>.json.
#   sbatch jobs/download_hf.sbatch Qwen/Qwen3-14B Qwen3-14B
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: sbatch jobs/download_hf.sbatch REPO LOCAL_NAME" >&2; exit 2; }
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
export HF_HOME=$PROJECT_DIR/.cache/hf HF_HUB_DISABLE_TELEMETRY=1
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pr_serve
[ -f "$HOME/.secrets/hf.env" ] && { source "$HOME/.secrets/hf.env"; export HF_TOKEN; }
set -u
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) repo=$1"; df -h "$PROJECT_DIR" | tail -1
python - "$1" "models/$2" <<'PY'
import json, os, sys
from huggingface_hub import HfApi, snapshot_download
repo, local = sys.argv[1], sys.argv[2]
info = HfApi().model_info(repo)
path = snapshot_download(repo_id=repo, revision=info.sha, local_dir=local)
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
rec = {repo: {"sha": info.sha, "local_dir": local, "bytes": size}}
json.dump(rec, open(f"results/env/models_{os.environ['SLURM_JOB_ID']}.json", "w"), indent=2)
print("OK", repo, info.sha, f"{size/1e9:.1f} GB")
PY
echo "done=$(date -Is)"
